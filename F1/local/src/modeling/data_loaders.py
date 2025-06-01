"""
Data Loaders for F1 Tyre Strategy RNN Model
===========================================

This script provides PyTorch Dataset and DataLoader classes for loading
the preprocessed sequence data for model training and evaluation.

Key Components:
- F1SequenceDataset: Custom Dataset for loading .npy sequence files.
- create_dataloaders: Factory function to create train, validation, and test DataLoaders.
- check_data_distribution: Utility to analyze class distribution in a DataLoader.

Adapted from 'Vincenzo/dataset/models/data_loaders.py'.
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from typing import Tuple, Dict, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class F1SequenceDataset(Dataset):
    """
    Custom PyTorch Dataset for F1 tire strategy sequences.
    Loads preprocessed .npy files (X, y_change, y_type).
    """
    def __init__(
        self,
        model_input_dir: Path,
        split: str = 'train',  # 'train', 'val', or 'test'
        augment_positive_samples: bool = False,
        augmentation_factor: int = 2, # How many times to repeat positive samples
        noise_level: float = 0.01 # Std dev of Gaussian noise for augmentation
    ):
        self.model_input_dir = Path(model_input_dir)
        self.split = split
        self.augment_positive = augment_positive_samples and self.split == 'train'
        self.augmentation_factor = augmentation_factor
        self.noise_level = noise_level

        self._load_data()

        if self.augment_positive:
            self._prepare_augmented_data()

    def _load_data(self):
        """Loads sequences and targets from .npy files."""
        try:
            X_path = self.model_input_dir / f"X_{self.split}.npy"
            y_change_path = self.model_input_dir / f"y_change_{self.split}.npy"
            y_type_path = self.model_input_dir / f"y_type_{self.split}.npy"

            self.X_original = np.load(X_path)
            self.y_change_original = np.load(y_change_path)
            self.y_type_original = np.load(y_type_path)

            logger.info(f"Loaded {self.split} data: "
                        f"X shape: {self.X_original.shape}, "
                        f"y_change shape: {self.y_change_original.shape}, "
                        f"y_type shape: {self.y_type_original.shape}")
            
            if self.X_original.shape[0] == 0:
                 logger.warning(f"Loaded {self.split} data is empty. X shape: {self.X_original.shape}")


        except FileNotFoundError as e:
            logger.error(f"Required data file not found for split '{self.split}' in {self.model_input_dir}: {e}")
            # Fallback to empty arrays to prevent crashing downstream if a split is legitimately empty (e.g. val set)
            self.X_original = np.array([]) 
            self.y_change_original = np.array([])
            self.y_type_original = np.array([])
            # Ensure X_original has at least 3 dimensions for consistency if it's empty
            if self.X_original.ndim == 1:
                 # Assuming sequence_length and num_features can be inferred or are fixed
                 # This part is tricky if we don't know num_features.
                 # For now, let's assume if it's empty, it's (0, seq_len, num_features)
                 # This will be handled by __len__ and __getitem__ returning empty or erroring.
                 # A better way is to get num_features from feature_columns.joblib if needed here.
                 # For now, if X is empty, len will be 0.
                 pass


    def _prepare_augmented_data(self):
        """Prepares data for augmentation if enabled."""
        if self.X_original.shape[0] == 0:
            logger.warning("Cannot perform augmentation on empty dataset.")
            self.X_augmented = self.X_original
            self.y_change_augmented = self.y_change_original
            self.y_type_augmented = self.y_type_original
            return

        positive_indices = np.where(self.y_change_original == 1)[0]
        if len(positive_indices) == 0:
            logger.info("No positive samples to augment.")
            self.X_augmented = self.X_original
            self.y_change_augmented = self.y_change_original
            self.y_type_augmented = self.y_type_original
            return

        X_pos = self.X_original[positive_indices]
        y_change_pos = self.y_change_original[positive_indices]
        y_type_pos = self.y_type_original[positive_indices]

        augmented_X_list = [self.X_original]
        augmented_y_change_list = [self.y_change_original]
        augmented_y_type_list = [self.y_type_original]

        for _ in range(self.augmentation_factor - 1):
            noise = np.random.normal(0, self.noise_level, X_pos.shape).astype(np.float32)
            augmented_X_list.append(X_pos + noise)
            augmented_y_change_list.append(y_change_pos)
            augmented_y_type_list.append(y_type_pos)
        
        self.X_augmented = np.concatenate(augmented_X_list, axis=0)
        self.y_change_augmented = np.concatenate(augmented_y_change_list, axis=0)
        self.y_type_augmented = np.concatenate(augmented_y_type_list, axis=0)
        
        logger.info(f"Augmentation complete for {self.split} set. "
                    f"New size: {len(self.X_augmented)}. "
                    f"Original positive samples: {len(positive_indices)}, "
                    f"Total positive after augmentation: {np.sum(self.y_change_augmented == 1)}")

    def __len__(self) -> int:
        if self.augment_positive:
            return len(self.X_augmented)
        return len(self.X_original)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.augment_positive:
            X_data = self.X_augmented
            y_change_data = self.y_change_augmented
            y_type_data = self.y_type_augmented
        else:
            X_data = self.X_original
            y_change_data = self.y_change_original
            y_type_data = self.y_type_original
        
        if idx >= len(X_data): # Should not happen with correct __len__
            raise IndexError("Index out of bounds")

        sequence = torch.from_numpy(X_data[idx].astype(np.float32))
        target_change = torch.tensor(y_change_data[idx], dtype=torch.long)
        target_type = torch.tensor(y_type_data[idx], dtype=torch.long)
        
        return sequence, target_change, target_type

    def get_class_weights_for_sampler(self) -> Optional[torch.Tensor]:
        """Calculates class weights for WeightedRandomSampler for the 'tire_change' task."""
        if self.X_original.shape[0] == 0: return None

        target_data = self.y_change_augmented if self.augment_positive else self.y_change_original
        if len(target_data) == 0: return None

        class_counts = np.bincount(target_data)
        if len(class_counts) < 2 : # Only one class present
            logger.warning("Only one class present in target data. Weighted sampling might not be effective.")
            return None
            
        class_weights = 1. / class_counts
        # Normalize weights to sum to 1 (optional, but can help)
        # class_weights = class_weights / np.sum(class_weights) 
        
        sample_weights = class_weights[target_data]
        return torch.from_numpy(sample_weights).float()


def create_dataloaders(
    model_input_dir: str,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    augment_train_positive_samples: bool = True,
    augmentation_factor: int = 2,
    use_weighted_sampler_for_train: bool = True
) -> Dict[str, DataLoader]:
    """Creates DataLoaders for train, validation, and test sets."""
    
    datasets = {}
    for split in ['train', 'val', 'test']:
        augment_this_split = augment_train_positive_samples and split == 'train'
        datasets[split] = F1SequenceDataset(
            model_input_dir=Path(model_input_dir),
            split=split,
            augment_positive_samples=augment_this_split,
            augmentation_factor=augmentation_factor
        )

    dataloaders = {}
    for split, dataset in datasets.items():
        if len(dataset) == 0: # Handle empty datasets (e.g., val set if only 2 years of data)
            logger.warning(f"Dataset for split '{split}' is empty. DataLoader will also be empty.")
            # Create an 'empty' DataLoader by not iterating or by returning a special object if needed
            # For now, this will result in a DataLoader that yields nothing.
            dataloaders[split] = DataLoader(dataset, batch_size=batch_size) # Will be empty
            continue

        sampler = None
        shuffle = True
        if split == 'train' and use_weighted_sampler_for_train:
            weights = dataset.get_class_weights_for_sampler()
            if weights is not None:
                sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
                shuffle = False # Sampler handles shuffling logic
                logger.info(f"Using WeightedRandomSampler for '{split}' set.")
            else:
                logger.warning(f"Could not get weights for sampler for '{split}' set. Using standard shuffling.")
        
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == 'train') # Drop last only for training set
        )
    logger.info(f"DataLoaders created. Train batches: {len(dataloaders['train']) if 'train' in dataloaders else 0}, "
                f"Val batches: {len(dataloaders['val']) if 'val' in dataloaders else 0}, "
                f"Test batches: {len(dataloaders['test']) if 'test' in dataloaders else 0}.")
    return dataloaders

def check_data_loader_distribution(dataloader: DataLoader, name: str):
    """Analyzes and logs class distribution for a DataLoader."""
    if len(dataloader.dataset) == 0:
        logger.info(f"DataLoader '{name}' is empty. Skipping distribution check.")
        return

    logger.info(f"Analyzing distribution for DataLoader: {name}")
    change_counts = {0: 0, 1: 0}
    type_counts = {} # For y_type when change is 1
    total_samples = 0

    for _, y_change, y_type in dataloader:
        for i in range(len(y_change)):
            change_val = y_change[i].item()
            change_counts[change_val] += 1
            if change_val == 1:
                type_val = y_type[i].item()
                type_counts[type_val] = type_counts.get(type_val, 0) + 1
        total_samples += len(y_change)
    
    if total_samples > 0:
        logger.info(f"'{name}' - Total samples processed: {total_samples}")
        logger.info(f"  Tire Change (0/1): {change_counts}")
        if change_counts[1] > 0:
            logger.info(f"  Positive class (1) percentage: {change_counts[1]/total_samples*100:.2f}%")
        logger.info(f"  Tire Type (when change=1): {type_counts}")
    else:
        logger.info(f"'{name}' - No samples processed (empty DataLoader).")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("--- Testing DataLoaders ---")
    
    # Assume model_input files (X_train.npy etc.) are in F1/local/drive/model_input/
    # This path is relative to where the script is run from.
    # If running this file directly, it's F1/local/src/modeling/
    # So, model_input_dir should be ../../drive/model_input
    
    current_script_path = Path(__file__).resolve()
    # project_root_f1_local = current_script_path.parent.parent.parent # F1/local/
    # model_input_dir_test = project_root_f1_local / "drive/model_input"
    
    # Using the path from the main training script context for consistency
    model_input_dir_test = Path("F1/local/drive/model_input")


    if not model_input_dir_test.exists() or not any(model_input_dir_test.iterdir()):
        logger.warning(f"Test model input directory {model_input_dir_test} is missing or empty.")
        logger.warning("Skipping DataLoader test with real data. Create dummy data if needed or run prepare_model_data.py.")
    else:
        try:
            test_dataloaders = create_dataloaders(
                model_input_dir=str(model_input_dir_test),
                batch_size=4, # Small batch for testing
                augment_train_positive_samples=True,
                use_weighted_sampler_for_train=True
            )
            
            if len(test_dataloaders['train'].dataset) > 0:
                logger.info("Checking train_loader batch:")
                X_batch, y_change_batch, y_type_batch = next(iter(test_dataloaders['train']))
                logger.info(f"  X_batch shape: {X_batch.shape}")
                logger.info(f"  y_change_batch shape: {y_change_batch.shape}, sample: {y_change_batch[:2]}")
                logger.info(f"  y_type_batch shape: {y_type_batch.shape}, sample: {y_type_batch[:2]}")
                check_data_loader_distribution(test_dataloaders['train'], "Train Loader")
            else:
                logger.info("Train loader is empty, skipping batch check.")

            if len(test_dataloaders['val'].dataset) > 0:
                 check_data_loader_distribution(test_dataloaders['val'], "Validation Loader")
            else:
                logger.info("Validation loader is empty.")

            logger.info("✅ DataLoader testing completed.")
        except Exception as e:
            logger.error(f"Error during DataLoader test: {e}", exc_info=True)
