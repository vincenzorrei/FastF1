"""
Training Utilities for F1 Tyre Strategy RNN Model
=================================================

This script provides utility classes and functions for training the RNN model,
including:
- EarlyStopping: Stops training if validation metric doesn't improve.
- TrainingMetrics: Tracks and computes metrics during training/validation.
- find_optimal_threshold: Optimizes decision threshold based on PR curve.
- ModelTrainer: Main class orchestrating the training and validation loops.
- create_trainer_components: Factory to set up trainer, optimizer, loss, scheduler.

Adapted from 'Vincenzo/dataset/models/training_utils.py'.
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score, confusion_matrix, precision_recall_curve
)
# import matplotlib.pyplot as plt # Matplotlib not used in original for saving, can be added if needed
import logging

# Assuming model_def contains LSTMTirePredictor and CombinedLoss
# Adjust import based on final project structure if this file is moved
from .model_def import CombinedLoss 

logger = logging.getLogger(__name__)

class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    Monitors a validation metric and stops training if it doesn't improve.
    """
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'max',  # 'max' for F1-score, 'min' for loss
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_metric = float('-inf') if mode == 'max' else float('inf')
        self.patience_counter = 0
        self.best_weights = None
        self.should_stop = False
        
    def step(self, metric: float, model: nn.Module) -> bool:
        """Checks if training should stop based on the current metric."""
        improved = False
        if self.mode == 'max':
            if metric > self.best_metric + self.min_delta:
                improved = True
        else: # mode == 'min'
            if metric < self.best_metric - self.min_delta:
                improved = True
            
        if improved:
            self.best_metric = metric
            self.patience_counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if self.verbose:
                logger.info(f"EarlyStopping: New best metric: {self.best_metric:.4f}")
        else:
            self.patience_counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping: Patience counter: {self.patience_counter}/{self.patience}")
            
        if self.patience_counter >= self.patience:
            self.should_stop = True
            if self.verbose:
                logger.info("EarlyStopping: Stopping training.")
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                model.to(next(model.parameters()).device) # Ensure model is on correct device
                if self.verbose:
                    logger.info(f"EarlyStopping: Restored best model weights (best metric: {self.best_metric:.4f}).")
                
        return self.should_stop

class TrainingMetricsTracker:
    """Calculates and tracks metrics for training and validation phases."""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.predictions_proba = [] # Store raw probabilities for PR/ROC AUC
        self.targets_all = []
        self.losses_all = []
        
    def update(self, predictions_proba: torch.Tensor, targets: torch.Tensor, loss_value: float):
        self.predictions_proba.extend(predictions_proba.detach().cpu().numpy().flatten())
        self.targets_all.extend(targets.detach().cpu().numpy().flatten())
        self.losses_all.append(loss_value)
        
    def compute_metrics(self, threshold: float = 0.5) -> Dict[str, float]:
        if not self.predictions_proba:
            return {} # Return empty if no data
            
        preds_proba_np = np.array(self.predictions_proba)
        targets_np = np.array(self.targets_all)
        
        binary_preds = (preds_proba_np >= threshold).astype(int)
        
        metrics = {
            'loss': np.mean(self.losses_all) if self.losses_all else 0.0,
            'f1_score': f1_score(targets_np, binary_preds, zero_division=0),
            'precision': precision_score(targets_np, binary_preds, zero_division=0),
            'recall': recall_score(targets_np, binary_preds, zero_division=0),
        }
        if len(np.unique(targets_np)) > 1: # ROC AUC and PR AUC require at least two classes
            metrics['roc_auc'] = roc_auc_score(targets_np, preds_proba_np)
            metrics['pr_auc'] = average_precision_score(targets_np, preds_proba_np)
        else:
            metrics['roc_auc'] = 0.0
            metrics['pr_auc'] = 0.0
        
        try: # Confusion matrix can fail if only one class predicted/present
            cm = confusion_matrix(targets_np, binary_preds)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            elif cm.shape == (1,1) and targets_np[0] == 0 : # Only negatives
                tn, fp, fn, tp = cm[0][0],0,0,0
            elif cm.shape == (1,1) and targets_np[0] == 1 : # Only positives
                tn, fp, fn, tp = 0,0,0,cm[0][0]
            else: # Default if CM is not 2x2 (e.g. all one class)
                tn, fp, fn, tp = 0,0,0,0
                logger.warning(f"Confusion matrix has unexpected shape: {cm.shape}. CM components might be inaccurate.")

            metrics.update({
                'true_positives': int(tp), 'true_negatives': int(tn),
                'false_positives': int(fp), 'false_negatives': int(fn),
                'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            })
        except Exception as e:
            logger.error(f"Error calculating confusion matrix components: {e}")

        return metrics

def find_optimal_threshold_on_pr_curve(
    predictions_proba: np.ndarray,
    targets: np.ndarray,
    target_recall: Optional[float] = None, # If None, maximizes F1 without recall constraint
    metric_to_optimize: str = 'f1' # 'f1' or 'precision'
) -> Tuple[float, Dict[str, float]]:
    """Finds an optimal threshold based on the Precision-Recall curve."""
    precisions, recalls, thresholds = precision_recall_curve(targets, predictions_proba)
    
    # thresholds array is 1 element shorter than precisions/recalls
    # Use precisions[:-1] and recalls[:-1] to match thresholds
    
    if target_recall is not None:
        valid_indices = np.where(recalls[:-1] >= target_recall)[0]
        if not len(valid_indices):
            logger.warning(f"No threshold achieves target recall >= {target_recall}. Maximizing {metric_to_optimize} over all thresholds.")
            # Fallback: consider all thresholds
            valid_precisions = precisions[:-1]
            valid_recalls = recalls[:-1]
            valid_thresholds = thresholds
            original_indices = np.arange(len(thresholds))
        else:
            valid_precisions = precisions[valid_indices]
            valid_recalls = recalls[valid_indices]
            valid_thresholds = thresholds[valid_indices]
            original_indices = valid_indices # Keep track of original indices for threshold lookup
    else: # No recall constraint, use all thresholds
        valid_precisions = precisions[:-1]
        valid_recalls = recalls[:-1]
        valid_thresholds = thresholds
        original_indices = np.arange(len(thresholds))

    if not len(valid_thresholds): # Should not happen if target_recall is None or fallback used
         logger.warning("No valid thresholds found. Defaulting to 0.5.")
         optimal_threshold = 0.5
    else:
        if metric_to_optimize == 'f1':
            f1_scores = 2 * (valid_precisions * valid_recalls) / (valid_precisions + valid_recalls + 1e-8)
            best_idx_in_valid = np.argmax(f1_scores)
        elif metric_to_optimize == 'precision':
            best_idx_in_valid = np.argmax(valid_precisions)
        else:
            raise ValueError(f"Unsupported metric_to_optimize: {metric_to_optimize}")
        optimal_threshold = valid_thresholds[best_idx_in_valid]

    # Calculate metrics at this optimal threshold
    binary_preds_at_optimal = (predictions_proba >= optimal_threshold).astype(int)
    metrics_at_threshold = {
        'threshold': float(optimal_threshold),
        'f1_score': f1_score(targets, binary_preds_at_optimal, zero_division=0),
        'precision': precision_score(targets, binary_preds_at_optimal, zero_division=0),
        'recall': recall_score(targets, binary_preds_at_optimal, zero_division=0),
    }
    if len(np.unique(targets)) > 1:
        metrics_at_threshold['roc_auc'] = roc_auc_score(targets, predictions_proba)
        metrics_at_threshold['pr_auc'] = average_precision_score(targets, predictions_proba)
    else:
        metrics_at_threshold['roc_auc'] = 0.0
        metrics_at_threshold['pr_auc'] = 0.0
        
    return float(optimal_threshold), metrics_at_threshold


class ModelTrainer:
    """Main trainer class for the F1 tire strategy model."""
    def __init__(
        self, model: nn.Module, train_loader, val_loader, optimizer, loss_fn_details: dict,
        device: str, log_dir: str, checkpoint_dir: str,
        config_training_params: dict
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        
        self.loss_fn = CombinedLoss( # Instantiate CombinedLoss using details from config
            alpha=loss_fn_details.get('alpha', 0.9),
            beta=loss_fn_details.get('beta', 0.1),
            pos_weight_tire_change=loss_fn_details.get('pos_weight_tire_change'),
            device=device
        )
        
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        self.current_epoch = 0
        self.best_val_f1 = 0.0
        self.best_threshold_val = 0.5 # Threshold optimized on validation set
        self.training_history = {'train': [], 'validation': []}

        self.target_recall = config_training_params.get('target_recall_constraint') # e.g. 0.8
        self.early_stopping_patience = config_training_params.get('early_stopping', {}).get('patience', 10)
        self.early_stopping_min_delta = config_training_params.get('early_stopping', {}).get('min_delta', 0.001)
        
        self.early_stopper = EarlyStopping(
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta,
            mode='max', restore_best_weights=True, verbose=True
        )
        logger.info(f"Trainer initialized. Device: {device}, LogDir: {log_dir}, CheckpointDir: {checkpoint_dir}")

    def _run_epoch(self, data_loader, is_training: bool) -> Tuple[Dict[str, float], Optional[float]]:
        if is_training:
            self.model.train()
        else:
            self.model.eval()

        metrics_tracker = TrainingMetricsTracker()
        epoch_start_time = time.time()

        for batch_idx, (sequences, targets_change, targets_type) in enumerate(data_loader):
            sequences = sequences.to(self.device)
            targets_change = targets_change.to(self.device)
            targets_type = targets_type.to(self.device)

            if is_training:
                self.optimizer.zero_grad()
            
            with torch.set_grad_enabled(is_training):
                predictions = self.model(sequences, return_sequences=False) # Use last step output
                loss_components = self.loss_fn(predictions, targets_change, targets_type)
                total_loss = loss_components['total_loss']

            if is_training:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            # For metrics, use probabilities for the primary task (tire_change)
            preds_proba_change = torch.sigmoid(predictions['tire_change_logits'])
            metrics_tracker.update(preds_proba_change, targets_change, total_loss.item())

            if is_training and (batch_idx % 50 == 0 or batch_idx == len(data_loader) -1 ):
                logger.debug(f"Epoch {self.current_epoch+1} Batch {batch_idx+1}/{len(data_loader)}: Loss={total_loss.item():.4f}")
        
        # For validation, find optimal threshold and compute metrics with it
        # For training, use the current best_threshold_val from previous validation
        current_threshold = self.best_threshold_val if is_training else 0.5 # Default for first val
        
        if not is_training and len(metrics_tracker.targets_all) > 0 : # Validation phase
            optimal_threshold_val, epoch_metrics = find_optimal_threshold_on_pr_curve(
                np.array(metrics_tracker.predictions_proba),
                np.array(metrics_tracker.targets_all),
                target_recall=self.target_recall 
            )
            # Recompute metrics with this new optimal threshold for consistent reporting
            epoch_metrics = metrics_tracker.compute_metrics(threshold=optimal_threshold_val)
            epoch_metrics['threshold'] = optimal_threshold_val # Add it to the dict
        else: # Training phase or empty val loader
            epoch_metrics = metrics_tracker.compute_metrics(threshold=current_threshold)
            optimal_threshold_val = None # No new threshold from training epoch

        epoch_duration = time.time() - epoch_start_time
        epoch_metrics['duration'] = epoch_duration
        return epoch_metrics, optimal_threshold_val

    def train_model(self, num_epochs: int, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None, save_freq: int = 5):
        logger.info(f"Starting training for {num_epochs} epochs.")
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training epoch
            train_metrics, _ = self._run_epoch(self.train_loader, is_training=True)
            self.training_history['train'].append(train_metrics)
            
            # Validation epoch
            if self.val_loader:
                val_metrics, new_optimal_threshold_val = self._run_epoch(self.val_loader, is_training=False)
                if new_optimal_threshold_val is not None: # Update best_threshold if validation produced one
                    self.best_threshold_val = new_optimal_threshold_val 
                self.training_history['validation'].append(val_metrics)
                
                current_val_f1 = val_metrics.get('f1_score', 0.0)
                if current_val_f1 > self.best_val_f1:
                    self.best_val_f1 = current_val_f1
                    self._save_checkpoint(epoch, is_best=True) # Save best model based on val F1
                
                if scheduler:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(current_val_f1)
                    else:
                        scheduler.step() # For schedulers like StepLR
                
                self._log_epoch_summary(epoch, train_metrics, val_metrics)
                if self.early_stopper.step(current_val_f1, self.model):
                    break 
            else: # No validation loader
                self._log_epoch_summary(epoch, train_metrics, None)
                if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                     scheduler.step()


            if (epoch + 1) % save_freq == 0 and not (val_metrics and current_val_f1 > self.best_val_f1):
                self._save_checkpoint(epoch, is_best=False) # Regular checkpoint

        self.writer.close()
        logger.info("Training finished.")
        if self.early_stopper.should_stop:
            logger.info(f"Early stopping criterion met. Best validation F1: {self.early_stopper.best_metric:.4f}")
        
        self._save_training_summary_json()
        return self.training_history

    def _log_epoch_summary(self, epoch, train_metrics, val_metrics):
        lr = self.optimizer.param_groups[0]['lr']
        log_str = (f"Epoch {epoch+1:03d} | Train Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1_score']:.4f}, Recall: {train_metrics['recall']:.4f} | LR: {lr:.2e}")
        self.writer.add_scalar("Loss/train", train_metrics['loss'], epoch)
        self.writer.add_scalar("F1/train", train_metrics['f1_score'], epoch)
        self.writer.add_scalar("Recall/train", train_metrics['recall'], epoch)
        self.writer.add_scalar("LearningRate", lr, epoch)

        if val_metrics:
            log_str += (f" | Val Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1_score']:.4f}, Recall: {val_metrics['recall']:.4f}, Thr: {val_metrics.get('threshold', self.best_threshold_val):.2f}")
            self.writer.add_scalar("Loss/validation", val_metrics['loss'], epoch)
            self.writer.add_scalar("F1/validation", val_metrics['f1_score'], epoch)
            self.writer.add_scalar("Recall/validation", val_metrics['recall'], epoch)
            self.writer.add_scalar("Threshold/validation", val_metrics.get('threshold', self.best_threshold_val), epoch)
        logger.info(log_str)

    def _save_checkpoint(self, epoch: int, is_best: bool):
        state = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_f1': self.best_val_f1,
            'best_threshold_val': self.best_threshold_val
        }
        filename = "best_model.pth" if is_best else f"checkpoint_epoch_{epoch+1}.pth"
        filepath = self.checkpoint_dir / filename
        torch.save(state, filepath)
        logger.info(f"Saved {'best ' if is_best else ''}checkpoint to {filepath}")

    def _save_training_summary_json(self):
        summary_path = self.log_dir / "training_summary.json"
        summary_data = {
            "best_validation_f1": self.best_val_f1,
            "best_threshold_on_validation": self.best_threshold_val,
            "epochs_trained": self.current_epoch +1,
            "training_history": self.training_history
        }
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=4, default=lambda o: '<not serializable>')
        logger.info(f"Training summary saved to {summary_path}")


def create_trainer_components(model: nn.Module, config: dict, device: str) -> Tuple[ModelTrainer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
    """Factory function to create optimizer, loss_fn, scheduler, and ModelTrainer."""
    
    train_cfg = config.get('training', {})
    loss_cfg = config.get('model', {}).get('loss', {})

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get('learning_rate', 0.001),
        weight_decay=train_cfg.get('weight_decay', 1e-4)
    )

    # Scheduler (optional)
    scheduler_cfg = train_cfg.get('scheduler', {})
    scheduler = None
    if scheduler_cfg.get('type') == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', 
            factor=scheduler_cfg.get('factor', 0.1), 
            patience=scheduler_cfg.get('patience', 10),
            min_lr=scheduler_cfg.get('min_lr', 1e-6)
            # verbose=True # Removed as it caused TypeError
        )
    elif scheduler_cfg.get('type') == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=scheduler_cfg.get('step_size', 10),
            gamma=scheduler_cfg.get('gamma', 0.1),
            verbose=True
        )
    
    # Loss function details (passed to ModelTrainer which instantiates CombinedLoss)
    loss_fn_details = {
        'alpha': loss_cfg.get('alpha', 0.9),
        'beta': loss_cfg.get('beta', 0.1),
        'pos_weight_tire_change': loss_cfg.get('pos_weight_tire_change') 
    }
    
    # Paths from config or defaults
    data_paths = config.get('data', {})
    log_dir = Path("F1/local") / train_cfg.get('log_dir', 'drive/logs/training_logs')
    checkpoint_dir = Path("F1/local") / train_cfg.get('checkpoint_dir', 'drive/models')


    # DataLoaders would be created here or passed in
    # For now, assuming they are passed to the main training script that uses this.
    # This function focuses on creating components related to the trainer object itself.

    # The ModelTrainer will be instantiated in the main training script.
    # This function provides the components it needs.
    # Returning optimizer and scheduler, loss_fn_details.
    # The trainer itself will be created in the main script.
    
    return optimizer, scheduler, loss_fn_details, log_dir, checkpoint_dir
