"""
Model Evaluation Utilities for F1 Tyre Strategy RNN
===================================================

This script provides tools for evaluating the performance of the trained
F1 tyre strategy prediction model. It includes:
- Comprehensive metrics calculation (F1, Precision, Recall, ROC AUC, etc.).
- Evaluation on primary (tire change) and secondary (tire type) tasks.
- Plotting of evaluation curves (ROC, Precision-Recall, Confusion Matrix).
- Threshold sensitivity analysis.
- Saving of evaluation reports.

Adapted from 'Vincenzo/dataset/models/evaluation.py'.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score
)
from pathlib import Path
import json
import logging
import sys # Import the sys module

# Assuming model_def contains LSTMTirePredictor
# from .model_def import LSTMTirePredictor # Not directly needed if model instance is passed

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive evaluator for the multi-task F1 tire strategy model.
    """
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        decision_threshold: float = 0.5 # Threshold for tire_change_task
    ):
        self.model = model.to(device)
        self.device = device
        self.decision_threshold = decision_threshold
        self.model.eval() # Ensure model is in evaluation mode
        logger.info(f"ModelEvaluator initialized. Device: {device}, Decision Threshold: {decision_threshold:.4f}")

    @torch.no_grad() # Disable gradient calculations for evaluation
    def evaluate_on_dataloader(
        self,
        data_loader: torch.utils.data.DataLoader,
        split_name: str = "Test"
    ) -> Dict:
        """
        Evaluates the model on a given DataLoader.

        Args:
            data_loader: DataLoader for the dataset split to evaluate.
            split_name: Name of the data split (e.g., "Validation", "Test").

        Returns:
            A dictionary containing detailed evaluation results.
        """
        logger.info(f"Starting evaluation on {split_name} set...")
        
        all_change_logits = []
        all_change_probs = []
        all_change_preds_at_threshold = []
        all_change_targets = []
        
        all_type_logits = []
        all_type_probs = []
        all_type_preds = []
        all_type_targets = []

        for batch_idx, (sequences, targets_change, targets_type) in enumerate(data_loader):
            sequences = sequences.to(self.device)
            # Targets are kept on CPU for numpy conversion later, or move to device if loss calculated here
            
            model_outputs = self.model(sequences, return_sequences=False) # Get last step output

            # --- Tire Change Task (Primary) ---
            logits_change = model_outputs['tire_change_logits'].squeeze(-1) # Shape: (batch_size)
            probs_change = torch.sigmoid(logits_change)
            preds_at_threshold = (probs_change >= self.decision_threshold).long()

            all_change_logits.extend(logits_change.cpu().numpy())
            all_change_probs.extend(probs_change.cpu().numpy())
            all_change_preds_at_threshold.extend(preds_at_threshold.cpu().numpy())
            all_change_targets.extend(targets_change.numpy()) # Assuming targets are on CPU

            # --- Tire Type Task (Secondary) ---
            logits_type = model_outputs['tire_type_logits'] # Shape: (batch_size, num_compounds)
            probs_type = torch.softmax(logits_type, dim=-1)
            preds_type = torch.argmax(probs_type, dim=-1)

            all_type_logits.extend(logits_type.cpu().numpy())
            all_type_probs.extend(probs_type.cpu().numpy())
            all_type_preds.extend(preds_type.cpu().numpy())
            all_type_targets.extend(targets_type.numpy()) # Assuming targets are on CPU

            if batch_idx % 100 == 0 and batch_idx > 0 :
                logger.debug(f"Evaluated {batch_idx}/{len(data_loader)} batches for {split_name} set.")
        
        # Convert lists to numpy arrays
        np_change_logits = np.array(all_change_logits)
        np_change_probs = np.array(all_change_probs)
        np_change_preds_at_threshold = np.array(all_change_preds_at_threshold)
        np_change_targets = np.array(all_change_targets)
        
        np_type_logits = np.array(all_type_logits)
        np_type_probs = np.array(all_type_probs)
        np_type_preds = np.array(all_type_preds)
        np_type_targets = np.array(all_type_targets)

        # Calculate metrics for the primary task (tire change)
        primary_task_metrics = self._calculate_binary_classification_metrics(
            targets=np_change_targets,
            probabilities=np_change_probs,
            predictions_at_threshold=np_change_preds_at_threshold
        )
        
        # Calculate metrics for the secondary task (tire type)
        secondary_task_metrics = self._calculate_multiclass_classification_metrics(
            targets=np_type_targets,
            predictions=np_type_preds,
            probabilities=np_type_probs, # For potential future use (e.g. top-k accuracy)
            condition_mask=(np_change_targets == 1) # Evaluate only where a tire change occurred
        )
        
        logger.info(f"Evaluation completed for {split_name} set. "
                    f"Primary Task F1: {primary_task_metrics.get('f1_score', 0.0):.4f}, "
                    f"Secondary Task Accuracy: {secondary_task_metrics.get('accuracy', 0.0):.4f}")

        return {
            "split_name": split_name,
            "decision_threshold": self.decision_threshold,
            "primary_task_metrics": primary_task_metrics,
            "secondary_task_metrics": secondary_task_metrics,
            "raw_outputs": {
                "change_logits": np_change_logits,
                "change_probabilities": np_change_probs,
                "change_predictions_at_threshold": np_change_preds_at_threshold,
                "change_targets": np_change_targets,
                "type_logits": np_type_logits,
                "type_probabilities": np_type_probs,
                "type_predictions": np_type_preds,
                "type_targets": np_type_targets,
            }
        }

    def _calculate_binary_classification_metrics(self, targets, probabilities, predictions_at_threshold) -> Dict:
        if len(targets) == 0: return {}
        metrics = {
            'f1_score': f1_score(targets, predictions_at_threshold, zero_division=0),
            'precision': precision_score(targets, predictions_at_threshold, zero_division=0),
            'recall': recall_score(targets, predictions_at_threshold, zero_division=0),
            'accuracy': np.mean(targets == predictions_at_threshold)
        }
        if len(np.unique(targets)) > 1:
            metrics['roc_auc'] = roc_auc_score(targets, probabilities)
            metrics['pr_auc'] = average_precision_score(targets, probabilities)
        else:
            metrics['roc_auc'] = 0.0; metrics['pr_auc'] = 0.0
        
        cm = confusion_matrix(targets, predictions_at_threshold)
        if cm.shape == (2,2): tn, fp, fn, tp = cm.ravel()
        else: tn, fp, fn, tp = (cm[0,0] if targets[0]==0 and len(targets)>0 else 0), 0,0, (cm[0,0] if targets[0]==1 and len(targets)>0 else 0) if cm.size==1 else (0,0,0,0)

        metrics.update({
            'true_positives': int(tp), 'true_negatives': int(tn),
            'false_positives': int(fp), 'false_negatives': int(fn),
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        })
        metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2.0 if (tn + fp + tp + fn) > 0 else 0.0
        return metrics

    def _calculate_multiclass_classification_metrics(self, targets, predictions, probabilities, condition_mask) -> Dict:
        if not np.any(condition_mask):
            return {"accuracy": 0.0, "num_samples": 0, "classification_report": {}, "message": "No samples met condition for evaluation."}

        targets_eval = targets[condition_mask]
        preds_eval = predictions[condition_mask]
        # probs_eval = probabilities[condition_mask] # For more detailed metrics if needed

        if len(targets_eval) == 0:
             return {"accuracy": 0.0, "num_samples": 0, "classification_report": {}, "message": "No samples after applying condition mask."}

        accuracy = np.mean(preds_eval == targets_eval)
        report_dict = classification_report(targets_eval, preds_eval, output_dict=True, zero_division=0, labels=np.unique(np.concatenate((targets_eval, preds_eval))))
        
        return {
            "accuracy": accuracy,
            "num_samples": len(targets_eval),
            "classification_report": report_dict
        }

    def plot_evaluation_graphics(self, eval_results: Dict, save_dir: Optional[Path] = None):
        """Generates and optionally saves key evaluation plots."""
        raw = eval_results["raw_outputs"]
        split_name = eval_results["split_name"]
        threshold = eval_results["decision_threshold"]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Evaluation Plots for {split_name} (Threshold: {threshold:.3f})", fontsize=16)

        # ROC Curve
        if len(np.unique(raw['change_targets'])) > 1:
            fpr, tpr, _ = roc_curve(raw['change_targets'], raw['change_probabilities'])
            roc_auc_val = auc(fpr, tpr)
            axes[0,0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_val:.3f})')
        axes[0,0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0,0].set_xlim([0.0, 1.0]); axes[0,0].set_ylim([0.0, 1.05])
        axes[0,0].set_xlabel('False Positive Rate'); axes[0,0].set_ylabel('True Positive Rate')
        axes[0,0].set_title('Receiver Operating Characteristic (ROC)'); axes[0,0].legend(loc="lower right"); axes[0,0].grid(True)

        # Precision-Recall Curve
        if len(np.unique(raw['change_targets'])) > 1:
            precision, recall, _ = precision_recall_curve(raw['change_targets'], raw['change_probabilities'])
            pr_auc_val = average_precision_score(raw['change_targets'], raw['change_probabilities'])
            axes[0,1].plot(recall, precision, color='blueviolet', lw=2, label=f'PR curve (area = {pr_auc_val:.3f})')
        axes[0,1].set_xlabel('Recall'); axes[0,1].set_ylabel('Precision')
        axes[0,1].set_title('Precision-Recall Curve'); axes[0,1].legend(loc="best"); axes[0,1].grid(True)

        # Confusion Matrix (Tire Change)
        cm_change = confusion_matrix(raw['change_targets'], raw['change_predictions_at_threshold'])
        sns.heatmap(cm_change, annot=True, fmt='d', cmap='Blues', ax=axes[1,0], cbar=False)
        axes[1,0].set_title('Confusion Matrix (Tire Change)'); axes[1,0].set_xlabel('Predicted'); axes[1,0].set_ylabel('Actual')

        # Prediction Probabilities Distribution (Tire Change)
        sns.histplot(data=pd.DataFrame({'Probs': raw['change_probabilities'], 'Target': raw['change_targets']}), 
                     x='Probs', hue='Target', multiple='stack', bins=50, ax=axes[1,1])
        axes[1,1].axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.3f})')
        axes[1,1].set_title('Prediction Probabilities (Tire Change)'); axes[1,1].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            plot_path = save_dir / f"evaluation_plots_{split_name.lower()}.png"
            plt.savefig(plot_path); logger.info(f"Saved evaluation plots to {plot_path}")
        plt.show() # Display plot

def generate_and_save_evaluation_report(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str,
    decision_threshold: float,
    report_dir: Path,
    split_name: str = "Test"
):
    """Generates, saves, and returns a full evaluation report."""
    report_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator = ModelEvaluator(model, device, decision_threshold)
    eval_results = evaluator.evaluate_on_dataloader(data_loader, split_name)
    
    # Save plots
    evaluator.plot_evaluation_graphics(eval_results, save_dir=report_dir)
    
    # Prepare JSON report content
    report_content = {
        "evaluation_timestamp": pd.Timestamp.now().isoformat(),
        "split_name": split_name,
        "decision_threshold_used": decision_threshold,
        "primary_task_metrics (tire_change)": eval_results["primary_task_metrics"],
        "secondary_task_metrics (tire_type)": eval_results["secondary_task_metrics"],
        # Optionally include raw outputs if small enough or needed, or summary stats
        # "raw_outputs_summary": {k: v.shape for k,v in eval_results["raw_outputs"].items()}
    }
    
    report_file_path = report_dir / f"evaluation_report_{split_name.lower()}.json"
    with open(report_file_path, 'w') as f:
        json.dump(report_content, f, indent=4, default=lambda o: str(o) if isinstance(o, np.generic) else o.__dict__ if hasattr(o, '__dict__') else str(o)) # Handle numpy types
    logger.info(f"Evaluation report saved to {report_file_path}")
    
    return report_content

if __name__ == '__main__':
    # This block is for basic testing of evaluation utilities.
    # A proper test would involve a dummy model, dummy data, and dummy dataloaders.
    logging.basicConfig(level=logging.INFO)
    logger.info("--- Testing Evaluation Utilities ---")

    # Example: Create a dummy model for testing structure
    class DummyModel(nn.Module):
        def __init__(self, input_size, num_compounds):
            super().__init__()
            self.fc1 = nn.Linear(input_size, 1) # For tire_change_logits
            self.fc2 = nn.Linear(input_size, num_compounds) # For tire_type_logits
        def forward(self, x, return_sequences=False): # x is (batch, seq, features)
            # Simulate taking last element of sequence
            x_last_step = x[:, -1, :] 
            return {
                'tire_change_logits': self.fc1(x_last_step),
                'tire_type_logits': self.fc2(x_last_step)
            }

    dummy_input_size = 36 # From previous logs
    dummy_num_compounds = 7 # From previous logs
    test_model = DummyModel(dummy_input_size, dummy_num_compounds)
    
    # Create dummy data and DataLoader
    # (Using F1SequenceDataset and create_dataloaders from data_loaders.py for more realistic test)
    try:
        # Adjust sys.path for direct execution of this script to find sibling modules
        current_script_path_test = Path(__file__).resolve()
        # src_dir_test = current_script_path_test.parent.parent # F1/local/src/
        project_root_f1_local_test = current_script_path_test.parent.parent.parent # F1/local/
        
        # Add F1/local/src to sys.path if not already there
        # This allows "from modeling.data_loaders import ..."
        src_path_for_test = project_root_f1_local_test / "src"
        if str(src_path_for_test) not in sys.path:
            sys.path.insert(0, str(src_path_for_test))
        
        # Also add F1/local to sys.path for imports like from modeling.model_def
        if str(project_root_f1_local_test) not in sys.path:
            sys.path.insert(0, str(project_root_f1_local_test))

        from modeling.data_loaders import create_dataloaders
        
        # Create dummy .npy files for the test_loader to load
        dummy_data_dir = project_root_f1_local_test / "drive/model_input_dummy_eval_test"
        dummy_data_dir.mkdir(parents=True, exist_ok=True)
        
        seq_len = 10
        n_samples_test = 100
        X_test_dummy = np.random.rand(n_samples_test, seq_len, dummy_input_size).astype(np.float32)
        y_change_test_dummy = np.random.randint(0, 2, n_samples_test).astype(np.int64)
        y_type_test_dummy = np.random.randint(0, dummy_num_compounds, n_samples_test).astype(np.int64)
        y_type_test_dummy[y_change_test_dummy == 0] = -1 # Simulate ignore_index for type target

        np.save(dummy_data_dir / "X_test.npy", X_test_dummy)
        np.save(dummy_data_dir / "y_change_test.npy", y_change_test_dummy)
        np.save(dummy_data_dir / "y_type_test.npy", y_type_test_dummy)

        dummy_dataloaders = create_dataloaders(
            model_input_dir=str(dummy_data_dir),
            batch_size=16,
            augment_train_positive_samples=False, # No train set here
            use_weighted_sampler_for_train=False
        )
        test_loader_dummy = dummy_dataloaders['test']

        if len(test_loader_dummy.dataset) > 0:
            dummy_report_dir = project_root_f1_local_test / "drive/evaluation_reports_dummy"
            generate_and_save_evaluation_report(
                model=test_model,
                data_loader=test_loader_dummy,
                device='cpu',
                decision_threshold=0.6,
                report_dir=dummy_report_dir,
                split_name="DummyTest"
            )
            logger.info("✅ Evaluation utility test with dummy data completed.")
        else:
            logger.warning("Dummy test dataset is empty. Cannot run full evaluation test.")

    except ImportError:
        logger.error("Could not import data_loaders for full test. Run this script from project root or ensure PYTHONPATH.")
    except Exception as e:
        logger.error(f"Error in evaluation test: {e}", exc_info=True)
    finally:
        # Clean up dummy data
        if 'dummy_data_dir' in locals() and dummy_data_dir.exists():
            for f_item in dummy_data_dir.iterdir(): f_item.unlink()
            dummy_data_dir.rmdir()
        if 'dummy_report_dir' in locals() and dummy_report_dir.exists():
            for f_item in dummy_report_dir.iterdir(): f_item.unlink()
            dummy_report_dir.rmdir()
        logger.info("Cleaned up dummy evaluation artifacts.")
