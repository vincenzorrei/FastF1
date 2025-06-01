"""
F1 Tyre Strategy RNN Model Definition
=====================================

This script defines the RNN/LSTM model architecture for predicting
tyre changes and next compound types. It adapts the architecture
from 'Vincenzo/dataset/models/lstm_architecture.py'.

Key Components:
- LSTMTirePredictor: Multi-task LSTM model.
- CombinedLoss: Custom loss function for multi-task learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import yaml
import joblib
from pathlib import Path
import logging # Import the logging module

logger = logging.getLogger(__name__) # Assuming logger is configured elsewhere or add basicConfig

class LSTMTirePredictor(nn.Module):
    """
    Multi-task LSTM for predicting F1 tire strategy.

    Args:
        input_size (int): Number of features per timestep.
        hidden_size (int): Dimension of LSTM hidden state.
        num_layers (int): Number of LSTM layers.
        num_compounds (int): Number of unique tire compounds for classification.
        dropout_lstm (float): Dropout rate for LSTM layers (if num_layers > 1).
        dropout_head (float): Dropout rate for task-specific heads.
        device (str): Device to run the model on ('cpu' or 'cuda').
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2, # Reduced default from 3 for potentially faster local training
        num_compounds: int = 7, # Default, will be updated from encoders
        dropout_lstm: float = 0.3,
        dropout_head: float = 0.2, # Consolidated dropout for heads
        device: str = 'cpu'
    ):
        super(LSTMTirePredictor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_compounds = num_compounds
        self.device = device
        
        # Shared LSTM trunk
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_lstm if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False # Unidirectional for real-time prediction context
        )
        
        # Task 1: Tire change prediction (Binary Classification)
        self.tire_change_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_head),
            nn.Linear(hidden_size // 2, 1)  # Sigmoid applied in loss
        )
        
        # Task 2: Tire compound type prediction (Multi-class Classification)
        self.tire_type_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), # Adjusted head complexity
            nn.ReLU(),
            nn.Dropout(dropout_head),
            nn.Linear(hidden_size // 2, num_compounds)  # Softmax applied in loss
        )
        
        self._init_weights()
        self.to(self.device)

    def _init_weights(self):
        """Initialize weights using Xavier and Orthogonal initialization."""
        for name, param in self.named_parameters():
            if 'lstm' in name: # LSTM weights
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
                    # Specific initialization for forget gate bias (optional but common)
                    # n = param.size(0)
                    # start, end = n // 4, n // 2
                    # param.data[start:end].fill_(1.) 
            elif 'head' in name: # Head weights
                 if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                 elif 'bias' in name:
                    param.data.fill_(0)
                
    def forward(
        self, 
        x: torch.Tensor,
        return_sequences: bool = False # Default to False for typical inference/last-step training
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        Args:
            x: Input tensor (batch_size, seq_length, input_size).
            return_sequences: If True, returns output for each timestep in the sequence.
                              If False, returns output only for the last timestep.
        Returns:
            A dictionary containing logits for both tasks.
        """
        x = x.to(self.device)
        batch_size, seq_length, _ = x.shape
        
        lstm_out, _ = self.lstm(x) # (hidden, cell) state not typically needed for direct output
        
        if return_sequences:
            # Use all timesteps (e.g., for sequence-to-sequence tasks or specific training regimes)
            features_for_heads = lstm_out.reshape(-1, self.hidden_size) # (batch*seq, hidden_size)
        else:
            # Use only the last timestep's output for prediction
            features_for_heads = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        tire_change_logits = self.tire_change_head(features_for_heads)
        tire_type_logits = self.tire_type_head(features_for_heads)
        
        if return_sequences:
            # Reshape back to (batch_size, seq_length, num_outputs)
            tire_change_logits = tire_change_logits.view(batch_size, seq_length, 1)
            tire_type_logits = tire_type_logits.view(batch_size, seq_length, self.num_compounds)
        
        return {
            'tire_change_logits': tire_change_logits,
            'tire_type_logits': tire_type_logits
        }

    def predict_probabilities(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generates predictions with probabilities for inference."""
        self.eval() # Set model to evaluation mode
        with torch.no_grad():
            outputs = self.forward(x, return_sequences=False) # Inference uses last timestep
            
            tire_change_probs = torch.sigmoid(outputs['tire_change_logits'])
            tire_type_probs = F.softmax(outputs['tire_type_logits'], dim=-1)
            
            return {
                'tire_change_probs': tire_change_probs,
                'tire_type_probs': tire_type_probs
            }

class CombinedLoss(nn.Module):
    """
    Combined loss for multi-task learning: tire change and tire type prediction.
    Loss = alpha * tire_change_loss + beta * tire_type_loss
    """
    def __init__(
        self,
        alpha: float = 0.9,  # Weight for tire_change_loss
        beta: float = 0.1,   # Weight for tire_type_loss
        pos_weight_tire_change: Optional[float] = None, # For BCEWithLogitsLoss pos_weight
        device: str = 'cpu'
    ):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.device = device
        
        pos_weight_tensor = torch.tensor(pos_weight_tire_change, device=self.device) if pos_weight_tire_change else None
        self.tire_change_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        
        # ignore_index for type loss: if target is -1, it's ignored (e.g. no tire change)
        self.tire_type_loss_fn = nn.CrossEntropyLoss(ignore_index=-1) 
        self.to(self.device)

    def forward(
        self,
        predictions: Dict[str, torch.Tensor], # Model outputs
        targets_change: torch.Tensor,         # (batch_size, [seq_length if sequence_out])
        targets_type: torch.Tensor            # (batch_size, [seq_length if sequence_out])
    ) -> Dict[str, torch.Tensor]:
        
        tire_change_logits = predictions['tire_change_logits']
        tire_type_logits = predictions['tire_type_logits']

        # Ensure targets are on the same device as logits
        targets_change = targets_change.to(self.device)
        targets_type = targets_type.to(self.device)

        # Reshape logits and targets if they are sequential outputs
        # (batch_size, seq_len, num_classes) -> (batch_size * seq_len, num_classes)
        # (batch_size, seq_len) -> (batch_size * seq_len)
        if tire_change_logits.ndim == 3: # (batch, seq_len, 1)
            tire_change_logits = tire_change_logits.reshape(-1, 1)
            targets_change = targets_change.reshape(-1)
        if tire_type_logits.ndim == 3: # (batch, seq_len, num_compounds)
            tire_type_logits = tire_type_logits.reshape(-1, tire_type_logits.size(-1))
            targets_type = targets_type.reshape(-1)

        # Tire Change Loss (BCE)
        # Squeeze logits if it's (N, 1) to match target (N)
        loss_change = self.tire_change_loss_fn(
            tire_change_logits.squeeze(-1) if tire_change_logits.shape[-1] == 1 else tire_change_logits,
            targets_change.float()
        )
        
        # Tire Type Loss (CrossEntropy)
        # Calculated only where a tire change is actually happening (target_change == 1)
        # Or, more simply, rely on ignore_index=-1 if target_type is set to -1 when no change.
        # The create_sequences_for_rnn sets target_type to encoded compound or -1 (error).
        # It should use a specific code for 'NO_CHANGE' that LabelEncoder handles,
        # and then CombinedLoss can use ignore_index for that specific code if desired,
        # or the mask approach.
        # Current Vincenzo's code uses a mask. Let's stick to that for fidelity.
        
        mask_positive_change = targets_change.bool() # Where a change actually occurs
        loss_type = torch.tensor(0.0, device=self.device)

        if mask_positive_change.sum() > 0:
            loss_type = self.tire_type_loss_fn(
                tire_type_logits[mask_positive_change],
                targets_type[mask_positive_change]
            )
        
        total_loss = self.alpha * loss_change + self.beta * loss_type
        
        return {
            'total_loss': total_loss,
            'tire_change_loss': loss_change,
            'tire_type_loss': loss_type
        }

def load_model_config(config_path: str) -> dict:
    """Loads model configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if 'model' not in config:
        raise ValueError("Model configuration must contain a 'model' key.")
    return config['model']

def get_model_hyperparameters(config_path: str, artifacts_dir: str, device: str) -> dict:
    """Loads hyperparameters from config and derives some from artifacts."""
    model_params_config = load_model_config(config_path)
    
    # Load feature_columns to get input_size
    feature_cols_path = Path(artifacts_dir) / "feature_columns.joblib"
    if not feature_cols_path.exists():
        raise FileNotFoundError(f"feature_columns.joblib not found at {feature_cols_path}")
    feature_columns = joblib.load(feature_cols_path)
    input_size = len(feature_columns)

    # Load encoders to get num_compounds
    encoders_path = Path(artifacts_dir) / "encoders.pkl"
    if not encoders_path.exists():
        raise FileNotFoundError(f"encoders.pkl not found at {encoders_path}")
    encoders = joblib.load(encoders_path)
    if 'Compound' not in encoders or not hasattr(encoders['Compound'], 'classes_'):
        raise ValueError("Compound encoder not found or is invalid in encoders.pkl")
    num_compounds = len(encoders['Compound'].classes_)

    # Get other params from config
    rnn_config = model_params_config.get('rnn', {})
    hidden_size = rnn_config.get('hidden_size', 128 if device == 'cuda' else 64)
    if device == 'cpu' and hidden_size > 64: # Cap for CPU
        logger.info(f"Reducing hidden_size from {hidden_size} to 64 for CPU.")
        hidden_size = 64
        
    num_layers = rnn_config.get('num_layers', 2)
    dropout_lstm = rnn_config.get('dropout_lstm', 0.3)
    
    head_config = model_params_config.get('heads', {})
    dropout_head = head_config.get('dropout_head', 0.2)

    return {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_compounds': num_compounds,
        'dropout_lstm': dropout_lstm,
        'dropout_head': dropout_head,
        'device': device
    }

if __name__ == '__main__':
    # This block is for basic testing of the model structure.
    # A proper test would involve loading data and checking outputs.
    logger.setLevel(logging.INFO) # Ensure logger is active for __main__
    
    # Create dummy artifacts for testing
    dummy_artifacts_path = Path("F1/local/drive/artifacts_dummy/")
    dummy_artifacts_path.mkdir(parents=True, exist_ok=True)
    
    # Dummy feature_columns.joblib
    dummy_features = [f'feature_{i}' for i in range(36)] # Matches previous run
    joblib.dump(dummy_features, dummy_artifacts_path / "feature_columns.joblib")
    
    # Dummy encoders.pkl
    from sklearn.preprocessing import LabelEncoder
    dummy_compound_encoder = LabelEncoder()
    dummy_compound_encoder.fit(['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET', 'ULTRASOFT', 'NO_CHANGE'])
    dummy_encoders = {'Compound': dummy_compound_encoder}
    joblib.dump(dummy_encoders, dummy_artifacts_path / "encoders.pkl")

    # Dummy model_config.yaml
    dummy_config_path = Path("F1/local/configs/dummy_model_config.yaml")
    dummy_config_content = {
        'model': {
            'rnn': {'hidden_size': 64, 'num_layers': 2, 'dropout_lstm': 0.25},
            'heads': {'dropout_head': 0.15}
            # num_compounds and input_size will be derived
        }
    }
    with open(dummy_config_path, 'w') as f:
        yaml.dump(dummy_config_content, f)

    logger.info("--- Testing Model Definition ---")
    device_to_test = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device_to_test}")

    try:
        model_params = get_model_hyperparameters(
            config_path=str(dummy_config_path),
            artifacts_dir=str(dummy_artifacts_path),
            device=device_to_test
        )
        logger.info(f"Loaded model parameters: {model_params}")

        model = LSTMTirePredictor(**model_params)
        logger.info("Model initialized successfully.")
        logger.info(f"Model is on device: {next(model.parameters()).device}")

        # Test forward pass
        batch_size = 4
        seq_length = 10 # From prepare_model_data.py
        test_input = torch.randn(batch_size, seq_length, model_params['input_size']).to(device_to_test)
        
        predictions = model(test_input)
        assert predictions['tire_change_logits'].shape == (batch_size, 1)
        assert predictions['tire_type_logits'].shape == (batch_size, model_params['num_compounds'])
        logger.info("Forward pass (last step output) successful.")

        predictions_seq = model(test_input, return_sequences=True)
        assert predictions_seq['tire_change_logits'].shape == (batch_size, seq_length, 1)
        assert predictions_seq['tire_type_logits'].shape == (batch_size, seq_length, model_params['num_compounds'])
        logger.info("Forward pass (sequence output) successful.")

        # Test loss function
        loss_params_config = dummy_config_content['model'].get('loss', {}) # Assuming loss params could be in config
        combined_loss_fn = CombinedLoss(
            alpha=loss_params_config.get('alpha', 0.9),
            beta=loss_params_config.get('beta', 0.1),
            pos_weight_tire_change=loss_params_config.get('pos_weight_tire_change', 25.0), # Example
            device=device_to_test
        )
        
        # Dummy targets (for last step output)
        dummy_targets_change = torch.randint(0, 2, (batch_size,)).to(device_to_test)
        dummy_targets_type = torch.randint(0, model_params['num_compounds'], (batch_size,)).to(device_to_test)
        # Set some type targets to -1 where change is 0, if loss expects that
        dummy_targets_type[dummy_targets_change == 0] = -1 


        loss_output = combined_loss_fn(predictions, dummy_targets_change, dummy_targets_type)
        logger.info(f"Loss calculation successful: {loss_output}")
        
        logger.info("✅ Model definition and basic tests completed successfully!")

    except Exception as e:
        logger.error(f"Error during model definition testing: {e}", exc_info=True)
    finally:
        # Clean up dummy files
        if (dummy_artifacts_path / "feature_columns.joblib").exists():
            (dummy_artifacts_path / "feature_columns.joblib").unlink()
        if (dummy_artifacts_path / "encoders.pkl").exists():
            (dummy_artifacts_path / "encoders.pkl").unlink()
        if dummy_artifacts_path.exists():
            dummy_artifacts_path.rmdir() # Fails if not empty, but fine for this test
        if dummy_config_path.exists():
            dummy_config_path.unlink()
        logger.info("Cleaned up dummy artifacts and config.")
