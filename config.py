# Configuration for Federated Learning of Language Model

config = {
    # Federated Learning Rounds
    'num_rounds': 10,

    # Batch Size for Each Client
    'client_batch_size': 32,

    # Learning Rate for Federated Updates
    'learning_rate': 0.01,

    # Optimization Algorithm
    'optimizer': 'adam',  # Options: 'sgd', 'adam', etc.

    # Model Architecture and Size
    'model': {
        'name': 'bert',  # Choose the base model (e.g., 'bert', 'gpt-2', 'roberta')
        'hidden_size': 768,  # Adjust based on your model's architecture
        'num_layers': 12,  # Number of layers in the model
    },

    # Dataset Paths
    'data': {
        'train_data_path': 'path/to/train_data.csv',
        'val_data_path': 'path/to/validation_data.csv',
    },

    # Privacy and Security Settings (for reference, not fully implemented here)
    'privacy': {
        'differential_privacy': False,  # Set to True if using differential privacy
        'epsilon': 1.0,  # Privacy parameter for differential privacy
    }
}

# Other settings and customizations can be added as needed.

# Usage:
# You can load this configuration in your federated learning script by importing it:
# from config import config
