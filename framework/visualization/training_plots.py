import matplotlib.pyplot as plt
import os
from ..utils import get_results_path


class TrainingVisualizer:
    def __init__(self, dataset_name=None, results_dir=None, model_name="autoencoder", time_span=300):
        if results_dir is None:
            self.results_dir = get_results_path(dataset_name, model_name, time_span, "models")
        else:
            self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
    def plot_training_history(self, history, model_name="autoencoder"):
        """Plot training history - adapts to available metrics"""
        # Check what metrics are available
        available_metrics = list(history.history.keys())
        has_mae = 'mae' in available_metrics
        has_val_loss = 'val_loss' in available_metrics and history.history['val_loss'] is not None
        
        # For models without training history (like One-Class SVM), just print info
        if len(history.history.get('loss', [])) <= 1:
            print("Model trained successfully (no iterative training history available)")
            return None
        
        # Determine plot layout
        if has_mae:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
        
        # Plot loss
        ax1.plot(history.history['loss'], label='Training Loss', color='blue')
        if has_val_loss:
            ax1.plot(history.history['val_loss'], label='Validation Loss', color='red')
        ax1.set_title('Model Loss During Training')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot MAE if available
        if has_mae:
            ax2.plot(history.history['mae'], label='Training MAE', color='blue')
            if 'val_mae' in available_metrics and history.history['val_mae'] is not None:
                ax2.plot(history.history['val_mae'], label='Validation MAE', color='red')
            ax2.set_title('Model MAE During Training')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Mean Absolute Error')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = os.path.join(self.results_dir, f"training_history.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training history plot saved to: {filename}")
        return filename
        
    def print_training_summary(self, history):
        """Print training summary statistics - adapts to available metrics"""
        available_metrics = list(history.history.keys())
        
        # For models without iterative training (like One-Class SVM)
        if len(history.history.get('loss', [])) <= 1:
            print("Training completed successfully")
            return
        
        print(f"Training completed after {len(history.history['loss'])} epochs")
        print(f"Final training loss: {history.history['loss'][-1]:.6f}")
        
        if 'val_loss' in available_metrics and history.history['val_loss'] is not None:
            print(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")
        
        if 'mae' in available_metrics:
            print(f"Final training MAE: {history.history['mae'][-1]:.6f}")
        
        if 'val_mae' in available_metrics and history.history['val_mae'] is not None:
            print(f"Final validation MAE: {history.history['val_mae'][-1]:.6f}")
