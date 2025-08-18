import matplotlib.pyplot as plt
import os


class TrainingVisualizer:
    def __init__(self, results_dir="./results/autoencoder"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
    def plot_training_history(self, history, model_name="autoencoder"):
        """Plot training loss and MAE history"""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss', color='blue')
        plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
        plt.title('Model Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE', color='blue')
        plt.plot(history.history['val_mae'], label='Validation MAE', color='red')
        plt.title('Model MAE During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = os.path.join(self.results_dir, f"training_history.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training history plot saved to: {filename}")
        return filename
        
    def print_training_summary(self, history):
        """Print training summary statistics"""
        print(f"Training completed after {len(history.history['loss'])} epochs")
        print(f"Final training loss: {history.history['loss'][-1]:.6f}")
        print(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")
        print(f"Final training MAE: {history.history['mae'][-1]:.6f}")
        print(f"Final validation MAE: {history.history['val_mae'][-1]:.6f}")
