import matplotlib.pyplot as plt

class LossAccuracyPlotter:
    def __init__(self, train_loss, val_loss, train_accuracy, val_accuracy, epochs:int=10):
        self.epochs = range(1, epochs+1)
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.train_accuracy = train_accuracy
        self.val_accuracy = val_accuracy
    
    def visualize(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
        axes[0].plot(self.epochs, self.train_loss, label='Train Loss', color='tab:blue')
        axes[0].plot(self.epochs, self.val_loss, label='Validation Loss', color='tab:orange')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
    
        axes[1].plot(self.epochs, self.train_accuracy, label='Train Accuracy', color='tab:blue')
        axes[1].plot(self.epochs, self.val_accuracy, label='Validation Accuracy', color='tab:orange')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        #axes[1].axhline(y=50, color='red', linestyle='--', label='Random')
        #axes[1].set_ylim(-1, 101)
        axes[1].legend()
        
        fig.tight_layout()
        plt.show()
        