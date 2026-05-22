import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import Config

def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, f'{model_name.lower()}_confusion_matrix.png'))
    plt.close()

def plot_training_history(history, model_name):
    if hasattr(history, 'history'):
        # Keras history object
        h = history.history
        acc = h.get('accuracy', h.get('acc', []))
        val_acc = h.get('val_accuracy', h.get('val_acc', []))
        loss = h.get('loss', [])
        val_loss = h.get('val_loss', [])
    elif isinstance(history, dict):
        # PyTorch history dict
        acc = history.get('train_acc', history.get('accuracy', []))
        val_acc = history.get('val_acc', history.get('val_accuracy', []))
        loss = history.get('train_loss', history.get('loss', []))
        val_loss = history.get('val_loss', [])
    else:
        print("Warning: Unsupported history format for plotting.")
        return

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Training acc')
    if val_acc:
        plt.plot(epochs, val_acc, 'r-', label='Validation acc')
    plt.title(f'{model_name} Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Training loss')
    if val_loss:
        plt.plot(epochs, val_loss, 'r-', label='Validation loss')
    plt.title(f'{model_name} Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, f'{model_name.lower()}_training_history.png'))
    plt.close()

