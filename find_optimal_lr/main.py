import os
import torch
import torch.nn.functional as F
from torch.optim import SGD
import sys
sys.path.append(os.path.abspath('..'))
from optimizer_comparison.utils import load_data, plot_metrics
from optimizer_comparison.nets import SimpleNeuralNet
import matplotlib.pyplot as plt

# Constants
BATCH_SIZE = 64
EPOCHS = 4
INPUT_SIZE = 28 * 28
HIDDEN_SIZE = 100
NUM_CLASSES = 10

def train_model(nb_epoch, optimizer, train_loader, test_loader, lr):
    """
    Trains the model for a set number of epochs using the specified optimizer.

    Args:
        nb_epoch (int): Number of epochs to train.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        train_loader (torch.utils.data.DataLoader): The training data loader.
        test_loader (torch.utils.data.DataLoader): The testing/validation data loader.
        lr (float): Learning rate used for training.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleNeuralNet(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)
    optimizer = optimizer(model.parameters(), lr=lr)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    print(f"\n{'='*40}\nTraining Model with LR={lr}\n{'='*40}\n")
    
    for epoch in range(nb_epoch):
        running_loss, corrects = 0, 0
        model.train()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.nll_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            corrects += (preds == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = 100. * corrects / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        # Validation after every epoch
        val_loss, val_acc = validate_model(model, test_loader, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch + 1}/{nb_epoch}] | Train Loss: {epoch_loss:.4f} | Train Accuracy: {epoch_accuracy:.2f}% "
              f"| Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")

    return train_losses, train_accuracies, val_losses, val_accuracies

def validate_model(model, test_loader, device):
    """
    Evaluates the model on the validation/test set.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        test_loader (torch.utils.data.DataLoader): DataLoader for test/validation set.
        device (str): Device (CPU/CUDA) to use.

    Returns:
        float: Validation loss.
        float: Validation accuracy.
    """
    model.eval()
    val_loss, corrects = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = F.nll_loss(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            corrects += (preds == labels).sum().item()

    val_loss /= len(test_loader.dataset)
    val_accuracy = 100. * corrects / len(test_loader.dataset)
    
    return val_loss, val_accuracy

def plot_metrics(train_metrics, val_metrics, epochs, lr_1, lr_2):
    """
    Plots train and validation metrics for both learning rates as separate subplots.
    Args:
        train_metrics (dict): Dictionary containing training losses and accuracies.
        val_metrics (dict): Dictionary containing validation losses and accuracies.
        epochs (int): Number of epochs.
        lr_1 (float): Initial learning rate.
        lr_2 (float): Optimal learning rate.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot Losses
    ax1.set_title('Training and Validation Loss')
    ax1.plot(range(1, epochs+1), train_metrics['loss_lr1'], 'r-', label=f'Train Loss LR={lr_1}')
    ax1.plot(range(1, epochs+1), val_metrics['loss_lr1'], 'r--', label=f'Val Loss LR={lr_1}')
    ax1.plot(range(1, epochs+1), train_metrics['loss_lr2'], 'b-', label=f'Train Loss LR={lr_2}')
    ax1.plot(range(1, epochs+1), val_metrics['loss_lr2'], 'b--', label=f'Val Loss LR={lr_2}')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot Accuracies
    ax2.set_title('Training and Validation Accuracy')
    ax2.plot(range(1, epochs+1), train_metrics['acc_lr1'], 'g-', label=f'Train Acc LR={lr_1}')
    ax2.plot(range(1, epochs+1), val_metrics['acc_lr1'], 'g--', label=f'Val Acc LR={lr_1}')
    ax2.plot(range(1, epochs+1), train_metrics['acc_lr2'], 'm-', label=f'Train Acc LR={lr_2}')
    ax2.plot(range(1, epochs+1), val_metrics['acc_lr2'], 'm--', label=f'Val Acc LR={lr_2}')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()

    # Save the figure
    if not os.path.exists('resources'):
        os.makedirs('resources')
    fig.tight_layout()
    plt.savefig('resources/metrics_comparison.png')
    plt.show()

def main():
    # Load data
    train_loader, test_loader = load_data(batch_size=BATCH_SIZE)
    
    # Train with initial learning rate 1e-2
    train_loss_lr1, train_acc_lr1, val_loss_lr1, val_acc_lr1 = train_model(EPOCHS, SGD, train_loader, test_loader, lr=1e-2)
    
    # Train with optimal learning rate (assumed to be 1e-1)
    train_loss_lr2, train_acc_lr2, val_loss_lr2, val_acc_lr2 = train_model(EPOCHS, SGD, train_loader, test_loader, lr=1e-1)

    # Plot and compare the metrics
    plot_metrics(
        train_metrics={'loss_lr1': train_loss_lr1, 'acc_lr1': train_acc_lr1, 'loss_lr2': train_loss_lr2, 'acc_lr2': train_acc_lr2},
        val_metrics={'loss_lr1': val_loss_lr1, 'acc_lr1': val_acc_lr1, 'loss_lr2': val_loss_lr2, 'acc_lr2': val_acc_lr2},
        epochs=EPOCHS,
        lr_1=1e-2,
        lr_2=1e-1
    )

if __name__ == '__main__':
    main()
