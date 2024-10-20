import os
import torch
import torch.nn.functional as F
from torch.optim import SGD
import sys
import time
sys.path.append(os.path.abspath('..'))
from optimizer_comparison.utils import load_data, plot_metrics
from optimizer_comparison.nets import SimpleNeuralNet
import matplotlib.pyplot as plt

# Constants
BATCH_SIZE = 64
EPOCHS = 15
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
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    print(device)
    
    model = SimpleNeuralNet(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)
    optimizer = optimizer(model.parameters(), lr=lr)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
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
        device (str): Device (CPU/CUDA/mps) to use.

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

def main():
    # Load data
    
    train_loader, test_loader = load_data(batch_size=BATCH_SIZE, num_workers=0, pin_memory=False)
    start_time = time.time()
    train_loss_lr1, train_acc_lr1, val_loss_lr1, val_acc_lr1 = train_model(EPOCHS, SGD, train_loader, test_loader, lr=1e-3)
    end_time = time.time()
    print(end_time-start_time)
    
    train_loader, test_loader = load_data(batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
    start_time = time.time()
    train_loss_lr1, train_acc_lr1, val_loss_lr1, val_acc_lr1 = train_model(EPOCHS, SGD, train_loader, test_loader, lr=1e-3)
    end_time = time.time()
    print(end_time-start_time)
    
    train_loader, test_loader = load_data(batch_size=BATCH_SIZE, num_workers=1, pin_memory=False)
    start_time = time.time()
    train_loss_lr1, train_acc_lr1, val_loss_lr1, val_acc_lr1 = train_model(EPOCHS, SGD, train_loader, test_loader, lr=1e-3)
    end_time = time.time()
    print(end_time-start_time)
    
    train_loader, test_loader = load_data(batch_size=BATCH_SIZE, num_workers=1, pin_memory=True)
    start_time = time.time()
    train_loss_lr1, train_acc_lr1, val_loss_lr1, val_acc_lr1 = train_model(EPOCHS, SGD, train_loader, test_loader, lr=1e-3)
    end_time = time.time()
    print(end_time-start_time)


if __name__ == '__main__':
    main()
