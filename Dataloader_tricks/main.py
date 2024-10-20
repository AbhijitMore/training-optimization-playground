import os
import time
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import SGD
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import contextlib

sys.path.append(os.path.abspath('..'))
from optimizer_comparison.nets import SimpleNeuralNet

# Constants
BATCH_SIZE = 2048
EPOCHS = 10
INPUT_SIZE = 32 * 32 * 3
HIDDEN_SIZE = 512
NUM_CLASSES = 10
SEED = 142

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.backends.mps.is_available():  # Check if MPS (Metal) is available
        torch.manual_seed(seed)
        
def load_data(batch_size=64, shuffle=True, num_workers=0, pin_memory=False):
    
    path = os.path.join('..')
    """Load CIFAR-10 dataset and apply transformations."""

    # Define the transformations including normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])  # Ensure mean and std are in list format for normalization
    ])

    # Re-load the dataset with the updated transformations
    with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull):
        train_set = datasets.CIFAR10(path, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(path, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, test_loader

def train_model(device, nb_epoch, optimizer, train_loader, test_loader, lr):
    """
    Trains the model for a set number of epochs using the specified optimizer.

    Args:
        nb_epoch (int): Number of epochs to train.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        train_loader (torch.utils.data.DataLoader): The training data loader.
        test_loader (torch.utils.data.DataLoader): The testing/validation data loader.
        lr (float): Learning rate used for training.
    """
    
    device = torch.device(device)
        
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
        
    configurations = [
    {"device":"cpu" ,"num_workers": 0, "pin_memory": False},
    {"device":"mps" ,"num_workers": 0, "pin_memory": False},
    {"device":"mps" ,"num_workers": 0, "pin_memory": True},
    # {"num_workers": 1, "pin_memory": False},
    # {"num_workers": 1, "pin_memory": True}
    ]
    
    set_seed(42)

    timing_results = []
    for config in configurations:
        device, num_workers, pin_memory = config["device"], config["num_workers"], config["pin_memory"]
        train_loader, test_loader = load_data(batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        print(f"Using device: {device}, num_workers: {num_workers} and pin_memory: {pin_memory}")
        
        start_time = time.time()
        train_loss, train_acc, val_loss, val_acc = train_model(device, EPOCHS, SGD, train_loader, test_loader, lr=1e-1)
        end_time = time.time()
        time_taken = end_time - start_time
        timing_results.append((f"{device}_{num_workers}_{pin_memory}", time_taken))
        print(f"Time taken: {time_taken:.2f} seconds.")
        
    # Extract configurations and corresponding times for plotting
    labels = [result[0] for result in timing_results]
    times = [result[1] for result in timing_results]
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.bar(labels, times)
    plt.xlabel("Configuration")
    plt.ylabel("Time (seconds)")
    plt.title("Training Time per Configuration")
    plt.xticks(rotation=45)
    plt.tight_layout()
    bars = plt.bar(labels, times)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), 
                 f'{bar.get_height():.2f}', 
                 ha='center', va='bottom', fontsize=10)
    plt.savefig('resources/timings_plot.png')

if __name__ == '__main__':
    main()
