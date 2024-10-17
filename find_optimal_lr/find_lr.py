import os
import sys
import math
import torch
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt

# Add the parent directory to the system path to import custom modules
sys.path.append(os.path.abspath('..'))
from optimizer_comparison.utils import load_data
from optimizer_comparison.nets import SimpleNeuralNet

# Constants (assuming they are defined elsewhere or need to be set)
INPUT_SIZE = 28 * 28  # Input size for a 28x28 pixel image
HIDDEN_SIZE = 100
NUM_CLASSES = 10

def find_optimal_lr(model, optimizer, criterion, train_loader, init_lr=1e-8, final_lr=10.0, beta=0.98):
    """
    Implements the learning rate finder, which helps determine the best learning rate by increasing 
    it exponentially and recording loss values.

    Args:
        model (torch.nn.Module): The neural network model.
        optimizer (torch.optim.Optimizer): The optimizer used for model training.
        criterion (torch.nn.Module): The loss function.
        train_loader (torch.utils.data.DataLoader): The DataLoader for training data.
        init_lr (float): The initial learning rate value.
        final_lr (float): The maximum learning rate value.
        beta (float): Smoothing factor for the loss (default: 0.98).
    
    Returns:
        log_lrs (list): List of logarithmic learning rate values.
        losses (list): List of corresponding loss values.
    """
    # Set up variables
    num_batches = len(train_loader) - 1
    lr_multiplier = (final_lr / init_lr) ** (1 / num_batches)
    current_lr = init_lr
    optimizer.param_groups[0]['lr'] = current_lr
    
    avg_loss, best_loss = 0.0, float('inf')
    batch_number = 0
    losses, log_lrs = [], []
    
    # Set device (GPU or CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    print(f"\n{'='*40}\nStarting Learning Rate Finder...\n{'='*40}\n")
    
    # Training loop over batches
    for inputs, labels in train_loader:
        batch_number += 1
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Compute smoothed loss
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** batch_number)
        
        # Stop if the loss explodes
        if batch_number > 1 and smoothed_loss > 4 * best_loss:
            print(f"Stopping early as loss exceeded 4 times the best loss at batch {batch_number}.")
            break
        
        # Track best loss
        if smoothed_loss < best_loss or batch_number == 1:
            best_loss = smoothed_loss
        
        # Log loss and learning rate
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(current_lr))
        
        # Backward pass and optimizer step
        loss.backward()
        optimizer.step()
        
        # Update learning rate
        current_lr *= lr_multiplier
        optimizer.param_groups[0]['lr'] = current_lr
        
        # Output progress
        if batch_number % 30 == 0 or batch_number == num_batches:
            print(f"Batch [{batch_number}/{num_batches}] | LR: {current_lr:.8f} | Loss: {smoothed_loss:.6f}")
    
    print(f"\n{'='*40}\nLearning Rate Finder Completed.\n{'='*40}\n")
    return log_lrs, losses


if __name__ == '__main__':
    # Load data (assuming the function returns train and test loaders)
    train_loader, _ = load_data(batch_size=64)

    # Initialize model, optimizer, and loss function
    model = SimpleNeuralNet(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)
    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    criterion = F.nll_loss
    
    # Run the learning rate finder
    log_lrs, losses = find_optimal_lr(model, optimizer, criterion, train_loader)
    
    # Plot the learning rate vs loss
    plt.plot(log_lrs[10:-5], losses[10:-5])
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.savefig('resources/learning_rate.png')
    plt.close()

    print("\nLearning rate plot saved at 'resources/learning_rate.png'\n")
