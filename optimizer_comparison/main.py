import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam, RMSprop
from utils import load_data, plot_metrics
from nets import SimpleNeuralNet

# Constants
BATCH_SIZE = 64
EPOCHS = 10
INPUT_SIZE = 28 * 28  # Input size for an image of 28x28 pixels
HIDDEN_LAYER_SIZE = 100
NUM_CLASSES = 10
LEARNING_RATE = 0.01


def train_model(num_epochs, optimizer_classes, train_loader):
    """
    Trains a Simple Neural Network model using different optimizers and logs performance metrics.

    Args:
        num_epochs (int): Number of epochs to train the model.
        optimizer_classes (list): List of optimizer classes to be used (SGD, Adam, RMSprop, etc.).
        train_loader (DataLoader): DataLoader for the training set.

    Returns:
        dict: Losses per optimizer.
        dict: Accuracies per optimizer.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Dictionaries to track losses and accuracies for each optimizer
    losses_per_optimizer = {opt.__name__: [] for opt in optimizer_classes}
    accuracies_per_optimizer = {opt.__name__: [] for opt in optimizer_classes}

    print(f"training started...")
    
    # Train with each optimizer
    for optimizer_class in optimizer_classes:
        print(f"\n------- Training with {optimizer_class.__name__} -------")
        
        # Initialize model and optimizer
        model = SimpleNeuralNet(INPUT_SIZE, HIDDEN_LAYER_SIZE, NUM_CLASSES).to(device)
        optimizer = optimizer_class(model.parameters(), lr=LEARNING_RATE)
        
        for epoch in range(num_epochs):
            running_loss, correct_predictions = 0.0, 0
            
            # Iterate over the training batches
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass and loss computation
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = F.nll_loss(outputs, labels)
                
                # Backward pass and optimization step
                loss.backward()
                optimizer.step()

                # Update loss and accuracy
                running_loss += loss.item() * inputs.size(0)
                predicted_labels = torch.argmax(outputs, dim=1)
                correct_predictions += (predicted_labels == labels).sum().item()

            # Compute average loss and accuracy for the epoch
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_accuracy = 100.0 * correct_predictions / len(train_loader.dataset)
            
            # Store metrics
            losses_per_optimizer[optimizer_class.__name__].append(epoch_loss)
            accuracies_per_optimizer[optimizer_class.__name__].append(epoch_accuracy)
            
            # Enhanced console output for progress
            print(f"Epoch [{epoch + 1}/{num_epochs}] | Loss: {epoch_loss:.4f} | "
                  f"Accuracy: {epoch_accuracy:.2f}%")

    print(f"TRAINING FINISHED...")
    
    return losses_per_optimizer, accuracies_per_optimizer


def main():
    """
    Main function to load data, train the model with different optimizers, and plot performance metrics.
    """
    # Load data for training and testing
    train_loader, test_loader = load_data(batch_size=BATCH_SIZE)
    
    # List of optimizers to experiment with
    optimizer_classes = [SGD, Adam, RMSprop]

    print(f"loading data...")
    print(f"Training batches loaded with batch size: {BATCH_SIZE}")

    # Train model and capture losses/accuracies
    losses, accuracies = train_model(EPOCHS, optimizer_classes, train_loader)
    
    # Plot the metrics for comparison
    plot_metrics(losses, accuracies, EPOCHS)


if __name__ == '__main__':
    main()
