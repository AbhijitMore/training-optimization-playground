import matplotlib.pyplot as plt
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_data(batch_size=64):
    
    path = os.path.join('..')
    """Load MNIST dataset and apply transformations."""

    transform = transforms.ToTensor()

    train_set = datasets.MNIST(path, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(path, train=False, download=True, transform=transform)

    # calculate mean and std
    train_data = train_set.data.float() / 255.0
    mean = train_data.mean()
    std = train_data.std()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])

    train_set = datasets.MNIST(path, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(path, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader =  DataLoader(train_set, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

def plot_metrics(losses, accuracies, nb_epoch):
    """ Plot training loss and accuracy"""

    plt.figure(figsize=(12,5))

    # Loss plot
    plt.subplot(1,2,1)
    for name, loss_list in losses.items():
        plt.plot(range(1, nb_epoch+1), loss_list, marker='o', label=name)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(range(1, nb_epoch +1))
    plt.grid()
    plt.legend()

    # Accuracy plot
    plt.subplot(1,2,2)
    for name, acc_list in losses.items():
        plt.plot(range(1, nb_epoch+1), acc_list, marker='o', label=name)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.xticks(range(1, nb_epoch +1))
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig('resources/metrics.png')
    plt.close()








