import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam, RMSprop
from utils import load_data, plot_metrics
from nets import SimpleNeuralNet

# Cconstants
BATCH_SIZE = 64
EPOCHS = 10
INPUT_SIZE = 28*28
HIDDEN_SIZE = 100
NUM_CLASSES = 10

def train_model(nb_epoch, optimizers, train_loader):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    losses = {opt.__name__:[] for opt in optimizers}
    accuracies = {opt.__name__:[] for opt in optimizers}

    for optimizer_class in optimizers:
        model = SimpleNeuralNet(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)
        optimizer = optimizer_class(model.parameters(), lr=0.01)
        
        for epoch in range(nb_epoch):
            running_loss, corrects = 0, 0
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
            losses[optimizer_class.__name__].append(epoch_loss)
            accuracies[optimizer_class.__name__].append(epoch_accuracy)
            
            print(f'{optimizer_class.__name__} - Epoch [{epoch + 1}/ {nb_epoch}]: Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
    
    return losses, accuracies

def main():
    train_loader, test_loader = load_data()
    optimizers = [SGD, Adam, RMSprop]
    losses, accuracies = train_model(EPOCHS, optimizers, train_loader)
    plot_metrics(losses, accuracies, EPOCHS)
    
    
if __name__ == '__main__':
    main()
    
                 
        

