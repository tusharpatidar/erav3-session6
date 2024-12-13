import torch
import torch.nn.functional as F
from tqdm import tqdm
from network import Network
from torchvision import datasets, transforms
from torch.utils.data import random_split
import os

RANDOM_SEED = 42
MOMENTUM = 0.9
BATCH_SIZE = 128
LR = 0.05
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 19

def prepare_data(batch_size):
    # Define transformations
    train_transform = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load full training dataset
    full_train_dataset = datasets.MNIST('../data', train=True, download=True,
                                      transform=train_transform)
    
    # Split training dataset into train and validation
    train_size = 50000  # 50k for training
    val_size = 10000    # 10k for validation/test
    
    train_dataset, _ = random_split(full_train_dataset, 
                                  [train_size, val_size],
                                  generator=torch.Generator().manual_seed(42))
    
    # Use the test set as our validation/test set
    val_dataset = datasets.MNIST('../data', train=False,
                               transform=test_transform)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size,
                                             shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)
    
    return train_loader, val_loader 

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'Epoch {epoch}: Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return test_loss, accuracy

def main():
    torch.manual_seed(RANDOM_SEED)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    train_loader, test_loader = prepare_data(BATCH_SIZE)
    
    # Calculate dataset sizes
    train_size = len(train_loader.dataset)
    test_size = len(test_loader.dataset)
    
    model = Network().to(DEVICE)
    total_params = count_parameters(model)
    print(f"\nTotal Model Parameters: {total_params:,}")
    print("\nDataset Split:")
    print(f"Training samples: {train_size:,}")
    print(f"Validation/Test samples: {test_size:,}")
    print(f"Split ratio: {train_size}/{test_size}")
    
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    
    best_accuracy = 0.0
    train_losses = []
    test_losses = []
    accuracies = []
    target_accuracy = 99.4
    
    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, DEVICE, train_loader, optimizer, epoch)
        test_loss, accuracy = test(model, DEVICE, test_loader, epoch)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        accuracies.append(accuracy)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'models/best_model.pth')
            
        scheduler.step(accuracy)
        
        # Early stopping if accuracy reaches target
        if accuracy >= target_accuracy:
            print(f"\nReached target accuracy of {target_accuracy}% at epoch {epoch}")
            break
    
    print("\nTraining Complete!")
    print("=" * 50)
    print(f"Dataset Split Summary:")
    print(f"Training Set: {train_size:,} samples")
    print(f"Validation/Test Set: {test_size:,} samples")
    print(f"Split Ratio: {train_size}/{test_size}")
    print("-" * 50)
    print(f"Total Model Parameters: {total_params:,}")
    print(f"Best Validation/Test Accuracy: {best_accuracy:.2f}%")
    print(f"Final Training Loss: {train_losses[-1]:.4f}")
    print(f"Final Validation/Test Loss: {test_losses[-1]:.4f}")
    print(f"Training stopped at epoch: {len(accuracies)}/{EPOCHS}")
    print("=" * 50)

if __name__ == '__main__':
    main() 