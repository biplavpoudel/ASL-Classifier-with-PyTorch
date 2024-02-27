import torch
import torch.optim as optim
from CNN_Model import ASLClassifier
from data_preparation import device_check, create_dataset, split_dataset
from CustomLayers.Loss import CrossEntropyLoss
import os
# Set the environment variable for PyTorch CUDA memory allocation configuration
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, device, checkpoint_path):
    model.to(device)
    best_valid_loss = float('inf')
    for epoch in range(num_epochs):
        model.train(True)
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*images.size(0)

        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            for i, (images, labels) in enumerate(valid_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                valid_loss += criterion(outputs, labels).item()*images.size(0)

        # Print training statistics
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_val_loss = valid_loss / len(valid_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')

        if epoch_val_loss < best_valid_loss:
            best_valid_loss = epoch_val_loss
            save_checkpoint(model, optimizer, epoch, best_valid_loss, device, checkpoint_path)

        torch.cuda.empty_cache()


def save_checkpoint(model, optimizer, epoch, best_valid_loss, device, checkpoint_path):
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_valid_loss': best_valid_loss,
        'device': device
    }, checkpoint_path)


if __name__ == '__main__':
    device_name = device_check()
    train, test, classnames = create_dataset()
    dictionary = split_dataset(train, test, device_name)
    train_loader = dictionary['train']
    valid_loader = dictionary['valid']
    test_loader = dictionary['test']

    model = ASLClassifier()
    model.to(device_name)
    criteron = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    checkpoint_path = r'D:\ASL Classifier\model\checkpoints\checkpoint.pt'

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_valid_loss = checkpoint['best_valid_loss']
        print(f"Checkpoint loaded. Resuming training from epoch {epoch}.")
    else:
        epoch = 0
        best_valid_loss = float('inf')
        print("Checkpoint not found. Starting training from scratch.")

    num_epochs = 10
    train_model(model, train_loader, valid_loader, criteron, optimizer, num_epochs, device_name, checkpoint_path)

    # Saving final model
    final_model_path = r'D:\ASL Classifier\model\final_model.pt'
    torch.save(model.state_dict(), final_model_path)
