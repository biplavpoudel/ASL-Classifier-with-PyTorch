import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from CNN_Model import ASLClassifier
from data_preparation import device_check, create_dataset, split_dataset
from torch.nn import CrossEntropyLoss, MSELoss
import time
import os
import matplotlib.pyplot as plt
# Set the environment variable for PyTorch CUDA memory allocation configuration
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, device, checkpoint_path):
    model.to(device)
    best_valid_loss = float('inf')
    valid_losses = []
    train_losses = []
    train_accuracies = []
    valid_accuracies = []
    total_time = 0
    total_start_time = time.time()

    for epoch in range(num_epochs):
        model.train(True)
        running_loss = 0.0
        train_corrects = 0.0
        total_predictions = 0.0
        start_time = time.time()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            train_corrects += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        train_accuracy = train_corrects / len(train_loader)
        train_accuracies.append(train_accuracy)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            correct_val_predictions = 0.0
            total_val_predictions = 0.0
            for i, (images, labels) in enumerate(valid_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                valid_loss += criterion(outputs, labels).item()*images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val_predictions += labels.size(0)
                correct_val_predictions += (predicted == labels).sum().item()

        # Printing training statistics
        epoch_val_loss = valid_loss / len(valid_loader.dataset)
        valid_losses.append(epoch_val_loss)
        epoch_accuracy = correct_val_predictions / total_val_predictions
        valid_accuracies.append(epoch_accuracy)
        end_time = time.time()
        print(f'Epoch {epoch+1}/{num_epochs}, Time taken:{(end_time-start_time)/60.0:.4f} minutes,'
              f' Train Loss: {epoch_val_loss:.4f},'
              f' Val Loss: {epoch_val_loss:.4f}, Validation Accuracy:{epoch_accuracy: .3f}')

        if epoch_val_loss < best_valid_loss:
            best_valid_loss = epoch_val_loss
            save_checkpoint(model, optimizer, epoch, best_valid_loss, device, checkpoint_path)

        torch.cuda.empty_cache()

    total_end_time = time.time()
    total_time_taken = total_end_time - total_start_time
    print(f'Total Time Taken to train is {total_time_taken/60:.2f} minutes')

    # For Plotting losses in each epoch
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss vs Validation Loss')
    plt.legend()

    # For Plotting Accuracies in each epoch
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), valid_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train Accuracy vs Validation Accuracy')
    plt.legend()

    plt.show()


def save_checkpoint(model, optimizer, epoch, best_valid_loss, device, checkpoint_path):
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_valid_loss': best_valid_loss,
        'device': device
    }, checkpoint_path)


def test_model(model, test_loader, criterion, device):
    model.eval()  # Setting the model to evaluation mode
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print test statistics
    print('Test Loss: {:.3f}'.format(test_loss / len(test_loader)))
    print('Test Accuracy: {:.2f}%'.format(100 * correct / total))


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
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    checkpoint_path = r'D:\ASL Classifier\model\checkpoints\checkpoint.pt'

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_valid_loss = checkpoint['best_valid_loss']
        device = checkpoint['device']
        model.to(device)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print(f"Checkpoint loaded. Resuming training from epoch {epoch}.")
    else:
        epoch = 0
        best_valid_loss = float('inf')
        print("Checkpoint not found. Starting training from scratch.")

    num_epochs = 50
    train_prompt = input("Do you want to train the model? yes/no: ")
    if train_prompt == "y" or train_prompt == "yes":
        remaining_epochs = num_epochs - 1 - epoch
        if remaining_epochs > 0:
            train_model(model, train_loader, valid_loader, criteron, optimizer, num_epochs, device_name, checkpoint_path)
            # Saving final model
            final_model_path = r'D:\ASL Classifier\model\final_model.pt'
            torch.save(model.state_dict(), final_model_path)
        else:
            print(f"Training already completed for the {num_epochs} number  of epochs.")
    else:
        test_prompt = input("Do you want to test the model? yes/no: ")
        if test_prompt == "y" or test_prompt == "yes":
            model_path = r'D:\ASL Classifier\model\final_model.pt'
            final_model = ASLClassifier().to("cuda")
            final_model.load_state_dict(torch.load(model_path))
            test_model(final_model, test_loader, criteron, device_name)
