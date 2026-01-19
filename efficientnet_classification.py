import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from PIL import Image
import datetime

class EfficientNetDFUClassification:
    def __init__(self):
        # Set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Set paths
        self.data_dir = 'dataset'
        self.model_dir = 'models'
        self.plots_dir = 'plots'
        
        # Create directories if they don't exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Set model parameters
        self.num_epochs = 25
        self.batch_size = 16
        self.input_size = 224
        self.num_classes = 2
        self.learning_rate = 0.001
        
        # Initialize data transforms
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(self.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        # Initialize model, criterion, optimizer
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        
        # Initialize metrics
        self.best_model_wts = None
        self.best_acc = 0.0
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []
        
    def load_data(self):
        # Load datasets
        print("Loading datasets...")
        
        # Check if dataset directory exists
        if not os.path.exists(self.data_dir):
            print(f"Error: Dataset directory '{self.data_dir}' not found.")
            print("Please organize your dataset first using organize_dataset.py")
            return False
        
        # Check if train and test directories exist
        if not os.path.exists(os.path.join(self.data_dir, 'train')) or not os.path.exists(os.path.join(self.data_dir, 'test')):
            print("Error: 'train' or 'test' directories not found in dataset.")
            print("Please organize your dataset first using organize_dataset.py")
            return False
        
        # Create dataset loaders - use 'test' directory for validation
        self.data_transforms['test'] = self.data_transforms['val']  # Use validation transforms for test data
        
        self.image_datasets = {
            'train': datasets.ImageFolder(os.path.join(self.data_dir, 'train'), self.data_transforms['train']),
            'val': datasets.ImageFolder(os.path.join(self.data_dir, 'test'), self.data_transforms['test'])
        }
        
        self.dataloaders = {
            'train': torch.utils.data.DataLoader(self.image_datasets['train'], batch_size=self.batch_size, shuffle=True, num_workers=4),
            'val': torch.utils.data.DataLoader(self.image_datasets['val'], batch_size=self.batch_size, shuffle=True, num_workers=4)
        }
        
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'val']}
        self.class_names = self.image_datasets['train'].classes
        
        print(f"Classes: {self.class_names}")
        print(f"Training images: {self.dataset_sizes['train']}")
        print(f"Validation images: {self.dataset_sizes['val']}")
        
        return True
    
    def initialize_model(self):
        # Initialize EfficientNet-B0 model
        print("Initializing EfficientNet-B0 model...")
        
        # Load pre-trained EfficientNet-B0
        self.model = models.efficientnet_b0(pretrained=True)
        
        # Modify the classifier for our binary classification
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, self.num_classes)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Define loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Define optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        
        # Define learning rate scheduler
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        
        print("Model initialized successfully.")
    
    def train_model(self):
        # Train the model
        print("Starting model training...")
        
        since = time.time()
        
        # Initialize best model weights
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.best_acc = 0.0
        
        # Training loop
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch+1}/{self.num_epochs}')
            print('-' * 10)
            
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode
                
                running_loss = 0.0
                running_corrects = 0
                
                # Iterate over data
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Zero the parameter gradients
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)
                        
                        # Backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                    
                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                if phase == 'train':
                    self.scheduler.step()
                
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                # Record history
                if phase == 'train':
                    self.train_loss_history.append(epoch_loss)
                    self.train_acc_history.append(epoch_acc.item())
                else:
                    self.val_loss_history.append(epoch_loss)
                    self.val_acc_history.append(epoch_acc.item())
                
                # Deep copy the model if best accuracy
                if phase == 'val' and epoch_acc > self.best_acc:
                    self.best_acc = epoch_acc
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())
            
            print()
        
        # Training complete
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {self.best_acc:.4f}')
        
        # Load best model weights
        self.model.load_state_dict(self.best_model_wts)
        
        # Save the model
        self.save_model()
        
        # Plot training curves
        self.plot_training_curves()
        
        return self.model
    
    def save_model(self):
        # Save the trained model
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.model_dir, f'pytorch_efficientnet_b0_final.pth')
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    def plot_training_curves(self):
        # Plot training and validation loss
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(self.num_epochs), self.train_loss_history, label='Train')
        plt.plot(range(self.num_epochs), self.val_loss_history, label='Validation')
        plt.title('Loss vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(range(self.num_epochs), self.train_acc_history, label='Train')
        plt.plot(range(self.num_epochs), self.val_acc_history, label='Validation')
        plt.title('Accuracy vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.plots_dir, 'efficientnet_training_curves.png')
        plt.savefig(plot_path)
        print(f"Training curves saved to {plot_path}")
        plt.close()
    
    def run(self):
        # Run the entire pipeline
        if self.load_data():
            self.initialize_model()
            self.train_model()
            print("EfficientNet DFU classification completed successfully!")
        else:
            print("Failed to load data. Please check your dataset organization.")

if __name__ == "__main__":
    efficientnet_project = EfficientNetDFUClassification()
    efficientnet_project.run()