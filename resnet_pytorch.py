import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time
import copy
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class ResNetPyTorchDFUClassification:
    def __init__(self):
        # Define paths to datasets
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Define the dataset structure created by organize_dataset.py
        self.dataset_dir = os.path.join(self.base_dir, 'dataset')
        self.train_dir = os.path.join(self.dataset_dir, 'train')
        self.test_dataset_dir = os.path.join(self.dataset_dir, 'test')
        
        # Model parameters
        self.img_size = 224  # Input size for ResNet18
        self.batch_size = 32
        self.epochs = 50
        self.learning_rate = 0.0001
        
        # Check if GPU is available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create directories for outputs
        self.models_dir = os.path.join(self.base_dir, 'models')
        self.plots_dir = os.path.join(self.base_dir, 'plots')
        for directory in [self.models_dir, self.plots_dir, self.dataset_dir, self.train_dir, self.test_dataset_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    def prepare_dataset_structure(self):
        """Check if the dataset structure is properly set up"""
        print("Checking dataset structure...")
        
        # Create class directories for training set if not exists
        train_abnormal_dir = os.path.join(self.train_dir, 'Abnormal(Ulcer)')
        train_normal_dir = os.path.join(self.train_dir, 'Normal (Healthy skin)')
        for dir_path in [train_abnormal_dir, train_normal_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        
        # Create class directories for test set if not exists
        test_abnormal_dir = os.path.join(self.test_dataset_dir, 'Abnormal(Ulcer)')
        test_normal_dir = os.path.join(self.test_dataset_dir, 'Normal (Healthy skin)')
        for dir_path in [test_abnormal_dir, test_normal_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        
        # Check if dataset is organized
        has_train_data = (len(os.listdir(train_abnormal_dir)) > 0 or len(os.listdir(train_normal_dir)) > 0)
        has_test_data = (len(os.listdir(test_abnormal_dir)) > 0 or len(os.listdir(test_normal_dir)) > 0)
        
        if not has_train_data or not has_test_data:
            print("Warning: Dataset not fully organized!")
            print("Please run 'python organize_dataset.py' first to prepare your dataset.")
            print("Expected structure:")
            print("dataset/")
            print("├── train/")
            print("│   ├── Abnormal(Ulcer)/")
            print("│   └── Normal (Healthy skin)/")
            print("├── test/")
            print("│   ├── Abnormal(Ulcer)/")
            print("│   └── Normal (Healthy skin)/")
        else:
            print("Dataset structure verified. Found data in both train and test directories.")
    
    def create_data_transforms(self):
        """Create data transforms with augmentation for training and validation"""
        print("Creating data transforms...")
        
        # Training transforms with data augmentation
        self.train_transforms = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Test transforms (no augmentation, just resizing and normalization)
        self.test_transforms = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def create_data_loaders(self):
        """Create data loaders for training and test datasets"""
        print("Creating data loaders...")
        
        # Create datasets using ImageFolder
        try:
            train_dataset = datasets.ImageFolder(
                root=self.train_dir,
                transform=self.train_transforms
            )
            
            test_dataset = datasets.ImageFolder(
                root=self.test_dataset_dir,
                transform=self.test_transforms
            )
            
            # Split train dataset into train and validation (80%/20%)
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
            
            # Update validation dataset transform to use test_transforms
            val_dataset.dataset.transform = self.test_transforms
            
            # Create data loaders
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4
            )
            
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4
            )
            
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4
            )
            
            # Get class names
            self.class_names = train_dataset.dataset.classes
            self.class_to_idx = train_dataset.dataset.class_to_idx
            print(f"Class names: {self.class_names}")
            print(f"Class to index mapping: {self.class_to_idx}")
            print(f"Training samples: {len(train_dataset)}")
            print(f"Validation samples: {len(val_dataset)}")
            print(f"Test samples: {len(test_dataset)}")
            
            return True
        except Exception as e:
            print(f"Error creating data loaders: {e}")
            print("Please ensure your dataset is properly organized.")
            print("Run 'python organize_dataset.py' to prepare your dataset according to the required structure.")
            return False
    
    def create_resnet_model(self):
        """Create and prepare the ResNet18 model"""
        print("Creating ResNet18 model...")
        
        # Load pre-trained ResNet18
        model = models.resnet18(pretrained=True)
        
        # Freeze all layers except the final fully connected layer
        for param in model.parameters():
            param.requires_grad = False
        
        # Replace the final fully connected layer with 2 outputs
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: Abnormal and Normal
        
        # Move model to device
        model = model.to(self.device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=self.learning_rate)
        
        return model, criterion, optimizer
    
    def unfreeze_layers(self, model):
        """Unfreeze some layers for fine-tuning"""
        print("Unfreezing layers for fine-tuning...")
        
        # Unfreeze the last few layers (e.g., layer4 and fc)
        for param in model.layer4.parameters():
            param.requires_grad = True
        
        # The fc layer is already trainable
        
        # Create new optimizer with lower learning rate for fine-tuning
        optimizer = optim.Adam(
            [
                {'params': model.layer4.parameters(), 'lr': self.learning_rate / 10},
                {'params': model.fc.parameters(), 'lr': self.learning_rate / 5}
            ]
        )
        
        return model, optimizer
    
    def train_model(self, model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=25):
        """Train the model with early stopping"""
        since = time.time()
        
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        
        # Variables to track loss and accuracy for plotting
        train_loss_history = []
        val_loss_history = []
        train_acc_history = []
        val_acc_history = []
        
        # Early stopping parameters
        patience = 10
        counter = 0
        
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)
            
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode
                
                running_loss = 0.0
                running_corrects = 0
                
                # Iterate over data
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # Forward
                    # Track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        # Backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                # Save history
                if phase == 'train':
                    train_loss_history.append(epoch_loss)
                    train_acc_history.append(epoch_acc.item())
                else:
                    val_loss_history.append(epoch_loss)
                    val_acc_history.append(epoch_acc.item())
                
                # Deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    counter = 0  # Reset counter when improvement is seen
                elif phase == 'val':
                    counter += 1  # Increment counter when no improvement
            
            # Early stopping
            if counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
            
            print()
        
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f}')
        
        # Load best model weights
        model.load_state_dict(best_model_wts)
        
        # Return model and training history
        history = {
            'train_loss': train_loss_history,
            'val_loss': val_loss_history,
            'train_acc': train_acc_history,
            'val_acc': val_acc_history
        }
        
        return model, history
    
    def evaluate_model(self, model, dataloader, dataset_size):
        """Evaluate the model on test data"""
        print("Evaluating model on test data...")
        
        model.eval()  # Set model to evaluate mode
        
        # Initialize lists to store predictions and true labels
        all_preds = []
        all_labels = []
        
        # Iterate over test data
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                # Store predictions and labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate accuracy
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        print(f'Test Accuracy: {accuracy:.4f}')
        
        # Generate classification report
        report = classification_report(all_labels, all_preds, target_names=self.class_names)
        print(f'\nClassification Report:\n{report}')
        
        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Save metrics
        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'y_true': all_labels,
            'y_pred': all_preds
        }
        
        return metrics
    
    def plot_results(self, history, metrics):
        """Plot training history and evaluation metrics"""
        print("Generating plots...")
        
        # Plot training & validation accuracy and loss
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy plot
        axes[0].plot(history['train_acc'], label='Training Accuracy')
        axes[0].plot(history['val_acc'], label='Validation Accuracy')
        axes[0].set_title('ResNet18 Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        
        # Loss plot
        axes[1].plot(history['train_loss'], label='Training Loss')
        axes[1].plot(history['val_loss'], label='Validation Loss')
        axes[1].set_title('ResNet18 Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'pytorch_resnet_training_history.png'))
        plt.close()
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', 
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names)
        plt.title('ResNet18 Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(os.path.join(self.plots_dir, 'pytorch_resnet_confusion_matrix.png'))
        plt.close()
        
        # Save accuracy to a text file for easy reference
        with open(os.path.join(self.plots_dir, 'pytorch_resnet_accuracy.txt'), 'w') as f:
            f.write(f"ResNet18 Test Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"\n{metrics['classification_report']}")
    
    def run(self):
        """Run the ResNet18 classification pipeline with PyTorch"""
        print("Starting ResNet18 Diabetic Foot Ulcer Classification with PyTorch...")
        
        # Step 1: Prepare dataset structure
        self.prepare_dataset_structure()
        
        # Step 2: Create data transforms
        self.create_data_transforms()
        
        # Step 3: Create data loaders
        data_ready = self.create_data_loaders()
        if not data_ready:
            print("Cannot proceed without properly organized data.")
            print("Please organize your data according to the required structure and run the script again.")
            return
        
        # Define dataloaders and dataset sizes for training
        dataloaders = {
            'train': self.train_loader,
            'val': self.val_loader
        }
        
        dataset_sizes = {
            'train': len(self.train_loader.dataset),
            'val': len(self.val_loader.dataset)
        }
        
        # Step 4: Create ResNet18 model
        print("\n===== Creating ResNet18 Model =====")
        model, criterion, optimizer = self.create_resnet_model()
        
        # Step 5: First training phase with frozen layers
        print("\n===== Initial Training (Frozen Layers) =====")
        model, initial_history = self.train_model(
            model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=self.epochs//2
        )
        
        # Step 6: Fine-tune the model by unfreezing some layers
        print("\n===== Fine-tuning Model =====")
        model, optimizer = self.unfreeze_layers(model)
        model, fine_tuned_history = self.train_model(
            model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=self.epochs//2
        )
        
        # Combine histories for plotting
        combined_history = {
            'train_acc': initial_history['train_acc'] + fine_tuned_history['train_acc'],
            'val_acc': initial_history['val_acc'] + fine_tuned_history['val_acc'],
            'train_loss': initial_history['train_loss'] + fine_tuned_history['train_loss'],
            'val_loss': initial_history['val_loss'] + fine_tuned_history['val_loss']
        }
        
        # Step 7: Evaluate the model on test data
        print("\n===== Evaluating Model on Test Data =====")
        metrics = self.evaluate_model(model, self.test_loader, len(self.test_loader.dataset))
        
        # Step 8: Generate plots
        self.plot_results(combined_history, metrics)
        
        # Step 9: Save the trained model
        model_path = os.path.join(self.models_dir, 'pytorch_resnet18_final.pth')
        torch.save(model.state_dict(), model_path)
        print(f"\nResNet18 model saved to: {model_path}")
        
        # Print completion message
        print(f"\nResNet18 Classification with PyTorch Completed!")
        print(f"Final Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Plots and results saved in: {self.plots_dir}")

if __name__ == "__main__":
    # Initialize and run the ResNet18 classification with PyTorch
    resnet_project = ResNetPyTorchDFUClassification()
    resnet_project.run()