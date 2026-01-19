import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

class DFUClassification:
    def __init__(self):
        # Define paths to datasets
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.patches_dir = os.path.join(self.base_dir, 'Patches')
        self.test_dir = os.path.join(self.base_dir, 'TestSet')
        
        # Model parameters
        self.img_size = (224, 224)  # Input size for ResNet18 and EfficientNet-B0
        self.batch_size = 32
        self.epochs = 50
        self.learning_rate = 0.0001
        
        # Create directories for outputs
        self.models_dir = os.path.join(self.base_dir, 'models')
        self.plots_dir = os.path.join(self.base_dir, 'plots')
        for directory in [self.models_dir, self.plots_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    def prepare_data(self):
        """Prepare data generators for training, validation, and testing"""
        print("Preparing data generators...")
        
        # Define data augmentation for training set
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2  # 20% of data for validation
        )
        
        # Validation data generator (only rescaling)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        # Test data generator
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create data generators
        self.train_generator = train_datagen.flow_from_directory(
            self.patches_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='training'
        )
        
        self.val_generator = val_datagen.flow_from_directory(
            self.patches_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='validation'
        )
        
        # Check if TestSet has subdirectories, otherwise create a temporary structure
        if not self._has_subdirectories(self.test_dir):
            self._create_test_subdirectories()
            test_data_exists = True
        else:
            test_data_exists = True
            
        if test_data_exists:
            self.test_generator = test_datagen.flow_from_directory(
                self.test_dir,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='binary',
                shuffle=False
            )
        
        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.val_generator.samples}")
        if test_data_exists:
            print(f"Test samples: {self.test_generator.samples}")
            
        # Get class indices
        self.class_indices = self.train_generator.class_indices
        self.class_labels = {v: k for k, v in self.class_indices.items()}
        print(f"Class mapping: {self.class_labels}")
    
    def _has_subdirectories(self, directory):
        """Check if directory contains subdirectories"""
        try:
            return any(
                os.path.isdir(os.path.join(directory, item)) 
                for item in os.listdir(directory)
            )
        except:
            return False
    
    def _create_test_subdirectories(self):
        """Create temporary subdirectories for test data if needed"""
        print("Creating temporary test subdirectories...")
        
        # Create placeholder directories
        normal_dir = os.path.join(self.test_dir, 'Normal(Healthy skin)')
        abnormal_dir = os.path.join(self.test_dir, 'Abnormal(Ulcer)')
        
        for dir_path in [normal_dir, abnormal_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        
        # This is a temporary solution - in practice, you should organize your test data properly
        print("Note: Test data is not properly organized. Please create 'Normal(Healthy skin)' and 'Abnormal(Ulcer)' subdirectories in TestSet.")
    
    def create_model(self, model_name):
        """Create and compile a model based on the specified architecture"""
        print(f"Creating {model_name} model...")
        
        # Load the pre-trained base model
        if model_name.lower() == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*self.img_size, 3))
        elif model_name.lower() == 'efficientnet':
            base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*self.img_size, 3))
        else:
            raise ValueError(f"Model {model_name} not supported. Use 'resnet50' or 'efficientnet'.")
        
        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Add custom top layers for binary classification
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        
        # Create the full model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model, base_model
    
    def unfreeze_layers(self, model, base_model, unfreeze_percent=0.3):
        """Unfreeze a percentage of the top layers for fine-tuning"""
        print(f"Unfreezing {unfreeze_percent*100}% of the top layers for fine-tuning...")
        
        # Calculate number of layers to unfreeze
        num_layers = len(base_model.layers)
        num_unfreeze = int(num_layers * unfreeze_percent)
        
        # Unfreeze the top layers
        for layer in base_model.layers[-num_unfreeze:]:
            layer.trainable = True
        
        # Recompile the model with a lower learning rate
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate / 10),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, model, model_name):
        """Train the model with callbacks"""
        print(f"Training {model_name} model...")
        
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7),
            ModelCheckpoint(
                os.path.join(self.models_dir, f'{model_name}_best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]
        
        # Train the model (initial training with frozen layers)
        history = model.fit(
            self.train_generator,
            epochs=self.epochs,
            validation_data=self.val_generator,
            callbacks=callbacks
        )
        
        return model, history
    
    def evaluate_model(self, model, model_name):
        """Evaluate the model on test data and generate metrics"""
        print(f"Evaluating {model_name} model...")
        
        # Evaluate on test data
        try:
            test_loss, test_acc = model.evaluate(self.test_generator, verbose=1)
            print(f"{model_name} Test Loss: {test_loss:.4f}")
            print(f"{model_name} Test Accuracy: {test_acc:.4f}")
            
            # Get predictions
            y_pred_prob = model.predict(self.test_generator).ravel()
            y_pred = (y_pred_prob > 0.5).astype(int)
            y_true = self.test_generator.classes
            
            # Generate classification report
            report = classification_report(y_true, y_pred, target_names=list(self.class_labels.values()))
            print(f"\n{model_name} Classification Report:\n{report}")
            
            # Generate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Calculate AUC-ROC
            fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
            roc_auc = auc(fpr, tpr)
            print(f"{model_name} AUC-ROC: {roc_auc:.4f}")
            
            # Save metrics
            metrics = {
                'test_loss': test_loss,
                'test_acc': test_acc,
                'roc_auc': roc_auc,
                'classification_report': report,
                'confusion_matrix': cm,
                'fpr': fpr,
                'tpr': tpr
            }
            
            return metrics
        except Exception as e:
            print(f"Error evaluating model: {e}")
            print("Ensure your TestSet directory has properly organized subdirectories.")
            return None
    
    def plot_results(self, history, metrics, model_name):
        """Plot training history and evaluation metrics"""
        print(f"Generating plots for {model_name}...")
        
        # Plot training & validation accuracy and loss
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        axes[0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title(f'{model_name} Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        
        axes[1].plot(history.history['loss'], label='Training Loss')
        axes[1].plot(history.history['val_loss'], label='Validation Loss')
        axes[1].set_title(f'{model_name} Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'{model_name}_training_history.png'))
        plt.close()
        
        # If metrics are available, plot confusion matrix and ROC curve
        if metrics:
            # Confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', 
                        xticklabels=list(self.class_labels.values()), 
                        yticklabels=list(self.class_labels.values()))
            plt.title(f'{model_name} Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.savefig(os.path.join(self.plots_dir, f'{model_name}_confusion_matrix.png'))
            plt.close()
            
            # ROC Curve
            plt.figure(figsize=(10, 8))
            plt.plot(metrics['fpr'], metrics['tpr'], color='darkorange', lw=2,
                     label=f'ROC curve (area = {metrics['roc_auc']:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{model_name} Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(self.plots_dir, f'{model_name}_roc_curve.png'))
            plt.close()
    
    def compare_models(self, metrics_resnet, metrics_efficientnet):
        """Compare the performance of ResNet18 and EfficientNet-B0"""
        print("Comparing model performances...")
        
        if not metrics_resnet or not metrics_efficientnet:
            print("Cannot compare models due to missing evaluation metrics.")
            return
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'Metric': ['Test Loss', 'Test Accuracy', 'AUC-ROC'],
            'ResNet50': [
                metrics_resnet['test_loss'],
                metrics_resnet['test_acc'],
                metrics_resnet['roc_auc']
            ],
            'EfficientNet-B0': [
                metrics_efficientnet['test_loss'],
                metrics_efficientnet['test_acc'],
                metrics_efficientnet['roc_auc']
            ]
        })
        
        print("\nModel Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Plot ROC curves together
        plt.figure(figsize=(10, 8))
        plt.plot(metrics_resnet['fpr'], metrics_resnet['tpr'], color='blue', lw=2,
                 label=f'ResNet50 ROC (area = {metrics_resnet['roc_auc']:.3f})')
        plt.plot(metrics_efficientnet['fpr'], metrics_efficientnet['tpr'], color='green', lw=2,
                 label=f'EfficientNet-B0 ROC (area = {metrics_efficientnet['roc_auc']:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic - Model Comparison')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.plots_dir, 'model_comparison_roc.png'))
        plt.close()
        
        # Save comparison to CSV
        comparison_df.to_csv(os.path.join(self.plots_dir, 'model_comparison.csv'), index=False)
    
    def run(self):
        """Run the complete diabetic foot ulcer classification pipeline"""
        print("Starting Diabetic Foot Ulcer Classification Project...")
        
        # Prepare data
        self.prepare_data()
        
        # Dictionary to store results
        results = {}
        
        # Train and evaluate ResNet50
        print("\n===== Training ResNet50 =====")
        resnet_model, resnet_base = self.create_model('resnet50')
        resnet_model, resnet_history = self.train_model(resnet_model, 'resnet50')
        
        # Fine-tune ResNet50
        print("\nFine-tuning ResNet50...")
        resnet_model = self.unfreeze_layers(resnet_model, resnet_base)
        resnet_model, resnet_history = self.train_model(resnet_model, 'resnet50_finetuned')
        
        # Evaluate ResNet50
        resnet_metrics = self.evaluate_model(resnet_model, 'ResNet50')
        self.plot_results(resnet_history, resnet_metrics, 'ResNet50')
        results['resnet'] = {'model': resnet_model, 'history': resnet_history, 'metrics': resnet_metrics}
        
        # Save the final ResNet50 model
        resnet_model.save(os.path.join(self.models_dir, 'resnet50_final.h5'))
        
        # Train and evaluate EfficientNet-B0
        print("\n===== Training EfficientNet-B0 =====")
        effnet_model, effnet_base = self.create_model('efficientnet')
        effnet_model, effnet_history = self.train_model(effnet_model, 'efficientnet')
        
        # Fine-tune EfficientNet-B0
        print("\nFine-tuning EfficientNet-B0...")
        effnet_model = self.unfreeze_layers(effnet_model, effnet_base)
        effnet_model, effnet_history = self.train_model(effnet_model, 'efficientnet_finetuned')
        
        # Evaluate EfficientNet-B0
        effnet_metrics = self.evaluate_model(effnet_model, 'EfficientNet-B0')
        self.plot_results(effnet_history, effnet_metrics, 'EfficientNet-B0')
        results['efficientnet'] = {'model': effnet_model, 'history': effnet_history, 'metrics': effnet_metrics}
        
        # Save the final EfficientNet-B0 model
        effnet_model.save(os.path.join(self.models_dir, 'efficientnet_final.h5'))
        
        # Compare models
        if 'metrics' in results['resnet'] and 'metrics' in results['efficientnet']:
            self.compare_models(results['resnet']['metrics'], results['efficientnet']['metrics'])
        
        print("\nDiabetic Foot Ulcer Classification Project Completed!")
        print(f"Models saved in: {self.models_dir}")
        print(f"Plots and results saved in: {self.plots_dir}")

if __name__ == "__main__":
    # Initialize and run the project
    dfu_project = DFUClassification()
    dfu_project.run()