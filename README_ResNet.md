# ResNet18 Diabetic Foot Ulcer Classification

This repository contains two implementations of a deep learning model using ResNet18 for the classification of diabetic foot ulcer images:
1. A TensorFlow implementation (`resnet_classification.py`)
2. A PyTorch implementation (`resnet_pytorch.py`)

Both implementations train on the provided dataset and evaluate performance, with a focus on accuracy metrics.

## Overview

Both implementations:
- Use ResNet18 with transfer learning for binary classification (Normal vs. Abnormal)
- Include data preprocessing, augmentation, and validation split
- Perform initial training with frozen base layers followed by fine-tuning
- Evaluate model performance with accuracy, confusion matrix, and classification report
- Save trained models and evaluation plots

## Prerequisites

### For TensorFlow Implementation
Before running the script, ensure you have installed the required packages:

```bash
pip install -r requirements.txt
```

The requirements include:
- tensorflow>=2.9.0
- numpy>=1.22.0
- pandas>=1.4.0
- matplotlib>=3.5.0
- seaborn>=0.11.0
- scikit-learn>=1.1.0

### For PyTorch Implementation
You'll need additional PyTorch packages:

```bash
pip install torch torchvision torchaudio
```

## Dataset Structure

Both implementations expect the dataset to be organized in the following structure:

```
DFU/
├── Patches/
│   ├── Abnormal(Ulcer)/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── Normal(Healthy skin)/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
├── TestSet/
│   ├── Abnormal(Ulcer)/
│   │   ├── image1.jpg
│   │   └── ...
│   └── Normal(Healthy skin)/
│       ├── image1.jpg
│       └── ...
├── resnet_classification.py       # TensorFlow implementation
└── resnet_pytorch.py               # PyTorch implementation
```

**Important:** Before running either script, make sure the TestSet directory is organized into class subdirectories. You can use the `prepare_test_data.py` script to help with this process.

## Usage

### For TensorFlow Implementation
1. Ensure your dataset is structured correctly
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the TensorFlow script:
   ```bash
   python resnet_classification.py
   ```

### For PyTorch Implementation
1. Ensure your dataset is structured correctly
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   pip install torch torchvision torchaudio
   ```
3. Run the PyTorch script:
   ```bash
   python resnet_pytorch.py
   ```

## Script Workflows

### TensorFlow Implementation Workflow

1. **Data Preparation:**
   - Sets up paths to the dataset
   - Creates training, validation, and test datasets
   - Applies data augmentation (resizing to 224×224, normalization, random flips, rotations)

2. **Model Creation:**
   - Loads a pre-trained ResNet18 model
   - Modifies the final classification layer for binary classification (2 outputs)
   - Compiles the model with CrossEntropyLoss and Adam optimizer

3. **Training Phase 1:**
   - Trains the model with frozen base layers
   - Uses early stopping to prevent overfitting
   - Saves the best model after this phase

4. **Fine-tuning Phase:**
   - Unfreezes some base layers
   - Continues training with a lower learning rate
   - Saves the final fine-tuned model

5. **Evaluation:**
   - Evaluates the model on the test dataset
   - Calculates accuracy and generates confusion matrix
   - Plots training/validation accuracy and loss curves
   - Plots ROC curve and confusion matrix

### PyTorch Implementation Workflow

1. **Data Preparation:**
   - Sets up paths to the dataset
   - Uses `torchvision.datasets.ImageFolder` for data loading
   - Creates data loaders with appropriate batch sizes
   - Applies data augmentation (resizing to 224×224, normalization, random flips, rotations)

2. **Model Creation:**
   - Loads a pre-trained ResNet18 model from `torchvision.models`
   - Replaces the last fully connected layer (fc) with 2 outputs
   - Sets initial layers to requires_grad=False for transfer learning

3. **Training Phase 1:**
   - Trains the model with frozen base layers
   - Uses CrossEntropyLoss and Adam optimizer
   - Saves the best model after this phase

4. **Fine-tuning Phase:**
   - Unfreezes all layers
   - Continues training with a lower learning rate
   - Saves the final fine-tuned model

5. **Evaluation:**
   - Evaluates the model on the test dataset
   - Computes accuracy, precision, recall, F1-score
   - Generates confusion matrix
   - Plots training/validation accuracy and loss curves

## Key Results

### TensorFlow Implementation Results
After running the script, you will get:
- Trained models: `models/resnet18_initial.keras` and `models/resnet18_finetuned.keras`
- Evaluation metrics: Accuracy, confusion matrix, ROC curve with AUC score
- Generated files in `plots/` directory with visualizations
- `model_evaluation_results.txt` with detailed evaluation results

### PyTorch Implementation Results
After running the script, you will get:
- Trained models: `models/resnet18_initial.pth` and `models/resnet18_finetuned.pth`
- Evaluation metrics: Accuracy, precision, recall, F1-score, confusion matrix
- Generated files in `plots/` directory with visualizations
- `model_evaluation_results.txt` with detailed evaluation results

## Next Steps

- Use the trained models for inference on new images
- Experiment with different hyperparameters (learning rate, batch size, etc.)
- Try different data augmentation techniques
- Explore other pre-trained models (e.g., ResNet50, EfficientNet)
- Compare the performance of TensorFlow and PyTorch implementations

## Notes

- Both implementations assume a binary classification task (Normal vs. Abnormal)
- Training time may vary depending on your hardware (GPU recommended)
- The PyTorch implementation includes comprehensive evaluation metrics (accuracy, precision, recall, F1-score)