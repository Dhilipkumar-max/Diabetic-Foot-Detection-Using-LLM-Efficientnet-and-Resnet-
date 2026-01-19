import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

class DFUPredictor:
    def __init__(self, model_path, model_type='resnet'):
        # Load the trained model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model_type = model_type
        
        if model_type == 'resnet':
            # Load the ResNet18 model
            self.model = models.resnet18(pretrained=False)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: Normal and Abnormal
        elif model_type == 'efficientnet':
            # Load the EfficientNet-B0 model
            self.model = models.efficientnet_b0(pretrained=False)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, 2)  # 2 classes: Normal and Abnormal
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Load the trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define the image transformation
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Class labels
        self.classes = ['Normal(Healthy skin)', 'Abnormal(Ulcer)']
    
    def predict_image(self, image_path):
        # Open and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            
            # Get probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
        return {
            'class_index': predicted.item(),
            'class_name': self.classes[predicted.item()],
            'confidence': probabilities[predicted.item()].item() * 100,
            'probabilities': {
                self.classes[i]: prob.item() * 100 for i, prob in enumerate(probabilities)
            }
        }
    
    def display_prediction(self, image_path, prediction):
        # Display the image with prediction
        image = Image.open(image_path).convert('RGB')
        
        plt.figure(figsize=(10, 6))
        plt.imshow(image)
        
        # Create title with prediction and confidence
        title = f"Prediction: {prediction['class_name']}\nConfidence: {prediction['confidence']:.2f}%"
        plt.title(title)
        plt.axis('off')
        
        # Show probabilities
        for class_name, prob in prediction['probabilities'].items():
            print(f"{class_name}: {prob:.2f}%")
        
        plt.show()

def main():
    # Test directory
    test_dir = 'TestSet'
    
    print("\nDFU Classification Model Runner")
    print("===============================")
    print("Select model to use:")
    print("1. ResNet18")
    print("2. EfficientNet-B0")
    model_choice = input("Enter your choice (1 or 2): ")
    
    if model_choice == '1':
        model_path = 'models/pytorch_resnet18_final.pth'
        model_type = 'resnet'
        print("Using ResNet18 model")
    elif model_choice == '2':
        model_path = 'models/pytorch_efficientnet_b0_final.pth'
        model_type = 'efficientnet'
        print("Using EfficientNet-B0 model")
    else:
        print("Invalid choice. Using ResNet18 model by default.")
        model_path = 'models/pytorch_resnet18_final.pth'
        model_type = 'resnet'
    
    # Create predictor
    predictor = DFUPredictor(model_path, model_type)
    
    # Ask user if they want to test a single image or all images
    print("\nTest Options:")
    print("1. Test a single image")
    print("2. Test multiple images")
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == '1':
        # List all images in the test directory
        image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        
        print("\nAvailable test images:")
        for i, img in enumerate(image_files):
            print(f"{i+1}. {img}")
        
        # Get user selection
        try:
            selection = int(input("\nEnter the number of the image to test: ")) - 1
            if 0 <= selection < len(image_files):
                image_path = os.path.join(test_dir, image_files[selection])
                
                # Make prediction
                prediction = predictor.predict_image(image_path)
                
                # Display results
                print(f"\nResults for {image_files[selection]}:")
                print(f"Prediction: {prediction['class_name']}")
                print(f"Confidence: {prediction['confidence']:.2f}%")
                
                # Display image with prediction
                predictor.display_prediction(image_path, prediction)
            else:
                print("Invalid selection.")
        except ValueError:
            print("Please enter a valid number.")
    
    elif choice == '2':
        # Ask how many images to test
        try:
            num_images = int(input("How many images do you want to test? "))
            
            # Get all image files
            image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
            
            # Select random images if there are more than requested
            import random
            if len(image_files) > num_images:
                image_files = random.sample(image_files, num_images)
            else:
                print(f"Only {len(image_files)} images available. Testing all of them.")
            
            # Process each image
            for img_file in image_files:
                image_path = os.path.join(test_dir, img_file)
                
                # Make prediction
                prediction = predictor.predict_image(image_path)
                
                # Display results
                print(f"\nResults for {img_file}:")
                print(f"Prediction: {prediction['class_name']}")
                print(f"Confidence: {prediction['confidence']:.2f}%")
                
                # Display image with prediction
                predictor.display_prediction(image_path, prediction)
                
                # Wait for user to continue
                input("Press Enter to continue to the next image...")
                
        except ValueError:
            print("Please enter a valid number.")
    
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()