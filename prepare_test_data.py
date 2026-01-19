import os
import shutil

class TestDataPreparer:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.test_dir = os.path.join(self.base_dir, 'TestSet')
        self.normal_dir = os.path.join(self.test_dir, 'Normal(Healthy skin)')
        self.abnormal_dir = os.path.join(self.test_dir, 'Abnormal(Ulcer)')
    
    def create_test_subdirectories(self):
        """Create the necessary subdirectories for test data"""
        print(f"Preparing test data structure in: {self.test_dir}")
        
        # Create class directories if they don't exist
        for dir_path in [self.normal_dir, self.abnormal_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"Created directory: {dir_path}")
            else:
                print(f"Directory already exists: {dir_path}")
        
        # List all files in TestSet directory
        all_files = [f for f in os.listdir(self.test_dir) 
                    if os.path.isfile(os.path.join(self.test_dir, f))]
        
        # Count image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
        image_files = [f for f in all_files 
                      if any(f.lower().endswith(ext) for ext in image_extensions)]
        
        print(f"Found {len(image_files)} image files in the TestSet directory.")
        
        # Create a sample mapping file to help with manual sorting
        mapping_file_path = os.path.join(self.test_dir, 'image_mapping_guide.csv')
        with open(mapping_file_path, 'w') as f:
            f.write('Filename,Suggested_Class\n')
            for img_file in image_files:
                # This is just a placeholder - the user needs to manually classify
                # We'll make some basic guesses based on filename keywords
                suggestion = 'Need manual classification'
                if any(keyword in img_file.lower() for keyword in ['normal', 'healthy', 'callus', 'corn']):
                    suggestion = 'Possibly Normal(Healthy skin)'
                elif any(keyword in img_file.lower() for keyword in ['ulcer', 'diabetic', 'wound', 'abnormal']):
                    suggestion = 'Possibly Abnormal(Ulcer)'
                f.write(f'{img_file},{suggestion}\n')
        
        print(f"Created image mapping guide: {mapping_file_path}")
        
        # Provide instructions
        print("\n\nIMPORTANT INSTRUCTIONS:")
        print("======================")
        print("1. The TestSet directory now has two subdirectories:")
        print(f"   - {self.normal_dir}")
        print(f"   - {self.abnormal_dir}")
        print("")
        print("2. YOU NEED TO MANUALLY SORT THE IMAGES into these subdirectories based on their actual classification.")
        print("")
        print("3. A CSV file 'image_mapping_guide.csv' has been created to help you with this task.")
        print("   This file suggests possible classifications based on filename keywords,")
        print("   but YOU SHOULD VERIFY EACH IMAGE MANUALLY.")
        print("")
        print("4. To move images, you can:")
        print("   - Drag and drop them using your file explorer")
        print("   - Or use the provided move_images function in this script (requires manual editing)")
        print("")
        print("5. Once all images are properly sorted, you can run dfu_classification.py to evaluate the models.")
    
    def move_images(self, classification_dict):
        """Move images based on a classification dictionary
        
        Args:
            classification_dict: Dictionary with {filename: class_label} where
                                class_label is 'normal' or 'abnormal'
        """
        for filename, class_label in classification_dict.items():
            src_path = os.path.join(self.test_dir, filename)
            
            if not os.path.exists(src_path):
                print(f"File not found: {filename}")
                continue
            
            if class_label.lower() == 'normal':
                dest_path = os.path.join(self.normal_dir, filename)
            elif class_label.lower() == 'abnormal':
                dest_path = os.path.join(self.abnormal_dir, filename)
            else:
                print(f"Invalid class label for {filename}: {class_label}")
                continue
            
            try:
                shutil.move(src_path, dest_path)
                print(f"Moved {filename} to {class_label} directory")
            except Exception as e:
                print(f"Error moving {filename}: {e}")

if __name__ == "__main__":
    preparer = TestDataPreparer()
    preparer.create_test_subdirectories()
    
    # Example usage of move_images function (commented out)
    """
    # To use this function, create a dictionary mapping filenames to their classes
    # Then uncomment the following lines
    
    # classification = {
    #     'image1.jpg': 'normal',
    #     'image2.jpg': 'abnormal',
    #     # Add more mappings as needed
    # }
    # preparer.move_images(classification)
    """