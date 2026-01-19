import os
import shutil
import random
from pathlib import Path

def main():
    # Get base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define source directories
    patches_dir = os.path.join(base_dir, 'Patches')
    testset_dir = os.path.join(base_dir, 'TestSet')
    
    # Define target directory structure
    dataset_dir = os.path.join(base_dir, 'dataset')
    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')
    
    # Create target directories
    print("Creating dataset directory structure...")
    os.makedirs(os.path.join(train_dir, 'Abnormal(Ulcer)'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'Normal (Healthy skin)'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'Abnormal(Ulcer)'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'Normal (Healthy skin)'), exist_ok=True)
    
    # Function to copy images
    def copy_images(src_dir, dst_dir, limit=None):
        if not os.path.exists(src_dir):
            print(f"Warning: Source directory {src_dir} does not exist")
            return 0
        
        # Get image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_files = [f for f in os.listdir(src_dir) if Path(f).suffix in image_extensions]
        
        # Apply limit if specified
        if limit is not None:
            image_files = image_files[:limit]
        
        # Copy files
        for f in image_files:
            src_path = os.path.join(src_dir, f)
            dst_path = os.path.join(dst_dir, f)
            try:
                shutil.copy(src_path, dst_path)
            except Exception as e:
                print(f"Error copying {f}: {e}")
        
        return len(image_files)
    
    # Copy training images from Patches directory
    print("Copying training images from Patches directory...")
    abnormal_train_src = os.path.join(patches_dir, 'Abnormal(Ulcer)')
    abnormal_train_dst = os.path.join(train_dir, 'Abnormal(Ulcer)')
    normal_train_src = os.path.join(patches_dir, 'Normal(Healthy skin)')
    normal_train_dst = os.path.join(train_dir, 'Normal (Healthy skin)')
    
    abnormal_count = copy_images(abnormal_train_src, abnormal_train_dst)
    normal_count = copy_images(normal_train_src, normal_train_dst)
    
    print(f"Copied {abnormal_count} abnormal (ulcer) training images")
    print(f"Copied {normal_count} normal (healthy skin) training images")
    
    # Process TestSet directory
    print("Processing TestSet directory...")
    
    # If TestSet is not already organized, we need to split it
    # For now, let's assume we need to split it randomly between abnormal and normal
    # Note: In a real scenario, you would want to manually classify these images
    
    if os.path.exists(testset_dir):
        # Get all image files in TestSet
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        test_images = [f for f in os.listdir(testset_dir) if Path(f).suffix in image_extensions]
        
        if test_images:
            print(f"Found {len(test_images)} images in TestSet")
            
            # Randomly split test images (this is a temporary solution)
            # In practice, you should manually classify these images
            random.shuffle(test_images)
            split_point = len(test_images) // 2
            
            abnormal_test_dst = os.path.join(test_dir, 'Abnormal(Ulcer)')
            normal_test_dst = os.path.join(test_dir, 'Normal (Healthy skin)')
            
            # Copy to respective directories
            for i, f in enumerate(test_images):
                src_path = os.path.join(testset_dir, f)
                if i < split_point:
                    dst_path = os.path.join(abnormal_test_dst, f)
                else:
                    dst_path = os.path.join(normal_test_dst, f)
                
                try:
                    shutil.copy(src_path, dst_path)
                except Exception as e:
                    print(f"Error copying {f}: {e}")
            
            print(f"Copied {split_point} images to test/Abnormal(Ulcer)")
            print(f"Copied {len(test_images) - split_point} images to test/Normal(Healthy skin)")
            print("\nNote: Test images have been randomly split. For accurate evaluation, you should manually classify these images.")
        else:
            print("No image files found in TestSet directory")
    else:
        print("TestSet directory does not exist")
    
    print("\n✅ Dataset organization completed!")
    print("\nNext steps:")
    print("1. Verify the dataset structure in the 'dataset' directory")
    print("2. Manually review and correct the test set classification if needed")
    print("3. Run the ResNet PyTorch script: python resnet_pytorch.py")

if __name__ == "__main__":
    main()