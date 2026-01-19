import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import __version__ as sklearn_version

class EnvironmentChecker:
    def __init__(self):
        self.required_packages = {
            'tensorflow': tf,
            'numpy': np,
            'pandas': pd,
            'matplotlib': plt,
            'seaborn': sns,
            'scikit-learn': sklearn_version
        }
        
        # Check GPU availability
        self.gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    
    def check_packages(self):
        """Check if required packages are installed and display their versions"""
        print("===== Checking Installed Packages ======")
        
        all_installed = True
        for package_name, package in self.required_packages.items():
            try:
                if package_name == 'scikit-learn':
                    version = package
                else:
                    version = package.__version__
                print(f"✓ {package_name} version {version} is installed")
            except Exception as e:
                print(f"✗ {package_name} is not installed correctly: {e}")
                all_installed = False
        
        return all_installed
    
    def check_system(self):
        """Check system information relevant to the project"""
        print("\n===== System Information ======")
        
        # Check GPU
        if self.gpu_available:
            print("✓ GPU is available for training")
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                print(f"   - {gpu.name}")
        else:
            print("✗ GPU not detected. Training will use CPU (may be slower)")
        
        # Check TensorFlow build
        print(f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
        
        # Check memory
        try:
            from psutil import virtual_memory
            mem = virtual_memory()
            print(f"System memory: {mem.total / (1024**3):.2f} GB available")
        except ImportError:
            print("psutil not installed, cannot check system memory")
    
    def check_directory_structure(self):
        """Check if the basic directory structure exists"""
        print("\n===== Project Directory Structure ======")
        
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Check main directories
        directories_to_check = [
            'Original Images',
            'Patches',
            'TestSet'
        ]
        
        for dir_name in directories_to_check:
            dir_path = os.path.join(base_dir, dir_name)
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                print(f"✓ Directory '{dir_name}' exists")
                
                # For Patches directory, check if it has subdirectories
                if dir_name == 'Patches':
                    subdirs = [d for d in os.listdir(dir_path) 
                              if os.path.isdir(os.path.join(dir_path, d))]
                    if subdirs:
                        print(f"   Contains {len(subdirs)} class subdirectories: {', '.join(subdirs)}")
                    else:
                        print(f"   WARNING: '{dir_name}' directory doesn't contain class subdirectories")
                        print("   Please create 'Normal(Healthy skin)' and 'Abnormal(Ulcer)' subdirectories")
            else:
                print(f"✗ Directory '{dir_name}' does not exist or is not a directory")
    
    def run_check(self):
        """Run all checks and provide a summary"""
        print("Running environment check for Diabetic Foot Ulcer Classification Project...\n")
        
        packages_ok = self.check_packages()
        self.check_system()
        self.check_directory_structure()
        
        print("\n===== Summary ======")
        if packages_ok:
            print("✅ All required packages are installed correctly.")
        else:
            print("❌ Some required packages are missing or not installed correctly.")
            print("   Please run 'pip install -r requirements.txt' to fix this.")
        
        # Final instructions
        print("\nNext Steps:")
        print("1. Make sure your dataset is properly organized in the directories")
        print("2. Run 'prepare_test_data.py' to set up your test data structure")
        print("3. Manually classify and move test images to appropriate subdirectories")
        print("4. Run 'dfu_classification.py' to start training and evaluating models")

if __name__ == "__main__":
    checker = EnvironmentChecker()
    checker.run_check()