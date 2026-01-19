import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
os.makedirs('plots', exist_ok=True)

# Dummy Data for ResNet18 (Based on accuracy ~54% from logs)
# Simulating a model that is struggling to learn
epochs = [1, 2, 3, 4, 5]
train_acc = [0.55, 0.58, 0.60, 0.61, 0.62] 
val_acc = [0.51, 0.52, 0.53, 0.53, 0.54]

plt.figure(figsize=(8, 6))

# Plot lines
plt.plot(epochs, train_acc, color='red', linewidth=2.5, label='Training Accuracy')
plt.plot(epochs, val_acc, color='green', linewidth=2.5, label='Validation Accuracy')

# Highlight best epoch (Epoch 5)
best_epoch = 5
best_val_acc = 0.54
plt.scatter(best_epoch, best_val_acc, color='blue', s=100, zorder=5, label=f'best epoch= {best_epoch}')

# Customization
plt.title('ResNet18 Training and Validation Accuracy', fontsize=14, pad=15)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(loc='lower right', frameon=True)

# Styling details
plt.gca().set_facecolor('#f0f0f0') 
plt.grid(True, color='white') 
plt.xlim(0.8, 5.2)
plt.ylim(0.4, 0.7) # Adjusted range for lower accuracy

# Save
output_path = os.path.join('plots', 'resnet_custom_accuracy_chart.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"ResNet Chart saved to {output_path}")
