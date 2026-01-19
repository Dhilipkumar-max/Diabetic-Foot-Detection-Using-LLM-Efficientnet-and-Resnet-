import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style to match the image (whitegrid is close to the grey grid look)
sns.set_style("whitegrid")

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Data approximation from the image
epochs = [1, 2, 3, 4, 5]
train_acc = [0.932, 0.978, 0.978, 0.995, 0.992]
val_acc = [0.640, 0.800, 0.915, 0.952, 0.961]

# Create the plot
plt.figure(figsize=(8, 6))

# Plot lines
plt.plot(epochs, train_acc, color='red', linewidth=2.5, label='Training Accuracy')
plt.plot(epochs, val_acc, color='green', linewidth=2.5, label='Validation Accuracy')

# Highlight best epoch (Epoch 5)
best_epoch = 5
best_val_acc = 0.961
plt.scatter(best_epoch, best_val_acc, color='blue', s=100, zorder=5, label='best epoch= 5')

# Customization
plt.title('Training and Validation Accuracy', fontsize=14, pad=15)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(loc='lower right', frameon=True)

# Set grid style specifically to look like the image (greyish background with white grid lines is "darkgrid", 
# but "whitegrid" with some tweaks is often safer. The image actually looks like "seaborn-whitegrid" or just standard seaborn default from older versions.
# Let's try to match the grey background if needed, but whitegrid is cleaner. 
# Looking closely at the user image: it has a light grey background.
plt.gca().set_facecolor('#f0f0f0') # Light grey background
plt.grid(True, color='white') # White grid lines

# Set axis limits and ticks to match closely if needed, but auto is usually fine.
# Image x-axis: 1.0 to 5.0
plt.xlim(0.8, 5.2)
plt.ylim(0.60, 1.01)

# Save the plot
output_path = os.path.join('plots', 'custom_accuracy_chart.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Chart saved to {output_path}")
