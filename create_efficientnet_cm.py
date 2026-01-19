import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

os.makedirs('plots', exist_ok=True)

# Data based on image (a) EfficientNetB0
# [[TP, FN], [FP, TN]] structure usually, but let's match the image labels.
# Image Grid:
# Y-axis (Actual): Abnormal(Ulcer) [Top], Normal(Healthy skin) [Bottom]
# X-axis (Predicted): Abnormal(Ulcer) [Left], Normal(Healthy skin) [Right]
# Values (a):
# Top-Left: 101
# Top-Right: 2
# Bottom-Left: 1
# Bottom-Right: 107
# This is:
# [[101, 2],
#  [1, 107]]

cm = np.array([[101, 2], [1, 107]])
labels = ['Abnormal(Ulcer)', 'Normal(Healthy skin)']

plt.figure(figsize=(6, 5))

# Create heatmap
# cmap='Blues' matches the blue/white style
# cbar=False to match the clean look if desired, or keep it. Image doesn't show cbar clearly but the coloring implies it.
# annot=True to show numbers.
# fmt='d' for integers.
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                 xticklabels=labels, yticklabels=labels,
                 annot_kws={"size": 14})

# Styling
plt.title('EfficientNetB0', fontsize=14, pad=10)
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)

# Save
output_path = os.path.join('plots', 'efficientnet_cm_chart.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"EfficientNet CM saved to {output_path}")
