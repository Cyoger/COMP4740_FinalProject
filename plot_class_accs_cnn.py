import matplotlib.pyplot as plt

# Class labels and corresponding classification scores
class_labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
classification_scores = [89.1, 92.7, 80.9, 74.9, 91.0, 76.9, 94.6, 90.3, 94.3, 93.0]

# Create horizontal bar graph
plt.figure(figsize=(10, 6))
plt.barh(class_labels, classification_scores, color='lightgreen')

# Add labels and title
plt.xlabel('Classification Scores (%)')
plt.ylabel('Class Labels')
plt.title('Classification Scores for Different Classes')

# Add the classification scores on the right of each bar
for i, v in enumerate(classification_scores):
    plt.text(v + 0.5, i, str(v) + '%', va='center')

# Show plot
plt.tight_layout()
plt.show()
