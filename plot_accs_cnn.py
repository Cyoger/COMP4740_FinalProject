import matplotlib.pyplot as plt

# Accuracy scores and corresponding model names
accuracy_scores = [62, 69.27, 70.98, 87.77]
model_names = ['CNN 1', 'CNN 2', 'CNN 3', 'CNN 4']

# Create bar graph
plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracy_scores, color='skyblue')

# Add labels and title
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Scores of Different Models')

# Add the accuracy scores on top of each bar
for i, v in enumerate(accuracy_scores):
    plt.text(i, v + 1, str(v) + '%', ha='center')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Show plot
plt.tight_layout()
plt.show()
