import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

# -------------------------------
# 1. Training Loss & Accuracy Graph
# -------------------------------
epochs = list(range(1, 11))
train_loss = [1.0, 0.8, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.28, 0.25]
val_loss = [1.1, 0.9, 0.7, 0.55, 0.5, 0.45, 0.42, 0.38, 0.35, 0.32]
train_acc = [0.5, 0.6, 0.65, 0.7, 0.73, 0.75, 0.78, 0.8, 0.82, 0.85]
val_acc = [0.48, 0.57, 0.63, 0.68, 0.7, 0.72, 0.75, 0.77, 0.8, 0.82]

plt.figure(figsize=(8,4))
plt.plot(epochs, train_loss, label='Train Loss', marker='o')
plt.plot(epochs, val_loss, label='Val Loss', marker='o')
plt.plot(epochs, train_acc, label='Train Accuracy', marker='x')
plt.plot(epochs, val_acc, label='Val Accuracy', marker='x')
plt.xlabel('Epochs')
plt.ylabel('Loss / Accuracy')
plt.title('Training & Validation Metrics')
plt.legend()
plt.grid(True)
plt.savefig('projects/image-classification/images/training_metrics.png')
plt.close()

# -------------------------------
# 2. Confusion Matrix
# -------------------------------
y_true = [0, 1, 0, 1, 0, 1, 0, 0, 1, 1]
y_pred = [0, 1, 0, 0, 0, 1, 0, 1, 1, 1]

ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
plt.title("Confusion Matrix")
plt.savefig('projects/image-classification/images/confusion_matrix.png')
plt.close()

# -------------------------------
# 3. Sample Predictions
# -------------------------------
images = np.random.rand(4, 28, 28)  # Random placeholder images
labels = ['Cat', 'Dog', 'Dog', 'Cat']
preds = ['Cat', 'Dog', 'Cat', 'Cat']

plt.figure(figsize=(8,4))
for i in range(4):
    plt.subplot(1,4,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(f"True: {labels[i]}\nPred: {preds[i]}")
    plt.axis('off')
plt.suptitle("Sample Model Predictions")
plt.savefig('projects/image-classification/images/sample_predictions.png')
plt.close()

