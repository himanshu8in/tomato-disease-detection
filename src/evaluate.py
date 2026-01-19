import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# CONFIGURATION
# ===============================
MODEL_PATH = r"C:\Users\himan\OneDrive\Desktop\Plant_disease\models\tomato_efficientnet_model.keras"   
TEST_DIR = r"C:\Users\himan\OneDrive\Desktop\Plant_disease\dataset\test"                     
IMG_SIZE = 224
BATCH_SIZE = 32

# ===============================
# LOAD MODEL
# ===============================
print("Loading trained EfficientNetB0 model...")
model = load_model(MODEL_PATH)

# ===============================
# TEST DATA GENERATOR
# ===============================
test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ===============================
# EVALUATE MODEL
# ===============================
print("\nEvaluating model on test data...")
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)

print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# ===============================
# PREDICTIONS
# ===============================
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes
class_names = list(test_generator.class_indices.keys())

# ===============================
# CLASSIFICATION REPORT
# ===============================
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# ===============================
# CONFUSION MATRIX
# ===============================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix - EfficientNetB0")
plt.tight_layout()
plt.show()
