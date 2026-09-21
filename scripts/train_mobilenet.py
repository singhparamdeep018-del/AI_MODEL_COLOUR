import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# MICRO-MEND AI
# COLOR CLASSIFICATION MODEL
# ============================================================

DATASET = "micromend_color_dataset"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# ============================================================
# LOAD DATA
# ============================================================

train_data = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET, "validation"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_data = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# CLASS NAMES
# ============================================================

class_names = train_data.class_names

print("\nClasses detected:")
for i, name in enumerate(class_names):
    print(i, ":", name)

NUM_CLASSES = len(class_names)

# ============================================================
# PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(AUTOTUNE)
validation_data = validation_data.prefetch(AUTOTUNE)
test_data = test_data.prefetch(AUTOTUNE)

# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# ============================================================
# BASE MODEL
# ============================================================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

# ============================================================
# BUILD MODEL
# ============================================================

inputs = layers.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

x = preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.25)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = models.Model(
    inputs,
    outputs
)

# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# TRAIN
# ============================================================

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS
)

# ============================================================
# TEST
# ============================================================

test_loss, test_accuracy = model.evaluate(
    test_data
)

print("\n==============================")
print("TEST RESULTS")
print("==============================")

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Test Loss: "
    f"{test_loss:.4f}"
)

# ============================================================
# SAVE MODEL
# ============================================================

model.save(
    "micromend_color_model.keras"
)

print(
    "\nModel saved as:"
    " micromend_color_model.keras"
)

# ============================================================
# PLOT TRAINING
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "Micro-Mend AI Training Accuracy"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "training_accuracy.png"
)

plt.show()