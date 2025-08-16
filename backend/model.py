import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import image_dataset_from_directory
import pathlib

print("TensorFlow version:", tf.__version__)

# --- Configuration ---
DATASET_DIR = '../CNN/dataset_final_augmented_with_feedback'
IMG_HEIGHT = 28
IMG_WIDTH = 28
BATCH_SIZE = 32

# 1. Load the Custom Dataset
data_dir = pathlib.Path(DATASET_DIR)
print(f"Loading data from: {data_dir.resolve()}")

# Create a training dataset (80% of the data)
train_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    label_mode='categorical'
)

# Create a validation dataset (20% of the data)
val_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    label_mode='categorical'
)

class_names = train_ds.class_names
print("Class names found:", class_names)

# Performance optimization for the dataset pipeline
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 2. Build the CNN Model
num_classes = len(class_names)
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# 3. Compile the Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 4. Train the Model
print("\n--- Starting Model Training on Custom Dataset ---")
epochs = 30
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)
print("--- Model Training Complete ---\n")

# 5. Save the Model
model_filename = '../models/sudoku_custom_model_with_feedback.keras'
model.save(model_filename)
print(f"Model saved successfully as {model_filename}")