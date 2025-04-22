import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Dataset Paths
IMAGE_DIR = "HAM10000_images_part_1"
CSV_PATH = "HAM10000_metadata.csv"

# Load metadata
df = pd.read_csv(CSV_PATH)
df['image_id'] = df['image_id'] + ".jpg"
df['path'] = df['image_id'].apply(lambda x: os.path.join(IMAGE_DIR, x))
df = df[df['path'].apply(os.path.exists)]
df['label'] = df['dx']
label_map = {label: idx for idx, label in enumerate(df['label'].unique())}
df['label_idx'] = df['label'].map(label_map)

# Train/Validation Split
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label_idx'], random_state=42)

# Image Preprocessing
IMG_SIZE = 224
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(
    train_df, x_col='path', y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
val_gen = val_datagen.flow_from_dataframe(
    val_df, x_col='path', y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Load Pretrained DenseNet121
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
x = GlobalAveragePooling2D()(base_model.output)
output = Dense(len(label_map), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# Freeze base model initially
for layer in base_model.layers:
    layer.trainable = False

# Compile model (initial training)
model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
history1 = model.fit(train_gen, validation_data=val_gen, epochs=5)

# Fine-tune the full model
for layer in model.layers:
    layer.trainable = True

model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
history2 = model.fit(train_gen, validation_data=val_gen, epochs=10)

# Save model
model.save("ham10000_densenet121_finetuned.h5")

# Combine histories
def combine_histories(h1, h2):
    combined = {}
    for k in h1.history:
        combined[k] = h1.history[k] + h2.history[k]
    return combined

history = combine_histories(history1, history2)

# Plot Training History
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title("Model Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Evaluation
val_preds = model.predict(val_gen)
y_pred = np.argmax(val_preds, axis=1)
y_true = val_gen.classes

cm = confusion_matrix(y_true, y_pred)
cr = classification_report(y_true, y_pred, target_names=val_gen.class_indices.keys())

# Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=val_gen.class_indices.keys(),
            yticklabels=val_gen.class_indices.keys())
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

print("Classification Report:\n", cr)

# Sample Predictions Visualization
from tensorflow.keras.preprocessing import image
sample_df = val_df.sample(5, random_state=1)
for _, row in sample_df.iterrows():
    img_path = row['path']
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    pred = model.predict(np.expand_dims(img_array, axis=0))[0]
    pred_label = list(label_map.keys())[np.argmax(pred)]

    plt.imshow(img)
    plt.axis('off')
    plt.title(f"True: {row['label']} | Predicted: {pred_label}")
    plt.show()
