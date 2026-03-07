#!/usr/bin/env python3
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

DATASET_DIR = r"F:\AI\dataset\fruits_dietbot"

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS_FROZEN = 5
EPOCHS_FINETUNE = 3

def main():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print("Classes:", class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    data_aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
    ])

    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = data_aug(inputs)
    x = layers.Rescaling(1./127.5, offset=-1)(x)  # 0..255 -> -1..1
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\n--- Training (Frozen base) ---")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FROZEN)

    print("\n--- Fine-tuning (last layers) ---")
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE)

    y_true, y_pred = [], []
    for images, labels in val_ds:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.tolist())

    print("\nClassification Report (MobileNetV2):")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("\nConfusion Matrix (MobileNetV2):")
    print(confusion_matrix(y_true, y_pred))

    project_models = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(project_models, exist_ok=True)

    model.save(os.path.join(project_models, "fruit_mobilenet.h5"))
    with open(os.path.join(project_models, "labels_mobilenet.json"), "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    print("\nSaved: models/fruit_mobilenet.h5 and models/labels_mobilenet.json")

if __name__ == "__main__":
    main()