import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

DATASET_DIR = r"F:\AI\dataset\fruits_dietbot"

IMG_SIZE = (100, 100)
BATCH_SIZE = 32
EPOCHS = 8

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
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    data_aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ])

    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        data_aug,
        layers.Rescaling(1./255),

        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    y_true, y_pred = [], []
    for images, labels in val_ds:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.tolist())

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    project_models = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(project_models, exist_ok=True)

    model.save(os.path.join(project_models, "fruit_model.h5"))
    with open(os.path.join(project_models, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    print("\nSaved: models/fruit_model.h5 and models/labels.json")

if __name__ == "__main__":
    main()