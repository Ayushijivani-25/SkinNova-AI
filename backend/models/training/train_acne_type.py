"""
SkinNova AI – Training Script 2: Acne Type Classifier
Dataset: AcneDataset  (Blackheads / Cyst / Papules / Pustules / Whiteheads)
Architecture: EfficientNetB0 fine-tuned

Run:
    python models/training/train_acne_type.py
"""

import os, json
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "../../data/AcneDataset")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../acne_type_model")
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EPOCHS_FROZEN   = 12
EPOCHS_UNFREEZE = 25
NUM_CLASSES = 5
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Class weights (handle imbalance: Whiteheads << Cyst) ─────────────────────
def compute_class_weights(train_gen):
    from sklearn.utils.class_weight import compute_class_weight
    labels = train_gen.classes
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return dict(enumerate(weights))


# ── Generators ────────────────────────────────────────────────────────────────
def build_generators():
    preprocess = tf.keras.applications.efficientnet.preprocess_input

    train_aug = ImageDataGenerator(
        preprocessing_function=preprocess,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=0.2,
        shear_range=0.1,
        brightness_range=[0.75, 1.25],
        fill_mode="nearest",
    )
    val_aug = ImageDataGenerator(preprocessing_function=preprocess)

    train_gen = train_aug.flow_from_directory(
        os.path.join(DATA_DIR, "train"),
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=True,
    )
    val_gen = val_aug.flow_from_directory(
        os.path.join(DATA_DIR, "valid"),
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=False,
    )
    test_gen = val_aug.flow_from_directory(
        os.path.join(DATA_DIR, "test"),
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=False,
    )
    return train_gen, val_gen, test_gen


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return Model(inputs, outputs), base


# ── Confusion Matrix plot ─────────────────────────────────────────────────────
def save_confusion_matrix(model, test_gen, class_names, output_dir):
    preds = model.predict(test_gen)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names,
                yticklabels=class_names, cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Acne Type – Confusion Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=120)
    print(f"\nClassification Report:\n{classification_report(y_true, y_pred, target_names=class_names)}")


# ── Main ──────────────────────────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("SkinNova – Acne Type Model Training")
    print("=" * 60)

    train_gen, val_gen, test_gen = build_generators()
    class_names = list(train_gen.class_indices.keys())
    print(f"Classes: {class_names}")
    print(f"Train: {train_gen.n}  Val: {val_gen.n}  Test: {test_gen.n}")

    with open(os.path.join(OUTPUT_DIR, "class_indices.json"), "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)

    class_weights = compute_class_weights(train_gen)
    print(f"Class weights: {class_weights}")

    model, base = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        ModelCheckpoint(os.path.join(OUTPUT_DIR, "best_phase1.h5"),
                        save_best_only=True, monitor="val_accuracy", mode="max"),
        EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7),
    ]

    print("\n[Phase 1] Frozen base training …")
    h1 = model.fit(
        train_gen, epochs=EPOCHS_FROZEN, validation_data=val_gen,
        callbacks=callbacks, class_weight=class_weights,
    )

    print("\n[Phase 2] Fine-tuning …")
    base.trainable = True
    for layer in base.layers[:-50]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-6),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks[0].filepath = os.path.join(OUTPUT_DIR, "best_phase2.h5")

    h2 = model.fit(
        train_gen, epochs=EPOCHS_UNFREEZE, validation_data=val_gen,
        callbacks=callbacks, class_weight=class_weights,
    )

    loss, acc = model.evaluate(test_gen)
    print(f"\nTest Loss: {loss:.4f}  |  Test Acc: {acc:.4f}")

    save_confusion_matrix(model, test_gen, class_names, OUTPUT_DIR)

    model.save(os.path.join(OUTPUT_DIR, "model.h5"))
    print(f"Model saved → {OUTPUT_DIR}/model.h5")


if __name__ == "__main__":
    train()
