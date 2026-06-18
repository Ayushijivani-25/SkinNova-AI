"""
SkinNova AI – Training Script 1: Skin Type Classifier
Dataset: Oily-Dry-Skin-Types  (dry / normal / oily)
Architecture: EfficientNetB0 fine-tuned

Run:
    python models/training/train_skin_type.py
"""

import os, sys
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "../../data/Oily-Dry-Skin-Types")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../skin_type_model")
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EPOCHS_FROZEN  = 10   # train only classifier head
EPOCHS_UNFREEZE = 20  # fine-tune top layers
NUM_CLASSES = 3       # dry / normal / oily
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Data Generators ──────────────────────────────────────────────────────────
def build_generators():
    train_aug = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
    )
    val_aug = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
    )

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
    base.trainable = False  # frozen initially

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs, outputs)
    return model, base


# ── Callbacks ─────────────────────────────────────────────────────────────────
def get_callbacks(name_prefix):
    return [
        ModelCheckpoint(
            os.path.join(OUTPUT_DIR, f"{name_prefix}_best.h5"),
            save_best_only=True, monitor="val_accuracy", mode="max", verbose=1,
        ),
        EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
        TensorBoard(log_dir=os.path.join(OUTPUT_DIR, "logs")),
    ]


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_history(h1, h2, output_dir):
    acc  = h1.history["accuracy"]  + h2.history["accuracy"]
    vacc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    fig, ax = plt.subplots()
    ax.plot(acc, label="train acc")
    ax.plot(vacc, label="val acc")
    ax.axvline(len(h1.history["accuracy"]) - 1, color="gray", linestyle="--", label="unfreeze")
    ax.legend(); ax.set_title("Skin Type – Training Accuracy")
    fig.savefig(os.path.join(output_dir, "training_curve.png"), dpi=120)
    print(f"Saved plot → {output_dir}/training_curve.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("SkinNova – Skin Type Model Training")
    print("=" * 60)

    train_gen, val_gen, test_gen = build_generators()
    print(f"\nClass indices: {train_gen.class_indices}")
    print(f"Train samples: {train_gen.n}  |  Val: {val_gen.n}  |  Test: {test_gen.n}\n")

    # Save class index mapping
    import json
    with open(os.path.join(OUTPUT_DIR, "class_indices.json"), "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)

    model, base = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # Phase 1 – Frozen base
    print("\n[Phase 1] Training classifier head (base frozen) …")
    h1 = model.fit(
        train_gen, epochs=EPOCHS_FROZEN, validation_data=val_gen,
        callbacks=get_callbacks("phase1"),
    )

    # Phase 2 – Unfreeze top 40 layers
    print("\n[Phase 2] Fine-tuning top 40 layers of EfficientNetB0 …")
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    h2 = model.fit(
        train_gen, epochs=EPOCHS_UNFREEZE, validation_data=val_gen,
        callbacks=get_callbacks("phase2"),
    )

    # Evaluate
    print("\n[Evaluate] Test set …")
    loss, acc = model.evaluate(test_gen)
    print(f"Test Loss: {loss:.4f}  |  Test Accuracy: {acc:.4f}")

    # Save final model
    model.save(os.path.join(OUTPUT_DIR, "model.h5"))
    print(f"\nModel saved → {OUTPUT_DIR}/model.h5")

    plot_history(h1, h2, OUTPUT_DIR)


if __name__ == "__main__":
    train()
