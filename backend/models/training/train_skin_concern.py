"""
SkinNova AI – Training Script 3: Skin Concern Detector (Multi-Label)
Dataset: Skin v2  (acne / blackheads / dark_spots / pores / wrinkles)
Architecture: EfficientNetB0 with sigmoid output (multi-label)

Run:
    python models/training/train_skin_concern.py
"""

import os, json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "../../data/Skin v2")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../skin_concern_model")
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EPOCHS_FROZEN   = 10
EPOCHS_UNFREEZE = 20
# Dataset 1 is structured as single-label per folder, so we treat it as
# multi-class but output sigmoid to enable thresholding per concern
NUM_CLASSES = 5
CLASS_NAMES = ["acne", "blackheades", "dark spots", "pores", "wrinkles"]
OUTPUT_NAMES = ["acne", "blackheads", "dark_spots", "pores", "wrinkles"]  # clean names
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Generators ────────────────────────────────────────────────────────────────
def build_generators():
    preprocess = tf.keras.applications.efficientnet.preprocess_input

    train_aug = ImageDataGenerator(
        preprocessing_function=preprocess,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.7, 1.3],
        shear_range=0.1,
        channel_shift_range=20.0,
    )
    val_aug = ImageDataGenerator(preprocessing_function=preprocess)

    # Dataset 1 has only one split folder – we create val from train
    full_gen = train_aug.flow_from_directory(
        DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=True,
    )

    # 80/20 split via subset (ImageDataGenerator supports validation_split)
    train_aug2 = ImageDataGenerator(
        preprocessing_function=preprocess,
        rotation_range=20, width_shift_range=0.15, height_shift_range=0.15,
        horizontal_flip=True, zoom_range=0.2, brightness_range=[0.7, 1.3],
        validation_split=0.2,
    )
    train_gen = train_aug2.flow_from_directory(
        DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=True, subset="training",
    )
    val_gen = ImageDataGenerator(
        preprocessing_function=preprocess, validation_split=0.2
    ).flow_from_directory(
        DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=False, subset="validation",
    )

    print(f"Dataset class mapping: {train_gen.class_indices}")
    return train_gen, val_gen


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
    # softmax for single-label (concern per image) – post-processing handles multi-label
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return Model(inputs, outputs), base


# ── LR Schedule ──────────────────────────────────────────────────────────────
def cosine_decay_with_warmup(epoch, total_epochs=EPOCHS_UNFREEZE, warmup=3, lr_max=1e-5):
    if epoch < warmup:
        return lr_max * (epoch + 1) / warmup
    progress = (epoch - warmup) / (total_epochs - warmup)
    return lr_max * 0.5 * (1 + np.cos(np.pi * progress))


# ── Main ──────────────────────────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("SkinNova – Skin Concern Model Training")
    print("=" * 60)

    train_gen, val_gen = build_generators()

    # Save class→output name mapping
    raw_to_clean = dict(zip(CLASS_NAMES, OUTPUT_NAMES))
    with open(os.path.join(OUTPUT_DIR, "class_indices.json"), "w") as f:
        json.dump({
            "raw_indices": train_gen.class_indices,
            "output_names": OUTPUT_NAMES,
            "raw_to_clean": raw_to_clean,
        }, f, indent=2)

    model, base = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks_p1 = [
        ModelCheckpoint(os.path.join(OUTPUT_DIR, "best_p1.h5"),
                        save_best_only=True, monitor="val_accuracy", mode="max"),
        EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    ]

    print("\n[Phase 1] Training head …")
    h1 = model.fit(
        train_gen, epochs=EPOCHS_FROZEN, validation_data=val_gen,
        callbacks=callbacks_p1,
    )

    print("\n[Phase 2] Fine-tuning top layers …")
    base.trainable = True
    for layer in base.layers[:-60]:
        layer.trainable = False

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda e: cosine_decay_with_warmup(e, total_epochs=EPOCHS_UNFREEZE)
    )
    callbacks_p2 = [
        ModelCheckpoint(os.path.join(OUTPUT_DIR, "best_p2.h5"),
                        save_best_only=True, monitor="val_accuracy", mode="max"),
        EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        lr_schedule,
    ]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    h2 = model.fit(
        train_gen, epochs=EPOCHS_UNFREEZE, validation_data=val_gen,
        callbacks=callbacks_p2,
    )

    val_loss, val_acc = model.evaluate(val_gen)
    print(f"\nFinal Val Loss: {val_loss:.4f}  |  Val Acc: {val_acc:.4f}")

    model.save(os.path.join(OUTPUT_DIR, "model.h5"))
    print(f"Model saved → {OUTPUT_DIR}/model.h5")

    # Training curve
    acc  = h1.history["accuracy"]  + h2.history["accuracy"]
    vacc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    fig, ax = plt.subplots()
    ax.plot(acc, label="train"); ax.plot(vacc, label="val")
    ax.axvline(len(h1.history["accuracy"]) - 1, color="gray", linestyle="--")
    ax.set_title("Skin Concern – Accuracy"); ax.legend()
    fig.savefig(os.path.join(OUTPUT_DIR, "training_curve.png"), dpi=120)
    print("Training curve saved.")


if __name__ == "__main__":
    train()
