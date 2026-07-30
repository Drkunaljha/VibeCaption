"""
VibeCaption — CNN + LSTM Image Captioning Training Script (FINAL)
====================================================================
HOW TO USE THIS FILE:
  1. On Kaggle: New Notebook -> Add Input -> attach the Flickr8k dataset.
  2. Settings -> Accelerator -> GPU T4 x2. Session type: Save & Run All (or a
     fresh interactive session).
  3. Delete every other code cell in the notebook. Paste this ENTIRE file
     into a single fresh cell (or upload it and run it as one script).
     Do not mix this with any older code you tried before.
  4. Run it top to bottom. It prints a clear status line after every stage,
     so if something is wrong you'll see exactly which stage failed and why
     — training will not silently start with 0 samples.

Output saved to /kaggle/working/:
  - caption_model.keras
  - feature_extractor.keras
  - tokenizer.pkl
  - config.pkl
Download all four and put them in the local `models/` folder of the
Streamlit repo.
"""

import os
import re
import pickle
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

print("TensorFlow:", tf.__version__)
print("GPUs detected:", tf.config.list_physical_devices("GPU"))

EPOCHS = 20
BATCH_SIZE = 64
FEATURE_EXTRACTION_BATCH = 64
EMBEDDING_DIM = 256
LSTM_UNITS = 256
SEED = 42
OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)
AUTOTUNE = tf.data.AUTOTUNE


# ====================================================================
# STAGE 1 — AUTO-DETECT DATASET PATHS
# Kaggle's mount path for the same dataset can differ (e.g.
# /kaggle/input/flickr8k/... vs /kaggle/input/datasets/<owner>/flickr8k/...
# depending on how it was added). Instead of hardcoding a guess, we search
# for the actual files under /kaggle/input.
# ====================================================================
def find_dataset_paths(root="/kaggle/input/datasets/adityajn105/flickr8k"):
    captions_file = None
    images_dir = None

    for dirpath, dirnames, filenames in os.walk(root):
        if captions_file is None:
            for fname in filenames:
                if fname.lower() in ("captions.txt", "flickr8k.token.txt"):
                    captions_file = os.path.join(dirpath, fname)
        if images_dir is None:
            for dname in dirnames:
                if dname.lower() in ("images", "flicker8k_dataset", "flickr8k_dataset"):
                    images_dir = os.path.join(dirpath, dname)
        if captions_file and images_dir:
            break

    return captions_file, images_dir


print("\n[Stage 1] Searching for the Flickr8k dataset under /kaggle/input ...")
CAPTIONS_FILE, IMAGES_DIR = find_dataset_paths()

if CAPTIONS_FILE is None or IMAGES_DIR is None:
    print("Could not auto-detect the dataset. Everything found under /kaggle/input:")
    for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
        print(" ", dirpath, "->", dirnames, filenames[:5])
    raise FileNotFoundError(
        "Could not find captions.txt / Images folder automatically. "
        "Check the printed tree above, then set CAPTIONS_FILE and IMAGES_DIR manually."
    )

print(f"  Captions file: {CAPTIONS_FILE}")
print(f"  Images dir:    {IMAGES_DIR}")
n_images_on_disk = len([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(".jpg")])
print(f"  .jpg files found in Images dir: {n_images_on_disk}")
assert n_images_on_disk > 100, (
    f"Only found {n_images_on_disk} jpg files in {IMAGES_DIR} — this doesn't look "
    "like the full Flickr8k dataset. Check the path above before continuing."
)


# ====================================================================
# STAGE 2 — LOAD & CLEAN CAPTIONS
# ====================================================================
def load_captions(captions_path):
    mapping = {}
    with open(captions_path, "r") as f:
        first_line = f.readline()
        # if the first line isn't a header (rare token-file format), rewind
        if "," not in first_line and "\t" not in first_line:
            pass
        f.seek(0)
        next(f)  # skip header row: image,caption
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.split(",", 1)
            if len(tokens) < 2:
                continue
            image_id, caption = tokens[0], tokens[1]
            image_id = image_id.split(".")[0]
            mapping.setdefault(image_id, []).append(caption)
    return mapping


def clean_captions(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i].lower()
            caption = re.sub(r"[^a-z\s]", "", caption)
            caption = re.sub(r"\s+", " ", caption).strip()
            words = [w for w in caption.split() if len(w) > 1]
            captions[i] = "startseq " + " ".join(words) + " endseq"
    return mapping


print("\n[Stage 2] Loading and cleaning captions...")
captions_mapping = load_captions(CAPTIONS_FILE)
captions_mapping = clean_captions(captions_mapping)
print(f"  Captions loaded for {len(captions_mapping)} image IDs")

# Keep only images that actually exist on disk as .jpg
captions_mapping = {
    img_id: caps for img_id, caps in captions_mapping.items()
    if os.path.exists(os.path.join(IMAGES_DIR, img_id + ".jpg"))
}
print(f"  Images with both captions AND a matching .jpg file: {len(captions_mapping)}")
assert len(captions_mapping) > 100, (
    "Fewer than 100 images matched between captions.txt and the Images folder. "
    "The image_id format in captions.txt may not match the .jpg filenames — "
    "print a few captions_mapping keys and os.listdir(IMAGES_DIR)[:5] to compare."
)


# ====================================================================
# STAGE 3 — TOKENIZER
# ====================================================================
print("\n[Stage 3] Building tokenizer...")
all_captions = [c for caps in captions_mapping.values() for c in caps]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(c.split()) for c in all_captions)
print(f"  Vocabulary size: {vocab_size}")
print(f"  Max caption length: {max_length}")


# ====================================================================
# STAGE 4 — CNN ENCODER: BATCH FEATURE EXTRACTION
# ====================================================================
def build_feature_extractor():
    base_model = InceptionV3(weights="imagenet")
    return Model(inputs=base_model.input, outputs=base_model.layers[-2].output)


def extract_all_features(image_ids, images_dir, cnn_model, target_size=(299, 299), batch_size=64):
    features = {}
    for start in tqdm(range(0, len(image_ids), batch_size), desc="Extracting CNN features"):
        batch_ids = image_ids[start:start + batch_size]
        batch_images, valid_ids = [], []
        for image_id in batch_ids:
            img_path = os.path.join(images_dir, image_id + ".jpg")
            img = load_img(img_path, target_size=target_size)
            batch_images.append(img_to_array(img))
            valid_ids.append(image_id)
        batch_array = preprocess_input(np.array(batch_images))
        batch_features = cnn_model.predict(batch_array, verbose=0)
        for image_id, feature in zip(valid_ids, batch_features):
            features[image_id] = feature
    return features


print("\n[Stage 4] Building InceptionV3 encoder and extracting features...")
cnn_encoder = build_feature_extractor()
image_ids = list(captions_mapping.keys())
image_features = extract_all_features(image_ids, IMAGES_DIR, cnn_encoder, batch_size=FEATURE_EXTRACTION_BATCH)
print(f"  Features extracted for {len(image_features)} images")
assert len(image_features) > 100, "Feature extraction produced almost no features — check Stage 4 errors above."


# ====================================================================
# STAGE 5 — TRAIN / VALIDATION SPLIT
# ====================================================================
print("\n[Stage 5] Splitting train/validation sets...")
rng = np.random.default_rng(SEED)
shuffled_ids = list(image_features.keys())
rng.shuffle(shuffled_ids)
split_idx = int(len(shuffled_ids) * 0.9)
train_ids = shuffled_ids[:split_idx]
val_ids = shuffled_ids[split_idx:]
print(f"  Train images: {len(train_ids)}  |  Validation images: {len(val_ids)}")


# ====================================================================
# STAGE 6 — BUILD (image_feature, partial_sequence) -> next_word PAIRS
# ====================================================================
def build_sequence_pairs(image_ids, mapping, features, tokenizer, max_length):
    X1, X2, y = [], [], []
    for image_id in image_ids:
        for caption in mapping[image_id]:
            seq = tokenizer.texts_to_sequences([caption])[0]
            for i in range(1, len(seq)):
                in_seq, out_word = seq[:i], seq[i]
                # padding="post" -> right-padding (required by cuDNN LSTM
                # when mask_zero=True; left-padding crashes on GPU).
                in_seq = pad_sequences([in_seq], maxlen=max_length, padding="post")[0]
                X1.append(features[image_id])
                X2.append(in_seq)
                y.append(out_word)
    return np.array(X1, dtype="float32"), np.array(X2, dtype="int32"), np.array(y, dtype="int32")


print("\n[Stage 6] Building training pairs...")
X1_train, X2_train, y_train = build_sequence_pairs(train_ids, captions_mapping, image_features, tokenizer, max_length)
X1_val, X2_val, y_val = build_sequence_pairs(val_ids, captions_mapping, image_features, tokenizer, max_length)
print(f"  Train pairs: {len(y_train)}  |  Validation pairs: {len(y_val)}")
assert len(y_train) > 0 and len(y_val) > 0, "No training pairs were built — check earlier stages."


# ====================================================================
# STAGE 7 — tf.data PIPELINE (keeps the GPU continuously fed)
# ====================================================================
def make_dataset(X1, X2, y, vocab_size, batch_size, shuffle):
    def gen():
        for i in range(len(y)):
            yield (X1[i], X2[i]), y[i]

    output_signature = (
        (
            tf.TensorSpec(shape=(2048,), dtype=tf.float32),
            tf.TensorSpec(shape=(max_length,), dtype=tf.int32),
        ),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    ds = ds.apply(tf.data.experimental.assert_cardinality(len(y)))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(10000, len(y)), seed=SEED)

    def one_hot(inputs, label):
        return inputs, tf.one_hot(label, depth=vocab_size)

    ds = ds.map(one_hot, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds


print("\n[Stage 7] Building tf.data pipelines...")
train_ds = make_dataset(X1_train, X2_train, y_train, vocab_size, BATCH_SIZE, shuffle=True)
val_ds = make_dataset(X1_val, X2_val, y_val, vocab_size, BATCH_SIZE, shuffle=False)


# ====================================================================
# STAGE 8 — LSTM DECODER MODEL
# ====================================================================
def build_caption_model(vocab_size, max_length, feature_dim=2048):
    inputs1 = Input(shape=(feature_dim,), name="image_features")
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(EMBEDDING_DIM, activation="relu")(fe1)

    inputs2 = Input(shape=(max_length,), name="caption_sequence")
    se1 = Embedding(vocab_size, EMBEDDING_DIM, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(LSTM_UNITS)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(LSTM_UNITS, activation="relu")(decoder1)
    outputs = Dense(vocab_size, activation="softmax")(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


print("\n[Stage 8] Building the LSTM decoder model...")
caption_model = build_caption_model(vocab_size, max_length)
caption_model.summary()


# ====================================================================
# STAGE 9 — TRAIN
# ====================================================================
checkpoint_path = os.path.join(OUTPUT_DIR, "caption_model.keras")
callbacks = [
    ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
]

print("\n[Stage 9] Training...")
history = caption_model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks,
    shuffle=False,  # dataset is already shuffled inside make_dataset()
    verbose=1,
)


# ====================================================================
# STAGE 10 — SAVE EVERYTHING NEEDED FOR INFERENCE
# ====================================================================
print("\n[Stage 10] Saving feature extractor, tokenizer and config...")
cnn_encoder.save(os.path.join(OUTPUT_DIR, "feature_extractor.keras"))

with open(os.path.join(OUTPUT_DIR, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)

with open(os.path.join(OUTPUT_DIR, "config.pkl"), "wb") as f:
    pickle.dump({"max_length": max_length, "vocab_size": vocab_size}, f)

print("\n✅ Done. Download these 4 files from /kaggle/working/:")
print("  - caption_model.keras")
print("  - feature_extractor.keras")
print("  - tokenizer.pkl")
print("  - config.pkl")