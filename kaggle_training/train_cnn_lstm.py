"""
Kaggle Training Script: CNN (InceptionV3) + LSTM for Image Captioning
Dataset: Flickr8k
"""
import os
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

# ==========================================
# 1. PATHS (Kaggle के हिसाब से एडजस्ट करें)
# ==========================================
BASE_DIR = '/kaggle/input/flickr8k' # अगर Kaggle डेटासेट का नाम कुछ और हो तो इसे बदल लें
WORKING_DIR = '/kaggle/working'
IMAGES_DIR = os.path.join(BASE_DIR, 'Images')
CAPTIONS_FILE = os.path.join(BASE_DIR, 'captions.txt')

# ==========================================
# 2. FEATURE EXTRACTION (CNN - InceptionV3)
# ==========================================
print("Loading InceptionV3 Model...")
inception = InceptionV3(weights='imagenet')
feature_extractor = Model(inputs=inception.input, outputs=inception.layers[-2].output)

# Save feature extractor for inference in app
feature_extractor.save(os.path.join(WORKING_DIR, 'feature_extractor.keras'))

def extract_features(directory):
    features = {}
    print("Extracting features from all images (this may take a few minutes)...")
    for img_name in tqdm(os.listdir(directory)):
        img_path = os.path.join(directory, img_name)
        try:
            image = Image.open(img_path).convert("RGB").resize((299, 299))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            feature = feature_extractor.predict(image, verbose=0)
            features[img_name] = feature[0]
        except Exception as e:
            continue
    return features

# आप चाहें तो इसे pickle कर सकते हैं ताकि दोबारा रन करने पर टाइम बचे
features = extract_features(IMAGES_DIR)

# ==========================================
# 3. TEXT PREPROCESSING
# ==========================================
print("Processing Captions...")
with open(CAPTIONS_FILE, 'r') as f:
    next(f) # Header skip
    captions_doc = f.read()

mapping = {}
for line in captions_doc.split('\n'):
    tokens = line.split(',')
    if len(tokens) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    caption = " ".join(caption)
    
    if image_id not in mapping:
        mapping[image_id] = []
    
    # Preprocessing caption + adding startseq/endseq
    caption = caption.lower().replace('[^a-z]', '')
    caption = f"startseq {caption} endseq"
    mapping[image_id].append(caption)

# Create a flat list of all captions
all_captions = []
for key in mapping:
    for cap in mapping[key]:
        all_captions.append(cap)

# Tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(cap.split()) for cap in all_captions)

# Save tokenizer and config
with open(os.path.join(WORKING_DIR, 'tokenizer.pkl'), 'wb') as f:
    pickle.dump(tokenizer, f)

config = {"max_length": max_length, "vocab_size": vocab_size}
with open(os.path.join(WORKING_DIR, 'config.pkl'), 'wb') as f:
    pickle.dump(config, f)

# ==========================================
# 4. DATA GENERATOR (To prevent RAM crash)
# ==========================================
def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    X1, X2, y = list(), list(), list()
    n = 0
    while True:
        for key in data_keys:
            if key not in features: continue
            n += 1
            captions = mapping[key]
            for cap in captions:
                seq = tokenizer.texts_to_sequences([cap])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    # Right padding to match model_utils logic
                    in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post')[0]
                    out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=vocab_size)[0]
                    
                    X1.append(features[key])
                    X2.append(in_seq)
                    y.append(out_seq)
            
            if n == batch_size:
                yield (np.array(X1), np.array(X2)), np.array(y)
                X1, X2, y = list(), list(), list()
                n = 0

# ==========================================
# 5. MODEL ARCHITECTURE (CNN + LSTM)
# ==========================================
print("Building Captioning Model...")
# Image feature extractor model
inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

# Sequence model
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

# Decoder model
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

caption_model = Model(inputs=[inputs1, inputs2], outputs=outputs)
caption_model.compile(loss='categorical_crossentropy', optimizer='adam')

# ==========================================
# 6. TRAINING
# ==========================================
train_keys = list(mapping.keys()) # For a real project, split this into 80/20 train/val
epochs = 20
batch_size = 32
steps = len(train_keys) // batch_size

print("Starting Training...")
for i in range(epochs):
    generator = data_generator(train_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size)
    caption_model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)

# Save the final trained model
caption_model.save(os.path.join(WORKING_DIR, 'caption_model.keras'))
print("Training Complete! Download the 4 files from /kaggle/working/")