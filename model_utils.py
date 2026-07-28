"""
VibeCaption — model_utils.py
==============================
All "backend" logic lives here. app.py only calls these functions —
it never talks to the models directly. This keeps the UI file clean
and makes the captioning logic reusable/testable on its own.

Two captioning paths are provided:

1. BLIP  (generate_caption_blip)
   Pretrained transformer, used for the live Streamlit demo.
   No training needed — works out of the box.

2. CNN + LSTM  (generate_caption_cnn_lstm)
   Your own model trained from scratch on Flickr8k in
   kaggle_training/train_cnn_lstm.py. Used to show the
   from-scratch pipeline in the project report / a secondary tab.
"""

import pickle
import numpy as np
import streamlit as st
from PIL import Image

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


# ------------------------------------------------------------------
# BLIP — pretrained, used for the main live demo
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_blip():
    """Loads BLIP processor + model once, cached across reruns."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device


def generate_caption_blip(image: Image.Image, processor, model, device, style_prefix: str = ""):
    """
    Generates a caption for a PIL image using BLIP.
    style_prefix (optional): conditions the caption, e.g. "a photo of"
    lets you nudge tone without retraining anything.
    """
    if style_prefix:
        inputs = processor(image, style_prefix, return_tensors="pt").to(device)
    else:
        inputs = processor(image, return_tensors="pt").to(device)

    output_ids = model.generate(**inputs, max_new_tokens=40)
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip().capitalize()


# ------------------------------------------------------------------
# CNN + LSTM — your own trained model (optional secondary path)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_cnn_lstm_assets(model_path, feature_extractor_path, tokenizer_path, config_path):
    """Loads everything the from-scratch model needs for inference."""
    caption_model = load_model(model_path)
    feature_extractor = load_model(feature_extractor_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    with open(config_path, "rb") as f:
        config = pickle.load(f)
    return caption_model, feature_extractor, tokenizer, config


def _extract_feature(image: Image.Image, feature_extractor, target_size=(299, 299)):
    image = image.convert("RGB").resize(target_size)
    array = img_to_array(image)
    array = np.expand_dims(array, axis=0)
    array = preprocess_input(array)
    return feature_extractor.predict(array, verbose=0)


def generate_caption_cnn_lstm(image: Image.Image, caption_model, feature_extractor, tokenizer, config):
    """
    Greedy word-by-word decoding: starts at 'startseq', predicts one
    word at a time using the image feature + words generated so far,
    stops at 'endseq' or when max_length is reached.
    """
    max_length = config["max_length"]
    feature = _extract_feature(image, feature_extractor)

    index_to_word = {idx: word for word, idx in tokenizer.word_index.items()}

    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # Must match training: padding="post" (right-padding), since the
        # model was trained with right-padded sequences (cuDNN LSTM requirement).
        sequence = pad_sequences([sequence], maxlen=max_length, padding="post")
        prediction = caption_model.predict([feature, sequence], verbose=0)
        predicted_idx = np.argmax(prediction)
        predicted_word = index_to_word.get(predicted_idx)

        if predicted_word is None or predicted_word == "endseq":
            break
        in_text += " " + predicted_word

    final_caption = in_text.replace("startseq", "").strip()
    return final_caption.capitalize()
