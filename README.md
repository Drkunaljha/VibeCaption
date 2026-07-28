# VibeCaption 🎨📸

An image captioning web app — upload a photo, get a natural-language description back.

Built as a final-year Data Science project, combining a from-scratch **CNN + LSTM**
model (trained on Flickr8k) with a pretrained **BLIP** transformer for the live demo.

---

## How it works

1. **CNN encoder** (InceptionV3) turns an image into a 2048-dimensional feature vector.
2. **LSTM decoder** takes that feature vector and generates a caption one word at a time.
3. The **BLIP** model (used in the deployed app) does the same job end-to-end using a
   pretrained vision-language transformer — no training required, and it produces more
   fluent captions for the live demo.

## Project structure

```
VibeCaption/
├── app.py                        # Streamlit UI — the deployed app
├── model_utils.py                # BLIP + CNN/LSTM inference logic
├── requirements.txt              # Python dependencies
├── .gitignore
├── README.md
├── kaggle_training/
│   └── train_cnn_lstm.py         # Full training script — run this on Kaggle GPU
└── models/                       # Trained model files go here (not pushed to GitHub)
    ├── caption_model.keras
    ├── feature_extractor.keras
    ├── tokenizer.pkl
    └── config.pkl
```

## Training the CNN + LSTM model (on Kaggle)

1. Create a new Kaggle notebook, attach the **Flickr8k** dataset, and turn on
   **GPU** acceleration (Settings → Accelerator → GPU T4 x2).
2. Upload `kaggle_training/train_cnn_lstm.py` and run it, or paste its contents
   into notebook cells.
3. After training finishes, download these 4 files from `/kaggle/working/`:
   - `caption_model.keras`
   - `feature_extractor.keras`
   - `tokenizer.pkl`
   - `config.pkl`
4. Place them inside the local `models/` folder.

> Note: these model files are large, so they're excluded from Git via `.gitignore`.
> The deployed Streamlit app uses BLIP directly and does not require these files —
> they're only needed if you want to demo the from-scratch CNN + LSTM pipeline too.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploying on Streamlit Cloud

1. Push this repository to GitHub (model files excluded — see `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo, and
   set `app.py` as the main file.
3. Streamlit Cloud installs everything from `requirements.txt` automatically.

## Tech stack

- Python, TensorFlow/Keras (CNN + LSTM training)
- PyTorch, Hugging Face Transformers (BLIP inference)
- Streamlit (web app + deployment)
- Dataset: Flickr8k
