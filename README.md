# Email Spam Detection using Machine Learning

A complete, beginner-friendly Python project that trains a spam/ham text classifier and provides an interactive Streamlit web app. It uses TF-IDF features with a Multinomial Naive Bayes classifier and the Apache SpamAssassin Public Corpus.

## Features

- Downloads and parses public email examples for training.
- Splits data into training and test sets and reports evaluation metrics.
- Saves the trained model locally (model and data files are not committed).
- Lets you paste a subject and message to get a spam/ham prediction and spam probability.
- Includes a configurable probability threshold and displays test accuracy.

## Requirements

- Python 3.10 or newer
- Internet connection for the first training run, to download the dataset

## Setup and run

Open a terminal in the project directory, then run:

```bash
# Optional but recommended: create and activate an isolated environment
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# Install dependencies
python -m pip install -r requirements.txt

# Train once; downloads the SpamAssassin corpus and saves the model
python train.py

# Start the web app
streamlit run app.py
```

Open the local URL printed by Streamlit (usually http://localhost:8501). You can also start the app before training and press **Train model** in the browser; it will download the corpus and train the model for you.

> Windows: run only the activation command that matches your shell. If PowerShell blocks activation, use Command Prompt's `.venv\Scripts\activate.bat` or continue without activating the environment.

## Use the app

1. Enter an email subject, message, or both.
2. Adjust the spam threshold if desired. Higher thresholds are more conservative about labeling an email as spam.
3. Click **Check email** to see the prediction and probability.

## Project structure

```text
.
├── app.py                 # Streamlit interface and prediction
├── train.py               # Train/evaluate and save the model
├── src/
│   └── data.py            # Corpus download, parsing, and loading
├── data/raw/              # Downloaded corpus (generated; ignored by Git)
├── models/                # Trained model and metrics (generated; ignored by Git)
├── requirements.txt
└── README.md
```

## Dataset and limitations

The training script downloads the `easy_ham` and `spam` archives from the [Apache SpamAssassin Public Corpus](https://spamassassin.apache.org/old/publiccorpus/). The data remains on your computer under `data/raw/`. Review the corpus source and its terms before redistributing any dataset contents.

This is an educational example, not a production email security product. It is trained on a small, historical dataset, so its predictions may not generalize to modern messages. Never treat the prediction as a guarantee or use it as the sole basis for security decisions. Do not paste confidential messages into a publicly hosted demo.

## Troubleshooting

- **Download fails:** Check your internet connection and run `python train.py` again.
- **No email messages found:** Remove the incomplete `data/raw/` directory and rerun training so the corpus downloads again.
- **Command not found:** Confirm the virtual environment is activated and dependencies installed; use `python -m pip` and `python -m streamlit` if needed.

## License

This project is released under the MIT License. Dataset contents are provided separately by the corpus maintainers.
