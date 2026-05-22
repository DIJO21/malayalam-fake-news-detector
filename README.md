# Malayalam Fake News Detection System with Live RAG Fact-Checking

This project implements a state-of-the-art **Malayalam Fake News Detection System** utilizing a fine-tuned Google **MuRIL** (Multilingual Representations for Indian Languages) Transformer model, integrated with a **Retrieval-Augmented Generation (RAG)** fact-checking validation pipeline. 

It features a production-ready Flask API and a premium, responsive glassmorphic user dashboard.

---

## Features
- **MuRIL Transformer Model**: Upgraded sequence classifier specifically optimized for Indian languages, resolving majority-class collapse issues observed in traditional RNN architectures.
- **RAG Live Fact-Checking**: Corroborates query text in real-time against live news feeds (via NewsData.io) using semantic search (Sentence Transformers) and generates credibility reasoning (OpenAI GPT-3.5 or an offline fallback context summary).
- **Offline Resilience & Query Fallbacks**:
  - Automatically loads tokenizer/weights locally for seamless offline/cached startup.
  - Uses a recursive query-width fallback (5, 3, 2, 1 words) to ensure search results are fetched even for highly specific Malayalam sentences.
- **Visual Glassmorphic Dashboard**: A professional split-screen UI featuring neon indicators, sample click cards, character counters, layout micro-animations, and detailed evidence reports.
- **Evaluation & Diagnostics**: Automated early-stopped training (patience = 3) saving learning curves and confusion matrices to the `results/` folder.

---

## Project Structure
```text
project_root/
├── data/                  # Datasets
│   ├── raw/               # Contains the raw CSV (malayalam_dataset_fast.csv)
│   └── processed/         # Preprocessed cleaned data
├── saved_models/          # Cached tokenizers and fine-tuned checkpoints
│   └── mbert_model/       # Best saved MuRIL checkpoint
├── preprocessing/         # Malayalam text cleaning and unicode normalization
├── retrieval/             # RAG Engine (Semantic search and LLM fact corroborator)
├── api/                   # Flask RESTful API & Server
├── frontend/              # Glassmorphic templates, CSS, and interactive JS
├── utils/                 # Visualizations and metrics helper scripts
├── results/               # Training curves and confusion matrix plots
├── train.py               # Main model fine-tuning and evaluation script
├── config.py              # Centralized hyperparameters & environment config
├── requirements.txt       # Python dependencies
└── README.md              # Documentation
```

---

## Setup Instructions

### 1. Create a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Credentials (.env)
Create a `.env` file in the project root:
```env
NEWS_API_KEY=your_newsdata_io_key
OPENAI_API_KEY=your_openai_api_key
```

---

## Execution Guide

### Phase 1: Train the MuRIL Classifier
To run the transformer fine-tuning loop (with stratified split, weighted cross-entropy, and early stopping):
```bash
python train.py
```
This will:
- Tokenize and clean the Malayalam dataset.
- Train the MuRIL model (saves the best-performing checkpoint on validation loss to `saved_models/mbert_model/`).
- Save confusion matrix and training curves to `results/`.

### Phase 2: Start the Web App
Start the Flask API to run the server:
```bash
python api/app.py
```
Open your browser and navigate to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Deployment Options
- **Local/Production**: Run the server with `gunicorn api.app:app` behind a reverse proxy (e.g. Nginx).
- **Cloud (Render/HuggingFace Spaces)**: Push the repository to GitHub (the `.gitignore` is pre-configured to exclude large 700MB weight folders and local credentials), and link it to Render or HF Spaces. Weights can be downloaded on initialization or kept in LFS.
