<a href="https://deepwiki.com/Anilkumar20102005/SentimetAnalysis"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>

# 🛍️ Product Sentiment Analysis — Full Stack AI Platform

> An end-to-end, production-ready AI application for real-time product review sentiment classification, powered by a frozen **TF-IDF + Logistic Regression** pipeline trained on the **Amazon Review Polarity** dataset, an asynchronous **FastAPI** backend, and a modern, glassmorphic **React** frontend.

---

## 📑 Table of Contents
1. [Overview & Highlights](#-overview--highlights)
2. [System Architecture](#-system-architecture)
3. [Repository Structure](#-repository-structure)
4. [Dataset Details](#-dataset-details)
5. [Machine Learning Pipeline](#-machine-learning-pipeline)
   - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
   - [Text Preprocessing](#text-preprocessing)
   - [Feature Engineering & Vectorization](#feature-engineering--vectorization)
   - [Model Architecture & Training](#model-architecture--training)
   - [Evaluation & Performance Metrics](#evaluation--performance-metrics)
   - [Model Freezing & Serialization](#model-freezing--serialization)
6. [Jupyter Notebook Walkthrough](#-jupyter-notebook-walkthrough)
7. [FastAPI Backend](#-fastapi-backend)
   - [Design & Lifespan Management](#design--lifespan-management)
   - [API Reference & Examples](#api-reference--examples)
8. [React Frontend UI](#-react-frontend-ui)
   - [Design System & Aesthetics](#design-system--aesthetics)
   - [Key Interactive Features](#key-interactive-features)
9. [Installation & Getting Started](#-installation--getting-started)
   - [Prerequisites](#prerequisites)
   - [Dependency Installation](#dependency-installation)
10. [Running the Application](#-running-the-application)
    - [Method 1: One-Command Dev Mode (Recommended)](#method-1-one-command-dev-mode-recommended)
    - [Method 2: Windows Batch Launcher](#method-2-windows-batch-launcher)
    - [Method 3: Unified Single-Port Server](#method-3-unified-single-port-server)
    - [Method 4: Independent Server Launch](#method-4-independent-server-launch)
11. [Testing & Verification](#-testing--verification)
12. [Troubleshooting & FAQs](#-troubleshooting--faqs)

---

## 🌟 Overview & Highlights

This project demonstrates a complete Machine Learning and Full-Stack Engineering lifecycle:
- **Data Science & NLP**: Full Exploratory Data Analysis, statistical linguistic inspection, text normalisation, and model training in a Jupyter Notebook.
- **High-Performance ML**: Optimized `TF-IDF + Logistic Regression` achieving **87.00% Test Accuracy** and **0.9448 ROC-AUC** with sub-millisecond (`< 1 ms`) inference latency.
- **Model Freezing**: Decoupled, portable model artifact (`sentiment_pipeline.joblib`) with embedded vocabulary and metadata JSON.
- **Modern REST API**: Asynchronous **FastAPI** backend with automated Pydantic validation, dual-route routing (`/` and `/api/`), and interactive Swagger/OpenAPI docs.
- **Glassmorphic React Frontend**: Custom CSS design system (no generic frameworks), real-time SVG circular confidence meters, live probability distribution bars, quick-test sample chips, and local analysis history.
- **Developer Experience**: One-command concurrent launcher (`start_dev.py` and `start.bat`) handling port management, log multiplexing, and graceful multi-process shutdown.

---

## 🏛️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        User Web Browser                                │
│                     http://localhost:5173                              │
└──────────────────┬─────────────────────────────────▲───────────────────┘
                   │                                 │
         HTTP POST /api/predict              JSON Response:
       {"text": "...", "title": "..."}       {sentiment, confidence, ...}
                   │                                 │
                   ▼                                 │
┌────────────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend Server                          │
│                         http://localhost:8000                          │
│                                                                        │
│  ┌───────────────────────┐             ┌────────────────────────────┐  │
│  │   Input Validation    │ ──────────► │     Text Preprocessing     │  │
│  │   (Pydantic Models)   │             │ (Clean, Stopwords, Lemmat) │  │
│  └───────────────────────┘             └─────────────┬──────────────┘  │
│                                                      │                 │
│                                                      ▼                 │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │               Frozen Pipeline (sentiment_pipeline.joblib)        │  │
│  │                                                                  │  │
│  │   [TF-IDF Vectorizer (80k features)] ──► [Logistic Regression]   │  │
│  └───────────────────────────────────────────────────┬──────────────┘  │
│                                                      │                 │
│  ┌──────────────────────────────────────────────┐    │                 │
│  │  Inference Engine (Probabilities & Metrics)  │ ◄──┘                 │
│  └──────────────────────────────────────────────┘                      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

```
sentiment-analysis/
│
├── backend/
│   ├── models/
│   │   ├── sentiment_pipeline.joblib    # Serialized scikit-learn pipeline (TF-IDF + LR)
│   │   └── model_metadata.json          # Model metrics, hyperparams, and training config
│   ├── main.py                          # FastAPI application, route handlers, and static mount
│   └── requirements.txt                 # Backend Python package requirements
│
├── frontend/
│   ├── dist/                            # Pre-compiled production React distribution
│   ├── src/
│   │   ├── App.jsx                      # Core interactive React application
│   │   ├── index.css                    # Premium dark-mode glassmorphic CSS design system
│   │   └── main.jsx                     # React DOM entrypoint
│   ├── index.html                       # HTML5 template with Inter Google Font
│   ├── package.json                     # Frontend dependencies & scripts
│   └── vite.config.js                   # Vite configuration with /api reverse proxy
│
├── data/
│   ├── polarity_sample.parquet          # Cached stratified sample (30,000 Amazon reviews)
│   ├── eda_class_distribution.png       # Generated plot: Class counts & proportions
│   ├── eda_text_length.png              # Generated plot: Title & content word distributions
│   ├── eda_top_keywords.png             # Generated plot: Top positive & negative keywords
│   ├── eval_confusion_roc.png           # Generated plot: Confusion matrix & ROC curve
│   └── eval_top_features.png            # Generated plot: Top predictive feature coefficients
│
├── notebook/
│   └── sentiment_analysis.ipynb         # Complete 9-stage Jupyter Notebook (EDA -> Export)
│
├── train.py                             # Standalone model training & evaluation script
├── start_dev.py                         # Concurrent launcher (FastAPI + Vite HMR)
├── start.bat                            # Windows double-click launcher
├── run.py                               # Single-command production launcher
└── README.md                            # Comprehensive documentation
```

---

## 📊 Dataset Details

The model is trained on the **Amazon Review Polarity** dataset (mirrored on Hugging Face as [CyranoB/polarity](https://huggingface.co/datasets/CyranoB/polarity) and [fancyzhx/amazon_polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity)):

- **Origin**: Stanford Network Analysis Project (SNAP) Amazon Reviews dataset spanning 18 years and over 34 million product reviews.
- **Classification Task**: Binary Sentiment Classification:
  - `0` / **Negative**: 1-star and 2-star reviews.
  - `1` / **Positive**: 4-star and 5-star reviews.
  *(3-star ambiguous reviews were excluded at source for clean polarity separation).*
- **Fields Utilized**:
  - `title`: Short summary / header of the customer review.
  - `content`: Detailed customer review narrative.
  - `label`: Integer class label (`0` or `1`).
- **Caching Mechanism**:
  A stratified subset of **30,000 reviews** (25,000 train, 5,000 test) is streamed and cached to `data/polarity_sample.parquet` (8.38 MB). Once cached, subsequent training runs and notebook executions operate completely offline in seconds.

---

## 🧠 Machine Learning Pipeline

### Exploratory Data Analysis (EDA)
The dataset undergoes comprehensive exploratory inspection:
1. **Class Balance**: Verification of exact 50/50 balance between positive and negative reviews to ensure models do not develop majority-class bias.
2. **Text Length Distributions**: Analysis of word counts across title and review bodies. Negative reviews exhibit a higher median word count (customers write longer explanations when complaining).
3. **Keyword Analysis**: Identification of frequent unigrams across polarities, guiding custom stopword pruning (`br`, `would`, `one`, `get`, etc.).

Visual artifacts generated:
- `data/eda_class_distribution.png`
- `data/eda_text_length.png`
- `data/eda_top_keywords.png`

### Text Preprocessing
The exact same deterministic preprocessing pipeline is applied during **both training and inference**:
```python
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)     # Remove HTML formatting
    text = re.sub(r'http\S+', '', text)      # Strip URLs
    text = re.sub(r'[^a-z\s]', ' ', text)    # Keep only alphabet characters
    tokens = [
        lemmatizer.lemmatize(w) 
        for w in text.split()
        if w not in STOP_WORDS and len(w) > 2
    ]
    return ' '.join(tokens)
```
- **Stopwords**: Standard NLTK English stopwords augmented with common e-commerce noise terms.
- **Lemmatization**: NLTK WordNet lemmatizer reduces inflected words (`loved`, `loving` -> `love`) while preserving semantic roots.

### Feature Engineering & Vectorization
We use Scikit-Learn's `TfidfVectorizer`:
- **N-gram Range**: `(1, 2)` (captures both individual words and bigram phrases like *"waste money"*, *"highly recommend"*).
- **Sublinear TF Scaling**: `sublinear_tf=True` applies $1 + \log(\text{tf})$ to prevent repetitive words from dominating weights.
- **Pruning**: `min_df=3` eliminates typos and rare noise; `max_df=0.90` strips corpus-wide ubiquitous words.
- **Vocabulary Size**: Up to **80,000** features.

### Model Architecture & Training
A `LogisticRegression` classifier wrapped in a Scikit-Learn `Pipeline`:
```python
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=80000, ngram_range=(1,2), sublinear_tf=True)),
    ('clf',   LogisticRegression(C=3.0, max_iter=1000, solver='lbfgs', random_state=42))
])
```
**Why Logistic Regression over Deep Learning for this task?**
1. **Speed & Latency**: Generates predictions in `< 1 millisecond` on standard CPU hardware without GPU overhead.
2. **Interpretability**: Linear feature coefficients directly expose why a review received its score (e.g., examining top positive and negative n-grams).
3. **High Accuracy on High-Dimensional Sparse Text**: Achieving **87% accuracy** and **0.945 ROC-AUC** rivaling complex Transformer models with a fraction of memory footprint (~935 KB serialized).

### Evaluation & Performance Metrics

| Metric | Validation Set | Test Set |
|---|---|---|
| **Accuracy** | **87.31%** | **87.00%** |
| **ROC-AUC** | **0.9467** | **0.9448** |
| **Negative Precision / Recall / F1** | `0.88` / `0.86` / `0.87` | `0.88` / `0.84` / `0.86` |
| **Positive Precision / Recall / F1** | `0.87` / `0.89` / `0.88` | `0.86` / `0.89` / `0.88` |

Visual artifacts generated:
- `data/eval_confusion_roc.png` (Confusion Matrix heatmap and ROC Curve)
- `data/eval_top_features.png` (Top 20 most influential negative and positive terms)

### Model Freezing & Serialization
The fitted pipeline is serialized using `joblib` with level-3 compression into:
- `backend/models/sentiment_pipeline.joblib` (935 KB)
- `backend/models/model_metadata.json` (architecture parameters, metrics, timestamp)

---

## 📓 Jupyter Notebook Walkthrough

The notebook [sentiment_analysis.ipynb](file:///d:/sentiment-analysis/notebook/sentiment_analysis.ipynb) is organized into 9 clear, self-contained sections:

1. **Imports & Setup**: Configures visualization styles, seeds, and downloads NLTK corpora.
2. **Dataset Acquisition**: Streams data from Hugging Face or loads from the local parquet cache.
3. **Exploratory Data Analysis**: Generates summary statistics and high-resolution plots for distributions.
4. **Text Preprocessing**: Normalizes reviews, strips noise, and demonstrates raw vs. cleaned token samples.
5. **Data Splitting**: Stratified 85/15 train/val split alongside dedicated holdout test set.
6. **Pipeline Construction & Fitting**: Configures the TF-IDF vectorizer and trains the linear classifier.
7. **Comprehensive Evaluation**: Classification reports, confusion matrices, ROC curves, and coefficient rankings.
8. **Model Export (Freezing)**: Dumps pipeline and metadata to `backend/models/`.
9. **Inference Verification**: Executes real-world test reviews through the loaded artifact.

---

## ⚡ FastAPI Backend

The backend is built with **FastAPI** and **Uvicorn** for maximum asynchronous throughput.

### Design & Lifespan Management
- **Lifespan Context**: Loads the model pipeline into memory once during application startup. If the model file is not present, the API gracefully reports `503 Service Unavailable` with actionable instructions.
- **Dual Routing**: All endpoints are mounted at both the root level and under `/api/*`, guaranteeing full compatibility with Vite proxying during development and direct URL access in production.
- **CORS Configured**: Permissive CORS headers enabled for flexible deployment configurations.
- **Pre-Compiled UI Serving**: If `frontend/dist/` exists, FastAPI automatically mounts static assets and serves `index.html` at `/` for single-port deployments.

### API Reference & Examples

#### 1. Single Prediction
- **Endpoint**: `POST /predict` (or `POST /api/predict`)
- **Request Body**:
```json
{
  "title": "Superb sound quality",
  "text": "These headphones blew me away! Crisp highs and deep bass. Worth every penny."
}
```
- **Response**:
```json
{
  "sentiment": "positive",
  "confidence": 0.9421,
  "positive_prob": 0.9421,
  "negative_prob": 0.0579,
  "processed_text": "superb sound quality headphone blew away crisp high deep bass worth every penny",
  "inference_ms": 0.74
}
```

#### 2. Batch Prediction
- **Endpoint**: `POST /predict/batch` (or `POST /api/predict/batch`)
- **Request Body**:
```json
{
  "reviews": [
    {"text": "Broke immediately after opening. Do not buy!"},
    {"text": "Delighted with my purchase. Fast shipping and high quality."}
  ]
}
```
- **Response**:
```json
{
  "count": 2,
  "results": [
    {
      "sentiment": "negative",
      "confidence": 0.9852,
      "positive_prob": 0.0148,
      "negative_prob": 0.9852,
      "inference_ms": 0.62,
      "text_preview": "Broke immediately after opening. Do not buy!"
    },
    {
      "sentiment": "positive",
      "confidence": 0.9124,
      "positive_prob": 0.9124,
      "negative_prob": 0.0876,
      "inference_ms": 0.58,
      "text_preview": "Delighted with my purchase. Fast shipping and high quality."
    }
  ]
}
```

#### 3. Model Information
- **Endpoint**: `GET /model/info` (or `GET /api/model/info`)
- **Response**:
```json
{
  "model_type": "TF-IDF + Logistic Regression Pipeline",
  "dataset": "fancyzhx/amazon_polarity (Amazon Review Polarity)",
  "test_accuracy": 0.87,
  "test_roc_auc": 0.9448,
  "train_samples": 21250,
  "model_loaded": true
}
```

#### 4. Example Reviews
- **Endpoint**: `GET /examples` (or `GET /api/examples`)
- Returns a set of curated positive, negative, and mixed customer reviews for immediate UI testing.

#### 5. Interactive Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## 🎨 React Frontend UI

The client application is built with **React** and bundled via **Vite**.

### Design System & Aesthetics
- **Dark-Mode Glassmorphism**: Tailored color palette with deep obsidian backgrounds (`#0a0f1e`), translucent layered cards (`rgba(255, 255, 255, 0.04)`), and subtle ambient glow effects.
- **Modern Typography**: Inter variable font loaded from Google Fonts for clean readability.
- **Dynamic Micro-Interactions**: Hover elevations, button state transitions, and smooth CSS entry animations.

### Key Interactive Features
- **Circular SVG Confidence Meter**: Dynamic SVG gauge displaying the model's confidence percentage with color transitions (Emerald for positive, Coral-red for negative).
- **Probability Breakdown Bars**: Real-time side-by-side visualization of negative vs. positive probability distribution.
- **Live Latency & Token Counters**: Displays execution latency in milliseconds and preprocessed token counts.
- **Quick-Fill Preset Buttons**: One-click review testing with presets like *Strong Positive*, *Strong Negative*, *Mild Positive*, and *Mixed/Neutral*.
- **Analysis History**: Persists previous analysis runs in memory with click-to-load functionality.
- **Real-time Backend Status Pill**: Header indicator displaying backend connection status and model metadata.

---

## 💻 Installation & Getting Started

### Prerequisites
- **Python**: Version `3.10` or higher (verified on Python 3.13).
- **Node.js**: Version `18.0.0` or higher (verified on Node.js 24).
- **Git** (optional).

### Dependency Installation

#### 1. Backend Python Dependencies
From the root of the project:
```bash
pip install -r backend/requirements.txt
```
*(Optionally install training/visualization packages if retraining: `pip install datasets matplotlib seaborn pyarrow`)*

#### 2. Frontend Node Dependencies
Navigate into the `frontend` directory and install npm packages:
```bash
cd frontend
npm install
cd ..
```

---

## 🚀 Running the Application

### Method 1: One-Command Dev Mode (Recommended)
Launches both the FastAPI backend and Vite React frontend concurrently with live reloading and opens your browser:
```bash
python start_dev.py
```
- **React Frontend (HMR)**: [http://localhost:5173](http://localhost:5173)
- **FastAPI Backend**: [http://localhost:8000](http://localhost:8000)
- **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Stop**: Press `Ctrl+C` in your terminal to shut down both servers cleanly.

---

### Method 2: Windows Batch Launcher
If you are on Windows, simply double-click the included batch script:
```cmd
start.bat
```

---

### Method 3: Unified Single-Port Server
If you prefer running the entire full-stack application on a single port (`8000`), the pre-built React app is served directly by FastAPI:
```bash
python run.py
```
- Open [http://localhost:8000](http://localhost:8000) in your web browser.

---

### Method 4: Independent Server Launch
You can also run both services in separate terminal windows:

**Terminal 1 (Backend):**
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 (Frontend):**
```bash
cd frontend
npm run dev
```

---

## 🧪 Testing & Verification

### Test via cURL
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Outstanding battery life and great build quality!"}'
```

### Test via Python Script
```python
import urllib.request
import json

req = urllib.request.Request(
    "http://localhost:8000/predict",
    data=json.dumps({"text": "Horrible purchase, arrived broken."}).encode(),
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req) as response:
    print(json.loads(response.read().decode()))
```
**Expected Output:**
```json
{
  "sentiment": "negative",
  "confidence": 0.9912,
  "positive_prob": 0.0088,
  "negative_prob": 0.9912,
  "processed_text": "horrible purchase arrived broken",
  "inference_ms": 0.81
}
```

---

## ❓ Troubleshooting & FAQs

**Q: I get `Model not loaded` when accessing the API.**  
A: Ensure `backend/models/sentiment_pipeline.joblib` exists. If missing, simply run `python train.py` to train and freeze the model.

**Q: `npm` command is not recognized in PowerShell on Windows.**  
A: Use `python start_dev.py` which automatically locates your Node.js installation (e.g. `C:\Program Files\nodejs\npm.cmd`) without requiring manual environment path edits.

**Q: Can I retrain the model with more data?**  
A: Yes! Open `train.py` or `notebook/sentiment_analysis.ipynb` and adjust `TRAIN_SAMPLES = 50_000` or higher. Running `python train.py` will regenerate the plots and update the frozen model artifact.

---

## 📄 License
This project is open-source and available under the **MIT License**.
