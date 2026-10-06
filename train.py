"""
Train the sentiment model using the CyranoB/polarity dataset.
This script mirrors the notebook logic so it can be run standalone
(no Jupyter required) to produce backend/models/sentiment_pipeline.joblib.
"""

import os, re, json, time, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',   quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

warnings.filterwarnings('ignore')

# ─── Config ──────────────────────────────────────────────────────────────────
TRAIN_SAMPLES = 25_000
TEST_SAMPLES  = 5_000
CACHE_FILE    = os.path.join('data', 'polarity_sample.parquet')
MODEL_DIR     = os.path.join('backend', 'models')
DATA_DIR      = 'data'

STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS.update({'br','would','one','also','get','got','use','used','using','like','make','made'})
lemmatizer = WordNetLemmatizer()

os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

NEG_COLOR = '#EF4444'
POS_COLOR = '#10B981'
ACCENT    = '#6366F1'


def preprocess(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split()
              if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(tokens)


# ─── 1. Load / Cache Dataset ──────────────────────────────────────────────────
if os.path.exists(CACHE_FILE):
    print(f'Loading cached data from {CACHE_FILE}...')
    df = pd.read_parquet(CACHE_FILE)
    print(f'  Loaded {len(df):,} rows')
else:
    print('Streaming CyranoB/polarity from HuggingFace...')
    ds_train = load_dataset('fancyzhx/amazon_polarity', split='train', streaming=True)
    ds_test  = load_dataset('fancyzhx/amazon_polarity', split='test',  streaming=True)

    train_rows, test_rows = [], []
    for i, row in enumerate(ds_train):
        if i >= TRAIN_SAMPLES: break
        train_rows.append(row)
        if (i+1) % 10000 == 0: print(f'  train: {i+1:,} / {TRAIN_SAMPLES:,}')

    for i, row in enumerate(ds_test):
        if i >= TEST_SAMPLES: break
        test_rows.append(row)
        if (i+1) % 5000 == 0: print(f'  test : {i+1:,} / {TEST_SAMPLES:,}')

    df_train = pd.DataFrame(train_rows); df_train['split'] = 'train'
    df_test  = pd.DataFrame(test_rows);  df_test['split']  = 'test'
    df = pd.concat([df_train, df_test], ignore_index=True)
    df.to_parquet(CACHE_FILE, index=False)
    print(f'  Saved {len(df):,} rows to {CACHE_FILE}')

print(f'\nDataset: {df.shape}  |  columns: {df.columns.tolist()}')
print(f'Labels:\n{df["label"].value_counts().rename({0:"negative",1:"positive"})}')


# ─── 2. EDA Plots ─────────────────────────────────────────────────────────────
print('\n--- EDA Plots ---')
plt.style.use('seaborn-v0_8-whitegrid')

# Class distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
label_counts = df['label'].value_counts().sort_index()
axes[0].bar(['Negative','Positive'], label_counts.values, color=[NEG_COLOR,POS_COLOR], edgecolor='white', linewidth=2, width=0.5)
axes[0].set_title('Class Distribution'); axes[0].set_ylabel('Count')
pct = label_counts.values / label_counts.values.sum() * 100
axes[1].pie(label_counts.values, labels=[f'Negative ({pct[0]:.1f}%)',f'Positive ({pct[1]:.1f}%)'],
            colors=[NEG_COLOR,POS_COLOR], startangle=90, wedgeprops={'edgecolor':'white','linewidth':3})
axes[1].set_title('Class Proportions')
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'eda_class_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  Saved eda_class_distribution.png')

# Text length
df['content_words'] = df['content'].fillna('').str.split().str.len()
df['title_words']   = df['title'].fillna('').str.split().str.len()
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
neg = df[df['label']==0]['content_words']; pos = df[df['label']==1]['content_words']
axes[0].hist(neg.clip(0,300), bins=60, alpha=0.65, color=NEG_COLOR, label='Negative', density=True)
axes[0].hist(pos.clip(0,300), bins=60, alpha=0.65, color=POS_COLOR, label='Positive', density=True)
axes[0].axvline(neg.median(), color=NEG_COLOR, linestyle='--', lw=2)
axes[0].axvline(pos.median(), color=POS_COLOR, linestyle='--', lw=2)
axes[0].set_title('Review Word Count'); axes[0].legend()
axes[1].hist(df[df['label']==0]['title_words'].clip(0,30), bins=30, alpha=0.65, color=NEG_COLOR, label='Negative', density=True)
axes[1].hist(df[df['label']==1]['title_words'].clip(0,30), bins=30, alpha=0.65, color=POS_COLOR, label='Positive', density=True)
axes[1].set_title('Title Word Count'); axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'eda_text_length.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  Saved eda_text_length.png')

# Top keywords
def get_top_words(texts, n=20):
    words = []
    for t in texts:
        if not isinstance(t, str): continue
        words.extend([w for w in re.sub(r'[^a-z ]', '', t.lower()).split()
                       if w not in STOP_WORDS and len(w) > 2])
    return Counter(words).most_common(n)

neg_words = get_top_words(df[df['label']==0]['content'])
pos_words = get_top_words(df[df['label']==1]['content'])
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Top 20 Keywords by Sentiment', fontweight='bold')
for ax, wcs, color, title in [(axes[0], neg_words, NEG_COLOR, 'Negative'), (axes[1], pos_words, POS_COLOR, 'Positive')]:
    words, counts = zip(*wcs)
    ax.barh(range(len(words)), counts, color=color, alpha=0.85)
    ax.set_yticks(range(len(words))); ax.set_yticklabels(words, fontsize=11)
    ax.invert_yaxis(); ax.set_title(f'{title} Reviews', fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'eda_top_keywords.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  Saved eda_top_keywords.png')


# ─── 3. Preprocessing ─────────────────────────────────────────────────────────
print('\n--- Preprocessing ---')
df['text_clean'] = (df['title'].fillna('') + ' ' + df['content'].fillna('')).apply(preprocess)
print(f'  Done. Sample: {df["text_clean"].iloc[0][:120]}')


# ─── 4. Split ─────────────────────────────────────────────────────────────────
df_train = df[df['split']=='train']
df_test  = df[df['split']=='test']
X_train, X_val, y_train, y_val = train_test_split(
    df_train['text_clean'], df_train['label'],
    test_size=0.15, random_state=42, stratify=df_train['label']
)
X_test, y_test = df_test['text_clean'], df_test['label']
print(f'  Train={len(X_train):,}  Val={len(X_val):,}  Test={len(X_test):,}')


# ─── 5. Train ─────────────────────────────────────────────────────────────────
print('\n--- Training TF-IDF + Logistic Regression ---')
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=80_000, ngram_range=(1,2), min_df=3, max_df=0.90, sublinear_tf=True)),
    ('clf',   LogisticRegression(C=3.0, max_iter=1000, solver='lbfgs', random_state=42))
])
t0 = time.time()
pipeline.fit(X_train, y_train)
print(f'  Training done in {time.time()-t0:.1f}s')


# ─── 6. Evaluate ──────────────────────────────────────────────────────────────
print('\n--- Evaluation ---')
for name, X, y in [('Val', X_val, y_val), ('Test', X_test, y_test)]:
    acc = accuracy_score(y, pipeline.predict(X))
    auc = roc_auc_score(y, pipeline.predict_proba(X)[:,1])
    print(f'  {name}: Accuracy={acc:.4f}  ROC-AUC={auc:.4f}')
    print(classification_report(y, pipeline.predict(X), target_names=['Negative','Positive']))

# Save evaluation plots
from sklearn.metrics import confusion_matrix, roc_curve
y_pred_test = pipeline.predict(X_test)
y_prob_test = pipeline.predict_proba(X_test)[:,1]
cm = confusion_matrix(y_test, y_pred_test)
cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
annot = np.array([[f'{cm[i,j]:,}\n({cm_pct[i,j]*100:.1f}%)' for j in range(2)] for i in range(2)])
fpr, tpr, _ = roc_curve(y_test, y_prob_test)
auc_score   = roc_auc_score(y_test, y_prob_test)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm_pct, annot=annot, fmt='', cmap='RdYlGn', ax=axes[0],
            xticklabels=['Pred Neg','Pred Pos'], yticklabels=['Actual Neg','Actual Pos'], linewidths=2, linecolor='white')
axes[0].set_title('Confusion Matrix'); axes[0].tick_params(axis='y', rotation=0)
axes[1].plot(fpr, tpr, color=ACCENT, lw=2.5, label=f'AUC={auc_score:.3f}')
axes[1].plot([0,1],[0,1],'k--',lw=1.5)
axes[1].fill_between(fpr, tpr, alpha=0.15, color=ACCENT)
axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR')
axes[1].set_title('ROC Curve'); axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'eval_confusion_roc.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  Saved eval_confusion_roc.png')

# Top features
feat_names = pipeline.named_steps['tfidf'].get_feature_names_out()
coefs      = pipeline.named_steps['clf'].coef_[0]
top_neg = np.argsort(coefs)[:20]; top_pos = np.argsort(coefs)[-20:][::-1]
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, idx, color, title in [(axes[0],top_neg,NEG_COLOR,'Top Negative Predictors'),
                               (axes[1],top_pos,POS_COLOR,'Top Positive Predictors')]:
    words = [feat_names[i] for i in idx]; vals = [abs(coefs[i]) for i in idx]
    ax.barh(range(len(words)), vals, color=color, alpha=0.8)
    ax.set_yticks(range(len(words))); ax.set_yticklabels(words, fontsize=11)
    ax.invert_yaxis(); ax.set_title(title, fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'eval_top_features.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  Saved eval_top_features.png')


# ─── 7. Save Model ────────────────────────────────────────────────────────────
print('\n--- Saving Frozen Model ---')
test_acc = accuracy_score(y_test, pipeline.predict(X_test))
test_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:,1])

pipeline_path = os.path.join(MODEL_DIR, 'sentiment_pipeline.joblib')
joblib.dump(pipeline, pipeline_path, compress=3)

metadata = {
    'model_type': 'TF-IDF + Logistic Regression Pipeline',
    'dataset': 'fancyzhx/amazon_polarity (Amazon Review Polarity — same data as CyranoB/polarity)',
    'train_samples': int(len(X_train)),
    'test_samples': int(len(X_test)),
    'tfidf_max_features': 80000,
    'ngram_range': '(1,2)',
    'test_accuracy': round(test_acc, 4),
    'test_roc_auc': round(test_auc, 4),
    'classes': {'0': 'negative', '1': 'positive'}
}
with open(os.path.join(MODEL_DIR, 'model_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print(f'  Model saved: {pipeline_path}')
print(f'  Test accuracy: {test_acc*100:.2f}%')
print(f'  Test ROC-AUC : {test_auc:.4f}')


# ─── 8. Inference Test ────────────────────────────────────────────────────────
print('\n--- Inference Test (Frozen Model) ---')
loaded = joblib.load(pipeline_path)
test_reviews = [
    ("Absolutely amazing product! Best purchase ever!", 'positive'),
    ("Terrible quality. Complete waste of money.", 'negative'),
    ("Average product. Nothing special.", 'neutral'),
]
for text, expected in test_reviews:
    cleaned = preprocess(text)
    proba = loaded.predict_proba([cleaned])[0]
    pred  = 'positive' if proba[1] > proba[0] else 'negative'
    conf  = max(proba)*100
    print(f'  [{pred.upper():8s} {conf:.1f}%]  {text}')

print('\n[SUCCESS] Training complete! Model ready for the FastAPI backend.')
