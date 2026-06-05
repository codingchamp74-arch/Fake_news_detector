"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  FAKE NEWS DETECTOR — Linear SVM                                            ║
║  Label mapping: 0 = FAKE, 1 = REAL                                          ║
║  Run: python fake_news_detector.py                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, re, string, warnings, time, pickle
import numpy as np
import pandas as pd

import nltk
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.svm import LinearSVC

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
TRAIN_PATH  = r"C:\Users\Akshit Tyagi\Downloads\train.csv"
TEST_PATH   = r"C:\Users\Akshit Tyagi\Downloads\test.csv"

MODEL_PKL   = "linsvm_model.pkl"
VEC_PKL     = "tfidf_vectorizer.pkl"
RESULTS_DIR = "results"

RANDOM_STATE = 42
MAX_FEATURES = 5000
NGRAM_RANGE  = (1, 2)
CV_FOLDS     = 5
SAMPLE_SIZE  = 50

os.makedirs(RESULTS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  LABEL MAPPING (confirmed from your dataset)
#  This is correct — NEVER inverted:
#    0 = FAKE
#    1 = REAL
# ══════════════════════════════════════════════════════════════════════════════
LABEL_MAP = {0: "FAKE", 1: "REAL"}

# ══════════════════════════════════════════════════════════════════════════════
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def combine_row(row):
    parts = []
    for c in ["title", "author", "text"]:
        if c in row and str(row[c]).strip():
            parts.append(str(row[c]))
    return " ".join(parts)


def predict(text, model, vec):
    c = clean_text(text)
    x = vec.transform([c])
    num = model.predict(x)[0]
    lbl = LABEL_MAP[num]
    try:
        s = model.decision_function(x)[0]
        cf = round(1 / (1 + np.exp(-abs(s))) * 100, 1)
    except Exception:
        cf = None
    return lbl, cf


# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 55)
    print("  FAKE NEWS DETECTOR - Linear SVM")
    print(f"  Labels: 0=FAKE  1=REAL")
    print("=" * 55)

    # ----------------------------------------
    # LOAD
    # ----------------------------------------
    print("-" * 55)
    print(f"Loading: {TRAIN_PATH}")
    print("-" * 55)

    df = pd.read_csv(TRAIN_PATH, index_col=0)
    print(f"  Train: {df.shape[0]:,} rows, cols={list(df.columns)}")

    label_col = "label" if "label" in df.columns else "labels"
    raw = df[label_col].values
    if np.issubdtype(raw.dtype, np.floating):
        raw = raw.astype(int)
    if not np.issubdtype(raw.dtype, np.integer):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        raw = le.fit_transform(raw)
    y = raw.astype(int)
    print(f"  FAKE (0)={sum(y==0):,}   REAL (1)={sum(y==1):,}")

    for c in ["title", "author", "text"]:
        if c in df.columns:
            df[c] = df[c].fillna("")

    X = df.apply(combine_row, axis=1)

    X_tr, X_held, y_tr, y_held = train_test_split(
        X.values, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_train = pd.Series(X_tr)
    X_held  = pd.Series(X_held)
    y_train = pd.Series(y_tr)
    y_held  = pd.Series(y_held)
    print(f"  Split: Train={len(X_train):,}  Test={len(X_held):,}")

    # ----------------------------------------
    # CLEAN
    # ----------------------------------------
    print("\nCleaning text...")
    t0 = time.time()
    X_tr_c = X_train.apply(clean_text)
    X_hl_c = X_held.apply(clean_text)
    ok_tr = X_tr_c.str.strip() != ""
    ok_hl = X_hl_c.str.strip() != ""
    X_tr_c = X_tr_c[ok_tr].reset_index(drop=True)
    y_train = y_train[ok_tr].reset_index(drop=True)
    X_hl_c = X_hl_c[ok_hl].reset_index(drop=True)
    y_held  = y_held[ok_hl].reset_index(drop=True)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ----------------------------------------
    # TF-IDF
    # ----------------------------------------
    print(f"\nTF-IDF (max_features={MAX_FEATURES})...")
    t0 = time.time()
    vec = TfidfVectorizer(
        max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE,
        min_df=3, max_df=0.85, sublinear_tf=True,
        strip_accents="unicode", token_pattern=r"\w{1,}"
    )
    X_tr_vec = vec.fit_transform(X_tr_c)
    X_hl_vec = vec.transform(X_hl_c)
    print(f"  Done in {time.time()-t0:.1f}s -> {X_tr_vec.shape[1]} features")

    # ----------------------------------------
    # TRAIN
    # ----------------------------------------
    print("\n" + "=" * 50)
    print("TRAINING Linear SVM...")
    print("=" * 50)

    model = LinearSVC(C=1.0, loss="squared_hinge", penalty="l2",
                       dual=False, max_iter=2000, random_state=RANDOM_STATE)
    t0 = time.time()
    model.fit(X_tr_vec, y_train)
    train_t = time.time() - t0

    y_pred_all = model.predict(X_hl_vec)
    acc  = accuracy_score(y_held, y_pred_all)
    prec = precision_score(y_held, y_pred_all, average="macro", zero_division=0)
    rec  = recall_score(y_held, y_pred_all, average="macro", zero_division=0)
    f1   = f1_score(y_held, y_pred_all, average="macro", zero_division=0)
    cm   = confusion_matrix(y_held, y_pred_all)
    try:
        roc = roc_auc_score(y_held, model.decision_function(X_hl_vec))
    except Exception:
        roc = None
    cv = cross_val_score(
        model, X_tr_vec, y_train,
        cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring="accuracy", n_jobs=-1
    )

    print(f"  Training time : {train_t:.1f}s")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  Precision     : {prec:.4f}")
    print(f"  Recall        : {rec:.4f}")
    print(f"  F1 Score      : {f1:.4f}")
    if roc: print(f"  ROC-AUC       : {roc:.4f}")
    print(f"  CV ({CV_FOLDS}-fold)  : {cv.mean():.4f} +/- {cv.std():.4f}")

    # ----------------------------------------
    # SAVE
    # ----------------------------------------
    pickle.dump(model, open(MODEL_PKL, "wb"))
    pickle.dump(vec,   open(VEC_PKL, "wb"))
    print(f"\n  Saved: {MODEL_PKL}, {VEC_PKL}")

    # ----------------------------------------
    # PLOTS
    # ----------------------------------------
    print(f"\nSaving 3 plots to '{RESULTS_DIR}/'...")

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=1, linecolor="gray",
                xticklabels=["FAKE (pred)", "REAL (pred)"],
                yticklabels=["FAKE (actual)", "REAL (actual)"],
                ax=ax, annot_kws={"fontsize": 18, "fontweight": "bold"})
    ax.set_title(f"Linear SVM  |  Acc={acc:.4f}  F1={f1:.4f}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(["Linear SVM"], [cv.mean()], xerr=[cv.std()],
            color="#6366F1", edgecolor="white", capsize=8, height=0.4)
    ax.set_xlim(0.85, 1.0); ax.set_xlabel("Accuracy")
    ax.set_title(f"{CV_FOLDS}-Fold Cross-Validation", fontsize=14, fontweight="bold")
    ax.text(cv.mean()+cv.std()+0.003, 0, f"{cv.mean():.4f} +/- {cv.std():.4f}", va="center", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "cross_validation.png"), dpi=150); plt.close()

    coef = model.coef_[0]; fn = vec.get_feature_names_out(); n = 20
    t1 = np.argsort(coef)[-n:][::-1]   # class 1 = REAL
    t0 = np.argsort(coef)[:n]          # class 0 = FAKE
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    axes[0].barh(range(n), coef[t1][::-1], color="#4ECDC4"); axes[0].set_yticks(range(n))
    axes[0].set_yticklabels([fn[i] for i in t1][::-1], fontsize=11)
    axes[0].set_title("Top words -> REAL", fontsize=14, fontweight="bold")
    axes[0].axvline(0, color="black")
    axes[1].barh(range(n), coef[t0], color="#FF6B6B"); axes[1].set_yticks(range(n))
    axes[1].set_yticklabels([fn[i] for i in t0], fontsize=11)
    axes[1].set_title("Top words -> FAKE", fontsize=14, fontweight="bold")
    axes[1].axvline(0, color="black")
    fig.suptitle("Most Discriminative Words (Linear SVM Coefficients)", fontsize=16, fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "top_words.png"), dpi=150); plt.close()
    print("  Done")

    # ══════════════════════════════════════════════════════════════════════
    #  BATCH TEST — from YOUR dataset
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 55)
    print(f"  BATCH TEST — {SAMPLE_SIZE} REAL + {SAMPLE_SIZE} FAKE")
    print("  (actual articles from YOUR dataset)")
    print("=" * 55)

    class0_idx = np.where(y_held.values == 0)[0]   # FAKE
    class1_idx = np.where(y_held.values == 1)[0]   # REAL
    np.random.seed(42)
    fake_idx = np.random.choice(class0_idx, size=min(SAMPLE_SIZE, len(class0_idx)), replace=False)
    real_idx = np.random.choice(class1_idx, size=min(SAMPLE_SIZE, len(class1_idx)), replace=False)

    # ---- REAL ----
    print(f"\n  REAL NEWS (should be REAL):")
    real_ok = 0
    for idx in real_idx:
        text = X_held.iloc[idx]
        pred_lbl, conf = predict(text, model, vec)
        if pred_lbl == "REAL": real_ok += 1
        else:
            snippet = text[:110].replace("\n", " ")
            print(f"  [WRONG] \"{snippet}...\"  -> got: {pred_lbl}")
    print(f"  REAL correct: {real_ok}/{len(real_idx)}")

    # ---- FAKE ----
    print(f"\n  FAKE NEWS (should be FAKE):")
    fake_ok = 0
    for idx in fake_idx:
        text = X_held.iloc[idx]
        pred_lbl, conf = predict(text, model, vec)
        if pred_lbl == "FAKE": fake_ok += 1
        else:
            snippet = text[:110].replace("\n", " ")
            print(f"  [WRONG] \"{snippet}...\"  -> got: {pred_lbl}")
    print(f"  FAKE correct: {fake_ok}/{len(fake_idx)}")

    total = real_ok + fake_ok
    denom = len(real_idx) + len(fake_idx)
    print(f"\n  ==========================================")
    print(f"  RESULT: {total}/{denom} = {total/denom*100:.1f}%")
    if total == denom:
        print(f"  PERFECT - model works correctly!")
    elif total >= denom * 0.9:
        print(f"  EXCELLENT RESULT")
    print(f"  ==========================================")

    # ══════════════════════════════════════════════════════════════════════
    #  INTERACTIVE TEST
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n" + "-" * 55)
    print(f"TEST CUSTOM HEADLINES  ('skip' = quit)")
    print(f"Labels: 0=FAKE  1=REAL")
    print("-" * 55)

    while True:
        try:
            text = input("\n  News: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in ("skip", "quit", "exit", ""):
            break
        if not text:
            continue
        lbl, cf = predict(text, model, vec)
        c = f" ({cf}%)" if cf else ""
        print(f"  -> {lbl}{c}")

    print("\nDone!\n")


if __name__ == "__main__":
    main()
