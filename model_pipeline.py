import os
import time
from pprint import pprint
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from category_encoders import BinaryEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import warnings

warnings.filterwarnings("ignore")

def clean_and_repair_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles systematic parsing anomalies such as duration attributes 
    leaking into the rating column, missing values, and date parsing.
    """
    df = df.copy()
    
    # 1. Convert date_added to datetime format
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    
    # 2. Fix the structural shifting flaw between duration and rating columns
    mask = (
        df['duration'].isna() & 
        df['rating'].astype(str).str.contains(r'^\d+\s*min$', na=False)
    )
    df.loc[mask, 'duration'] = df.loc[mask, 'rating']
    df.loc[mask, 'rating'] = np.nan
    
    # 3. High-missingness categorical columns imputation
    high_missing_cols = ['director', 'cast', 'country']
    for col in high_missing_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            
    return df

def impute_missing_values(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """
    Imputes numerical and low-missingness categorical values using training statistics 
    to prevent downstream leakage during inference.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    mean_treat_cols = ['release_year']
    
    # Impute date_added and release_year with mean training values
    train_date_mean = train_df['date_added'].mean()
    train_df['date_added'] = train_df['date_added'].fillna(train_date_mean)
    test_df['date_added'] = test_df['date_added'].fillna(train_date_mean)
    
    for col in mean_treat_cols:
        train_mean = train_df[col].mean()
        train_df[col] = train_df[col].fillna(train_mean)
        test_df[col] = test_df[col].fillna(train_mean)
        
    # Impute rating with the mode grouped by type from the training set
    duration_mode_by_type = (
        train_df.groupby('type')['rating']
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown')
    )
    
    train_df['rating'] = train_df['rating'].fillna(train_df['type'].map(duration_mode_by_type))
    test_df['rating'] = test_df['rating'].fillna(test_df['type'].map(duration_mode_by_type))
    
    # Safe backup for any leftover NaN categories
    train_df['rating'] = train_df['rating'].fillna('Unknown')
    test_df['rating'] = test_df['rating'].fillna('Unknown')
    
    return train_df, test_df

def extract_and_transform_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses and builds numeric specific features for Movie runtimes and TV Show seasons.
    """
    df = df.copy()
    
    df['duration_num'] = df['duration'].astype(str).str.extract(r'(\d+)').astype(float)
    
    df['duration_movies'] = np.where(df['type'] == 'Movie', df['duration_num'], np.nan)
    df['duration_tv_series'] = np.where(df['type'] == 'TV Show', df['duration_num'], np.nan)
    
    df[['duration_movies', 'duration_tv_series']] = df[['duration_movies', 'duration_tv_series']].fillna(0)
    df.drop(columns=['duration', 'duration_num'], inplace=True, errors='ignore')
    
    return df

def apply_quantile_binning(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """
    Transforms numerical variables into categorical bin numbers based on training quartiles.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    train_df['date_added'] = train_df['date_added'].dt.year
    test_df['date_added'] = test_df['date_added'].dt.year
    
    cols_to_bin = ['duration_movies', 'duration_tv_series', 'date_added', 'release_year']
    
    for col in cols_to_bin:
        # Compute quantile boundaries on train only
        _, bins = pd.qcut(train_df[col], q=6, labels=False, retbins=True, duplicates='drop')
        
        train_df[f'{col}_bin'] = pd.cut(train_df[col], bins=bins, labels=False, include_lowest=True)
        test_df[f'{col}_bin'] = pd.cut(test_df[col], bins=bins, labels=False, include_lowest=True)
        
        # Fill boundary overflows with edge values
        train_df[f'{col}_bin'] = train_df[f'{col}_bin'].fillna(0).astype(int)
        test_df[f'{col}_bin'] = test_df[f'{col}_bin'].fillna(0).astype(int)
        
    train_df.drop(columns=cols_to_bin, inplace=True)
    test_df.drop(columns=cols_to_bin, inplace=True)
    
    return train_df, test_df

def encode_categorical_and_text(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """
    Performs manual category mapping, Binary Encoding for ratings, and LSA on concatenated text features.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # 1. Manual Encodings
    platform_map = {'Disney': 0, 'Netflix': 1}
    type_map = {'Movie': 0, 'TV Show': 1}
    
    for df in [train_df, test_df]:
        df['platform'] = df['platform'].replace(platform_map).fillna(0).astype(int)
        df['type'] = df['type'].replace(type_map).fillna(0).astype(int)
        
    # 2. Binary Encoding on High-Cardinality rating feature
    be = BinaryEncoder(cols=['rating'])
    train_encoded = be.fit_transform(train_df[['rating']])
    test_encoded = be.transform(test_df[['rating']])
    
    train_df = pd.concat([train_df.drop(columns=['rating']), train_encoded], axis=1)
    test_df = pd.concat([test_df.drop(columns=['rating']), test_encoded], axis=1)
    
    # 3. Comprehensive LSA Text Representation Pipeline (TF-IDF + Truncated SVD)
    text_cols = ['country', 'title', 'director', 'cast', 'description']
    
    train_tfidf_matrices = []
    test_tfidf_matrices = []
    
    for col in text_cols:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        
        train_vec = vectorizer.fit_transform(train_df[col].fillna('').astype(str))
        test_vec = vectorizer.transform(test_df[col].fillna('').astype(str))
        
        train_tfidf_matrices.append(train_vec)
        test_tfidf_matrices.append(test_vec)
        
    X_train_text = hstack(train_tfidf_matrices)
    X_test_text = hstack(test_tfidf_matrices)
    
    max_components = min(X_train_text.shape) - 1
    svd = TruncatedSVD(n_components=max_components, random_state=42)
    
    X_train_svd = svd.fit_transform(X_train_text)
    X_test_svd = svd.transform(X_test_text)
    
    # Find number of components to capture 85% variance threshold
    cum_var = np.cumsum(svd.explained_variance_ratio_)
    n_components = np.argmax(cum_var >= 0.85) + 1
    
    X_train_svd = X_train_svd[:, :n_components]
    X_test_svd = X_test_svd[:, :n_components]
    
    svd_cols = [f'text_svd_{i}' for i in range(n_components)]
    train_svd_df = pd.DataFrame(X_train_svd, columns=svd_cols, index=train_df.index)
    test_svd_df = pd.DataFrame(X_test_svd, columns=svd_cols, index=test_df.index)
    
    train_df = pd.concat([train_df.drop(columns=text_cols), train_svd_df], axis=1)
    test_df = pd.concat([test_df.drop(columns=text_cols), test_svd_df], axis=1)
    
    return train_df, test_df

def intersection_accuracy(y_true, y_pred):
    """
    Computes average intersection accuracy:
    |Actual ∩ Predicted| / |Actual|

    Parameters
    ----------
    y_true : ndarray (n_samples, n_labels)
    y_pred : ndarray (n_samples, n_labels)

    Returns
    -------
    float
        Average intersection accuracy.
    """

    # Number of correctly predicted labels per sample
    intersection = np.logical_and(y_true, y_pred).sum(axis=1)

    # Number of actual labels per sample
    actual = y_true.sum(axis=1)

    # Avoid division by zero
    scores = np.divide(
        intersection,
        actual,
        out=np.zeros_like(intersection, dtype=float),
        where=actual != 0
    )

    return float(scores.mean())

def run_production_pipeline(train_path: str = "train.csv", test_path: str = "test.csv", output_path: str = "prediction.csv"):
    """
    Main execution pipeline running full end-to-end multi-label genre modeling.
    """
    start_time = time.time()
    print(" | Starting execution of the pipeline...")

    # Load raw assets
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    
    # Capture original tracking indices for mapping output predictions
    if 'id' in train_raw.columns:
        train_raw.set_index('id', inplace=True)
    if 'id' in test_raw.columns:
        test_raw.set_index('id', inplace=True)
    
    print(" | Starting data pre-processing...")
    # Process multi-label target array
    y_raw = train_raw["listed_in"].astype(str).str.split(",").apply(lambda x: [i.strip() for i in x])
    mlb = MultiLabelBinarizer()
    Y_train = mlb.fit_transform(y_raw)
    
    X_train_raw = train_raw.drop(columns=["listed_in"], errors="ignore")
    X_test_raw = test_raw.copy()
    
    # Run structural pipeline steps
    X_train_cleaned = clean_and_repair_data(X_train_raw)
    X_test_cleaned = clean_and_repair_data(X_test_raw)
    
    X_train_imp, X_test_imp = impute_missing_values(X_train_cleaned, X_test_cleaned)
    
    print(" | Starting data transformation...")
    X_train_dur = extract_and_transform_duration(X_train_imp)
    X_test_dur = extract_and_transform_duration(X_test_imp)
    
    X_train_bin, X_test_bin = apply_quantile_binning(X_train_dur, X_test_dur)
    
    X_train_final, X_test_final = encode_categorical_and_text(X_train_bin, X_test_bin)
    
    # -- Cross-validation Block --
    print(" | Running Logistic Regression CV...\n")
    cv = MultilabelStratifiedKFold(
        n_splits=2,
        shuffle=True,
        random_state=42
    )
    
    lr_cv_model = OneVsRestClassifier(
        LogisticRegression(
            solver="liblinear",
            C=1.0,
            max_iter=750,
            class_weight="balanced",
            random_state=42
        ),
        n_jobs=-2
    )

    lr_scores = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X_train_final, Y_train), start=1):
        X_tr = X_train_final.iloc[train_idx]
        X_te = X_train_final.iloc[test_idx]
        Y_tr = Y_train[train_idx]
        Y_te = Y_train[test_idx]

        lr_cv_model.fit(X_tr, Y_tr)
        Y_pred_cv = lr_cv_model.predict(X_te)

        metrics = {
            "Fold": fold,
            "Micro_F1": round(f1_score(Y_te, Y_pred_cv, average="micro", zero_division=0), 6),
            "Macro_F1": round(f1_score(Y_te, Y_pred_cv, average="macro", zero_division=0), 6),
            "Accuracy": round(accuracy_score(Y_te, Y_pred_cv), 6),
            "Intersection_Accuracy": round(intersection_accuracy(Y_te, Y_pred_cv), 6),
        }
        lr_scores.append(metrics)

    lr_results = pd.DataFrame(lr_scores)
    print(" └─ Fold-wise Results:")
    print(lr_results)

    print("\n | Starting Model Predictions...")

    # Initialize high-dimensional linear multi-label baseline classifier
    lr_model = OneVsRestClassifier(
        LogisticRegression(
            solver="liblinear",
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        n_jobs=-2
    )
    
    # Model fitting and inference execution
    lr_model.fit(X_train_final, Y_train)
    Y_pred_binarized = lr_model.predict(X_test_final)
    
    # Decode back into comma-separated genre tokens
    decoded_genres = mlb.inverse_transform(Y_pred_binarized)
    predictions_joined = [", ".join(genres) if genres else "Unknown" for genres in decoded_genres]
    
    # Build final output artifact format matching initial conditions
    output_df = pd.DataFrame(index=test_raw.index)
    output_df['listed_in'] = predictions_joined
    output_df.reset_index(inplace=True)
    
    os.makedirs(os.path.join("data", "output"), exist_ok=True)
    output_df.to_csv(output_path, index=False)
    
    # Calculate end-to-end runtime differences
    end_time = time.time()
    elapsed_seconds = int(round(end_time - start_time))
    
    # Calculate H:M:S breakdown
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f" | Optimization sequence finished successfully.")
    print(f" | Prediction matrix outputted to: {output_path}")
    print(f" | Total E2E Execution Runtime [H:M:S]: {hours:02d}:{minutes:02d}:{seconds:02d}")

if __name__ == "__main__":
    output_dir = "data"
    run_production_pipeline(train_path=os.path.join(output_dir, "train.csv"), 
                            test_path=os.path.join(output_dir, "test.csv"), 
                            output_path=os.path.join(output_dir, "output", "prediction.csv"))