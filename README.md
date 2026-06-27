# Multi-Label Genre Classification Pipeline for OTT TV Shows & Movies

An end-to-end Machine Learning and Data Science pipeline built to analyze, pre-process, transform, and classify streaming content titles from OTT platforms into multiple genre categories. This repository transitions from initial exploratory analysis into a fully vectorized, engineered multi-label classification system using multi-class wrappers.

---

## 📂 Project Architecture & Repository Structure

The workflow is strictly split into two progressive stages across isolated Jupyter Notebooks:

1. **`EDA.ipynb` (Exploratory Data Analysis)**
   * Initial data quality auditing, missing value profiling, and temporal distribution analysis.
   * Visual discovery of systemic data alignment anomalies (e.g., duration leakage into rating masks).
   * Feature-space cardinality analysis across content providers, directors, cast members, and global production footprints.

2. **`Model_Pipeline.ipynb` (Feature Engineering & Model Pipeline)**
   * Dynamic imputation, conditional row repair, and text/categorical alignment.
   * Multi-modal vectorization combining text semantic embeddings via TF-IDF + Truncated SVD alongside structural Binary Encoding.
   * Multi-label classifier cross-validation (`OneVsRestClassifier`) benchmarking tree-based, linear, and ensemble methods.

---

## 📊 1. Deep-Dive Exploratory Data Analysis (`EDA.ipynb`)

Before code construction, an exhaustively documented exploratory phase was run on the dataset (`tv-shows.csv`), highlighting the following characteristics:

### A. Temporal Growth and Distribution
* **`date_added` Profile**: Converted from plain text to proper `datetime` format. Tracking month-over-month and year-over-year upload frequencies revealed a heavily right-skewed (positively skewed) distribution, demonstrating a rapid acceleration of content acquisition on streaming services over the past decade.
* **`release_year` Profile**: Also heavily right-skewed, exhibiting a significant concentration of contemporary media (2015–2021) mixed with sparse historical titles.

### B. Missing Value & Null Traversal
A comprehensive audit of the feature completeness revealed varying degrees of data missingness:
* **High Missingness**: `director`, `cast`, and `country` presented massive structural null blocks, meaning simple list-wise deletion would eliminate the vast majority of row records.
* **Low Missingness**: Columns like `date_added` and `rating` were highly complete but required strategic mode or median imputations to prevent downstream model failures.

### C. Structural Alignment Defect Discovery
* **The Shifting Anomaly**: Inspection of the `rating` column revealed a distinct structure: string text representing durations (e.g., `"74 min"`, `"84 min"`) had systematically leaked into the `rating` feature space. This pinpointed an upstream parsing flaw where missing categorical records forced numeric shifts. This finding informed a critical algorithmic correction block in the final pipeline.

### D. High-Cardinality Domain Specifics
* **Content Types**: Distribution skew between standard standalone `Movies` (measured in runtime minutes) and multi-part `TV Shows` (measured in seasons).
* **Exploded Cardinality**: Evaluating `cast`, `director`, and `country` fields required parsing comma-separated strings into single, unique items via string explosion. Top production regions were dominated by the US and India, while genre distributions presented a long-tail distribution covering 84 distinct overlapping categories.

---

## ⚙️ 2. Production Feature Engineering Pipeline (`Model_Pipeline.ipynb`)

The data transformation layers are fully optimized to cleanly bridge raw data inputs into high-performance numeric matrices without leaking validation data:

### Data Pre-processing & Repair Operations
* **Anomalous Shift Corrections**: A boolean mask searches the `rating` column for strings containing duration metrics (`"min"`, `"Season"`, `"Seasons"`). Where found, the script shifts those text values into the true `duration` field and fills the empty rating block with an `Unknown` token.
* **Stratified Imputation Blocks**:
  * High-missingness columns (`director`, `cast`, `country`) are labeled with a distinct categorical level (`'Unknown'`) to treat lack of information as an explicit predictive signal.
  * Low-missingness columns are patched cleanly using structural mode replacement based on overall row context.
* **Dual-Track Duration Splits**: The unified `duration` string is parsed, separating numeric metrics into two standalone continuous attributes: `duration_movies` (numerical minutes) and `duration_tv_series` (integer number of seasons).
* **Threshold-Based Binning**: To capture non-linear relationship thresholds (e.g., mini-series vs. long-running series or shorts vs. feature-length movies), both continuous duration vectors are bucketed into explicit, interpretable numerical bins (`duration_movies_bin`, `duration_tv_series_bin`).

### Feature Vectorization & Matrix Transformation
To accommodate multi-modal data (dense text description + high-cardinality metadata tags), features are built using distinct feature generation tracks:

1. **Text Fusion & Semantics (TF-IDF + Truncated SVD)**:
   * String properties (`title`, `director`, `cast`, `description`) are concatenated per row into a single holistic text block.
   * This textual representation is vectorized via a `TfidfVectorizer` to extract word-level and n-gram significance weights across the complete dictionary space.
   * To prevent severe dimensions expansion, **Truncated SVD (Latent Semantic Analysis)** reduces the sparse TF-IDF output into a dense matrix calibrated to capture an **85% explained variance threshold**.
2. **High-Cardinality Categorical Encoding (Binary Encoding)**:
   * To prevent a massive sparse matrix from one-hot encoding variables like `rating` or binned intervals, **Binary Encoding** translates individual category levels into binary code representations, structuring information into logarithmic column lengths.
3. **Sparse/Dense Concatenation**:
   * The text feature matrices and encoded categorical matrices are merged horizontally using `scipy.sparse.hstack` to formulate a consolidated, highly optimized model input matrix $X$.

---

## 🤖 3. Multi-Label Modeling, Evaluation, & Benchmarks

### The Multi-Label Setup
Because a single film or series naturally maps to multiple overlapping genres (e.g., a title can concurrently be *Action*, *Sci-Fi*, and *International Movie*), standard multi-class models are insufficient. The targets are transformed using `MultiLabelBinarizer` across **84 distinct classes**, and modeled using a **OneVsRestClassifier (Binary Relevance)** architecture pattern.

### Custom Metric Implementation
Beyond traditional cross-entropy or strict accuracy, the pipeline measures cross-validation success using a custom **Intersection Accuracy** (Jaccard Index equivalent for multi-label subsets) alongside Micro/Macro F1 metrics:

$$\text{Intersection Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \frac{|Y_{true, i} \cap Y_{pred, i}|}{|Y_{true, i} \cup Y_{pred, i}|}$$

### 🏆 Empirical Performance Matrix
A 2-fold cross-validation routine evaluated multiple machine learning architectures, revealing that linear approaches with balanced penalties significantly outpaced complex gradient boosting structures due to the sparse, high-dimensional target layout:

| Model Architecture | Intersection Accuracy | Micro F1-Score | Macro F1-Score | Accuracy (Exact Match) |
| :--- | :---: | :---: | :---: | :---: |
| 🥇 **Logistic Regression** | **0.804075** | **0.594861** | **0.357239** | 0.097905 |
| 🥈 **Bagged Logistic Regression** | **0.752917** | 0.481626 | 0.347537 | 0.001617 |
| 🥉 **Linear SVC (Support Vector)** | **0.652356** | 0.619891 | 0.350948 | 0.191400 |
| 🏅 **Bagged SVC** | **0.599936** | 0.463836 | 0.296500 | 0.003559 |
| ❌ **LightGBM Classifier** | **0.376384** | 0.510174 | 0.165432 | 0.141101 |

### Key Modeling Insights:
* **The Linear Advantage**: **Logistic Regression** achieved an exceptional average intersection accuracy of **~80.4%**. It drastically out-performed tree models, showing that linear log-odds modeling remains optimal when operating inside high-dimensional spaces produced by TF-IDF and SVD layers.
* **The Micro vs. Macro Gap**: All models recorded a higher Micro F1-score relative to Macro F1-scores. This points to strong classification patterns on highly populated, dominant genres (like *Dramas* or *Comedies*), while rare, long-tail genres remain difficult for the algorithms to reliably pinpoint.

## 🛠️ 4. End-to-End Pipeline

### Core Production Engine: `model_pipeline.py`

The script `model_pipeline.py` serves as the core production machine learning engine for this project, transitioning the experimental findings from the exploratory notebooks into a modular, engineering-ready pipeline using a high-dimensional Logistic Regression model.

#### 📦 Input Data Specification

The script expects a raw dataset file as its entry point, typically named **`train.csv`** (representing an 80% stratified training block) or **`test.csv`** (representing the remaining 20% validation slice). This raw data input contains structured content metadata spanning streaming features such as textual components (`title`, `director`, `cast`, `description`), geographic footprint (`country`), standard category configurations (`platform`, `type`, `rating`), and continuous time markers (`release_year`, `date_added`).

* **Training Phase:** The pipeline ingests the multi-label target column **`listed_in`**, which contains comma-separated genre categories.
* **Inference Phase:** The target column is omitted entirely from the input data file, simulating a pure out-of-sample inference routine.

#### ⚙️ Step-by-Step Processing Execution

When invoked, the pipeline handles feature transformation and inference sequentially through a series of isolated functional blocks:

1. **Structural Data Repair:** The pipeline applies a boolean mask tracking pattern across the dataset to discover text strings that shifted columns during upstream parsing (specifically moving runtime lengths like `"74 min"` from the missing `rating` field back to its true `duration` attribute).
2. **Context-Aware Imputation:** High-missingness string columns (`director`, `cast`, `country`) are explicitly mapped to an `'Unknown'` category level to retain missingness as a predictive signal, while low-missingness attributes are imputed safely using localized statistical means or the conditional mode grouped by content `type`.
3. **Feature Extraction & Duration Splitting:** The consolidated `duration` string is parsed using regular expressions to extract numeric values, which are split into two new continuous variables tracking standalone runtime matrices: `duration_movies` (minutes) and `duration_tv_series` (seasons).
4. **Quantile Transformation Binning:** To capture non-linear threshold patterns, the raw continuous values for runtimes, acquisition years, and release calendars are mapped into discrete categorical buckets based on training sample quartiles.
5. **Multi-Modal Vectorization & Dimensionality Reduction:** High-cardinality categorical attributes undergo structural **Binary Encoding** to minimize dimension expansion. Concurrently, all textual metadata blocks are merged per row, transformed via a word-level and n-gram `TfidfVectorizer`, and compressed using **Truncated SVD (Latent Semantic Analysis)** to capture an optimized 85% explained variance threshold.
6. **Multi-Label Machine Learning Inference:** The numeric sparse-dense matrices are joined and fed to a high-dimensional **Logistic Regression** classifier wrapped inside a parallelized **`OneVsRestClassifier`** (Binary Relevance) framework, which models individual log-odds boundaries independently for all 84 potential genre combinations.

#### 📄 Output Artifact Specification

The operational execution yields a final structured output file saved directly into your workspace as **`prediction.csv`**. This prediction artifact matches the row length and initial identification order of the incoming `test.csv` file. It appends the newly computed predictions into a single, comprehensive string array column named **`listed_in`**, where predicted multi-label target boundaries are cleanly rejoined back into standard comma-separated format (e.g., `Dramas, International Movies, Thrillers`).

#### Run using CLI
> run python model_pipeline.py

```text
| Starting execution of the pipeline...
| Starting data pre-processing...
| Starting data transformation...
| Running Logistic Regression CV...
└─ Fold-wise Results:

   Fold  Micro_F1  Macro_F1  Accuracy  Intersection_Accuracy
0     1  0.586897  0.352070  0.091419               0.791410
1     2  0.591997  0.352709  0.096541               0.785778

| Starting Model Predictions...
| Optimization sequence finished successfully.
| Prediction matrix outputted to: data/output/prediction.csv
| Total E2E Execution Runtime [H:M:S]: 00:07:23
```

---