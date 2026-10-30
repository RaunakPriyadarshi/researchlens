# ResearchLens
### Research Paper Abstract Classification Using Natural Language Processing and Machine Learning

ResearchLens is a machine-learning web application that classifies academic research-paper abstracts into one of three subject areas: **Science**, **Business**, and **Social Science**. It combines a reproducible NLP and model-comparison workflow with a Streamlit interface for single-abstract and batch CSV predictions.

> **Project type:** Supervised, multiclass text classification  
> **Core techniques:** Text preprocessing, TF-IDF, Logistic Regression, model evaluation  
> **Interface:** Streamlit  
> **Model artifact:** `best_model.joblib`

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Problem Statement and Objectives](#2-problem-statement-and-objectives)
- [3. Features](#3-features)
- [4. Dataset](#4-dataset)
- [5. Methodology and Workflow](#5-methodology-and-workflow)
- [6. Model Training and Evaluation](#6-model-training-and-evaluation)
- [7. Application Architecture](#7-application-architecture)
- [8. Technology Stack](#8-technology-stack)
- [9. Repository Structure](#9-repository-structure)
- [10. Installation and Local Setup](#10-installation-and-local-setup)
- [11. Running the Application](#11-running-the-application)
- [12. Input Format and Output](#12-input-format-and-output)
- [13. Deployment on Streamlit Community Cloud](#13-deployment-on-streamlit-community-cloud)
- [14. Limitations and Responsible Use](#14-limitations-and-responsible-use)
- [15. Future Improvements](#15-future-improvements)
- [16. Author / Academic Information](#16-author--academic-information)

---

## 1. Project Overview

Academic databases contain large collections of research papers across many disciplines. Organizing these papers by subject area can be time-consuming when it is done manually.

ResearchLens addresses this task by using the text of a paper's abstract as input and predicting its subject category. The project follows a supervised learning approach: abstracts in the training dataset are paired with known category labels, the text is converted into numerical TF-IDF features, and classification algorithms learn patterns associated with each category.

The trained model is saved as a reusable scikit-learn pipeline. The Streamlit application loads this artifact to classify new abstracts without retraining the model whenever the application is opened.

### Supported subject areas

| Category | Description |
|---|---|
| **Science** | Scientific and technical research |
| **Business** | Business, management, finance, and market-related research |
| **Social Science** | Research concerning society, people, and public policy |

The category descriptions are broad guides. The model predicts from patterns learned in the supplied labeled dataset; it does not independently verify the discipline or quality of a paper.

---

## 2. Problem Statement and Objectives

### Problem statement

Researchers and students may need to organize many academic abstracts into broad subject areas. Manually reviewing and categorizing every abstract can require substantial time and effort. A text-classification system can help by generating an initial category prediction that a user can review.

### Objectives

1. Prepare a labeled dataset of research abstracts from three subject areas.
2. Clean and normalize abstract text using NLP preprocessing.
3. Convert text into numerical features using TF-IDF.
4. Train and compare multiple supervised machine-learning classifiers.
5. Evaluate model performance using classification metrics and cross-validation.
6. Save the selected model as a reusable artifact.
7. Develop a web interface for individual and batch predictions.
8. Allow users to download batch prediction results as a CSV file.

---

## 3. Features

### 3.1 Single Abstract Classification

- Accepts a research abstract through a text input area.
- Applies the project's text-cleaning procedure.
- Predicts one of the three subject categories.
- Displays the predicted category.
- Displays class probability estimates when supported by the saved classifier.

### 3.2 Batch CSV Classification

- Accepts a CSV file containing an `Abstract` or `text` column (column matching is case-insensitive).
- Processes multiple abstracts in one upload.
- Retains the original input columns.
- Adds a `predicted_label` column.
- Adds a `confidence` column when the classifier supports `predict_proba`.
- Displays a category-wise prediction count.
- Provides a downloadable CSV containing all input rows and generated predictions.

### 3.3 Model Insights

The application provides an overview of the classification task, the TF-IDF and Logistic Regression approach, and the recorded model evaluation results.

### 3.4 User Interface

The application is built with Streamlit and includes a sidebar for navigating between the three workspaces. Custom CSS is used to provide a consistent visual layout.

---

## 4. Dataset

### 4.1 Data Sources

The training notebook reads three separate Scopus-exported CSV files:

| Input file | Assigned label |
|---|---|
| `scopus_science.csv` | Science |
| `scopus_business.csv` | Business |
| `scopus_social.csv` | Social Science |

The notebook reads available metadata fields including `Title`, `Year`, `Abstract`, `Document Type`, and `EID`, and requires an `Abstract` column. The classification input is the abstract text; the category label is assigned from the source file.

### 4.2 Dataset Preparation

The notebook performs the following preparation steps:

1. Loads the three source CSV files.
2. Assigns a category label to each source dataset.
3. Removes rows with missing abstracts.
4. Removes records containing `[No abstract available]`.
5. Removes copyright text where detected.
6. Retains abstracts containing at least 30 words after the initial cleaning step.
7. Removes duplicate abstracts within a class.
8. Removes identical abstracts that appear across different classes.
9. Samples up to 700 abstracts from each category using a fixed random seed.

### 4.3 Dataset Summary

In the recorded notebook execution, the source files each contained 1,000 rows. After cleaning and filtering, the notebook retained enough records to create a balanced dataset of **2,100 abstracts**:

| Category | Selected abstracts |
|---|---:|
| Science | 700 |
| Business | 700 |
| Social Science | 700 |
| **Total** | **2,100** |

The notebook exports the prepared dataset as `labeled_dataset.csv`, with the `abstract` and `label` columns.

**Note:** These counts describe the recorded notebook run. Re-running the notebook with different source CSV contents may produce different counts.

---

## 5. Methodology and Workflow

The project uses a text-classification pipeline with the following stages.

### 5.1 Text Cleaning and Preprocessing

The notebook's `clean_text()` function applies these operations:

1. Converts text to lowercase.
2. Replaces characters other than English letters and whitespace with spaces.
3. Splits text into tokens.
4. Removes tokens of two characters or fewer.
5. Removes English stopwords and an additional project-specific stopword list.
6. Applies NLTK's `WordNetLemmatizer`.
7. Joins the remaining tokens into cleaned text.

The notebook also removes copyright-related text during dataset preparation. The web application applies the corresponding copyright stripping and text-cleaning steps to new inputs.

### 5.2 TF-IDF Feature Extraction

The notebook uses `TfidfVectorizer` with the following configuration:

| Parameter | Value | Purpose |
|---|---:|---|
| `max_features` | 5000 | Limits the vocabulary to at most 5,000 features |
| `ngram_range` | `(1, 2)` | Uses unigrams and bigrams |
| `min_df` | 3 | Ignores terms appearing in fewer than three documents |
| `sublinear_tf` | `True` | Applies sublinear term-frequency scaling |

**TF-IDF (Term Frequency–Inverse Document Frequency)** represents text numerically. It gives weight to terms based on their frequency in a document and their distribution across the document collection. Unigrams represent individual words; bigrams represent two-word sequences.

### 5.3 Train-Test Split

The dataset is divided using a stratified train-test split:

| Setting | Value |
|---|---:|
| Training proportion | 80% |
| Testing proportion | 20% |
| Random state | 42 |
| Stratification | By category label |

For the recorded 2,100-row dataset, this gives 1,680 training examples and 420 test examples, with 140 test examples per category.

Stratification preserves the approximate class distribution in both subsets.

### 5.4 Classification Pipeline

Each candidate classifier is combined with TF-IDF using a scikit-learn `Pipeline`:

1. TF-IDF transforms cleaned abstracts into feature vectors.
2. The classifier uses those feature vectors to predict a category.

This ensures that the same fitted feature transformation is applied when predicting unseen abstracts.

---

## 6. Model Training and Evaluation

### 6.1 Algorithms Compared

The notebook trains and evaluates six classifiers:

1. Logistic Regression
2. Decision Tree
3. Multinomial Naive Bayes
4. k-Nearest Neighbors (kNN)
5. Support Vector Machine (SVM) with a linear kernel
6. Random Forest

### 6.2 Evaluation Metrics

The notebook calculates:

- **Accuracy:** proportion of all test examples classified correctly.
- **Precision:** proportion of predicted examples for a class that belong to that class.
- **Recall:** proportion of actual examples of a class that are correctly identified.
- **F1-score:** harmonic mean of precision and recall.
- **Macro average:** calculates a metric independently for each class and gives each class equal weight.

It also produces model-comparison charts, confusion matrices, a per-class classification report, and five-fold stratified cross-validation results.

### 6.3 Recorded Holdout Test Results

The following results are from the output saved in the supplied notebook. Precision, recall, and F-measure are macro-averaged.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| **Logistic Regression** | **0.8357** | **0.8375** | **0.8357** | **0.8353** |
| SVM | 0.8333 | 0.8354 | 0.8333 | 0.8325 |
| Multinomial Naive Bayes | 0.8286 | 0.8283 | 0.8286 | 0.8280 |
| kNN | 0.8095 | 0.8127 | 0.8095 | 0.8105 |
| Random Forest | 0.8095 | 0.8079 | 0.8095 | 0.8079 |
| Decision Tree | 0.6762 | 0.6752 | 0.6762 | 0.6754 |

Logistic Regression achieved the highest macro F1-score in the recorded holdout test comparison (0.8353). It was therefore selected by the notebook's holdout-results ranking for the final saved model.

### 6.4 Five-Fold Cross-Validation

The notebook additionally uses `StratifiedKFold` with five folds and macro F1 scoring.

| Model | Mean Macro F1 | Standard Deviation |
|---|---:|---:|
| Logistic Regression | 0.8544 | 0.0157 |
| Decision Tree | 0.7179 | 0.0148 |
| Multinomial Naive Bayes | 0.8557 | 0.0114 |
| kNN | 0.8474 | 0.0124 |
| SVM | 0.8505 | 0.0180 |
| Random Forest | 0.8352 | 0.0163 |

**Interpretation:** Multinomial Naive Bayes has the highest mean macro F1 in the recorded five-fold cross-validation table (0.8557), while Logistic Regression was selected by the separate holdout test-set ranking. These are different evaluation procedures, and the results should not be presented as if they were the same comparison.

### 6.5 Selected Model: Logistic Regression

The recorded holdout test classification report for Logistic Regression is:

| Category | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Business | 0.87 | 0.78 | 0.82 | 140 |
| Science | 0.85 | 0.90 | 0.88 | 140 |
| Social Science | 0.79 | 0.83 | 0.81 | 140 |
| **Macro average** | **0.84** | **0.84** | **0.84** | **420** |

### 6.6 Model Interpretation

The notebook includes feature visualizations:

- Top Logistic Regression features for each category, based on learned class coefficients.
- Top Random Forest features, based on feature importance.

These visualizations help inspect which terms or phrases influence the fitted models. They do not, by themselves, establish causal relationships.

### 6.7 Saving the Model

The notebook fits the selected classifier in a final TF-IDF-plus-classifier pipeline using the prepared labeled dataset and saves it using Joblib as:

`best_model.joblib`

The saved pipeline contains the fitted TF-IDF vectorizer and classifier. This is the model artifact loaded by the Streamlit application.

---

## 7. Application Architecture

The project has two main components:

### Training component — `Individual_Project.ipynb`

- Reads and labels the source data.
- Prepares the balanced dataset.
- Cleans the text.
- Trains and evaluates six classifiers.
- Selects the model based on the holdout macro F1 ranking.
- Saves the final pipeline as `best_model.joblib`.

### Inference component — `app.py`

- Loads the saved model using Joblib.
- Initializes NLTK stopwords and lemmatization resources.
- Accepts one abstract or a CSV upload.
- Applies text preprocessing.
- Calls the saved pipeline's `predict()` method.
- Displays predictions and probability estimates when supported.
- Allows users to download batch results.

### High-level flow

```text
Scopus CSV files
       |
       v
Dataset preparation and labeling
       |
       v
Text cleaning and preprocessing
       |
       v
Train-test split
       |
       v
TF-IDF feature extraction + model comparison
       |
       v
Selected Logistic Regression pipeline
       |
       v
best_model.joblib
       |
       v
Streamlit application (app.py)
       |
       +-----------------------+
       |                       |
       v                       v
Single abstract           Batch CSV
       |                       |
       v                       v
Predicted category        Predicted labels
and probabilities         and downloadable CSV
```

---

## 8. Technology Stack

| Technology / Library | Role in the project |
|---|---|
| Python | Main programming language |
| Pandas | Dataset loading, cleaning, and CSV processing |
| NumPy | Numerical operations |
| Regular Expressions (`re`) | Text normalization |
| NLTK | Stopword removal and lemmatization |
| Scikit-learn | TF-IDF, classifiers, pipelines, splitting, and evaluation |
| Joblib | Model serialization and loading |
| Matplotlib | Visualization |
| Seaborn | Statistical plots and confusion-matrix presentation |
| Streamlit | Interactive web application |
| GitHub | Source-code repository and version control |
| Streamlit Community Cloud | Web application hosting |

---

## 9. Repository Structure

The deployed application repository should contain:

```text
researchlens/
│
├── app.py                    # Streamlit application
├── best_model.joblib         # Trained TF-IDF + classifier pipeline
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .gitignore                # Files excluded from version control
```

The model-training notebook and source Scopus CSV files may be maintained separately or added to the repository when their redistribution is permitted and repository size allows it.

The notebook used for model development is `Individual_Project.ipynb` in the project materials. The uploaded notebook supplied for this documentation is named `classification.ipynb`.

---

## 10. Installation and Local Setup

### Prerequisites

- Python 3.10 or 3.11 is recommended as a starting point; use a version compatible with the installed scikit-learn and other dependencies.
- pip
- Git (optional for cloning the repository)

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/researchlens.git
cd researchlens
```

Replace `YOUR_GITHUB_USERNAME` with the GitHub account that owns the repository.

Alternatively, download the repository ZIP and extract it.

### Step 2: Create a virtual environment

**Windows PowerShell:**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Check the model file

Ensure `best_model.joblib` is in the same directory as `app.py`.

### Step 5: Start the application

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal, usually:

`http://localhost:8501`

---

## 11. Running the Application

### Single abstract

1. Open ResearchLens.
2. Select **Single abstract** from the sidebar.
3. Paste a research abstract into the input area.
4. Select **Classify abstract**.
5. Review the predicted category and class probability breakdown, if available.

### Batch CSV

1. Select **Batch CSV** from the sidebar.
2. Download the sample CSV if you need a format reference.
3. Prepare a CSV containing an `Abstract` or `text` column.
4. Upload the CSV.
5. Review the prediction counts and result preview.
6. Download the complete prediction CSV.

### Model insights

Select **Model insights** to view the classification approach and recorded evaluation information.

---

## 12. Input Format and Output

### CSV input example

```csv
Title,Abstract
Example Paper,This study examines customer satisfaction and purchasing behavior in online markets.
Another Paper,This research investigates the effects of environmental changes on biological systems.
```

The application accepts a column named `Abstract` or `text`, regardless of capitalization. Other columns are retained in the output.

### Output columns

| Column | Description |
|---|---|
| Original columns | Input metadata and abstract text are retained |
| `predicted_label` | Predicted category: Science, Business, or Social Science |
| `confidence` | Highest class probability when `predict_proba` is supported |

The application is a prediction workflow. It does not calculate accuracy from user-supplied `true_label` values. Reliable evaluation requires independently verified ground-truth labels and a separate evaluation procedure.

---

## 13. Deployment on Streamlit Community Cloud

ResearchLens can be deployed using Streamlit Community Cloud.

1. Push the application files to a GitHub repository.
2. Confirm that `app.py`, `requirements.txt`, and `best_model.joblib` are available at the expected repository paths.
3. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
4. Sign in using GitHub and authorize repository access.
5. Select **Create app** / **Deploy a public app from GitHub**.
6. Select the `researchlens` repository.
7. Select the branch that contains the application (`main` or `master`, as applicable).
8. Set the main file path to `app.py`.
9. Choose an available Streamlit app subdomain.
10. Deploy and inspect the build logs if an error occurs.

After deployment, Streamlit provides a public application URL ending in `.streamlit.app`.

### Deployment requirements

- The model file must be accessible at the path expected by `app.py`.
- `requirements.txt` must list the application's runtime dependencies.
- The Python and scikit-learn versions must be compatible with the saved Joblib model.
- NLTK's `stopwords`, `wordnet`, and `omw-1.4` resources must be available. The application attempts to download them during initialization.
- Do not commit passwords, API keys, private data, or credentials to a public repository.

If `best_model.joblib` is too large for a normal GitHub upload, use an appropriate model-artifact storage solution or Git LFS and update the application's model-loading logic accordingly.

---

## 14. Limitations and Responsible Use

1. **Three-category scope:** The model predicts only Science, Business, or Social Science. It is not designed to identify every academic discipline.
2. **Dataset dependence:** Predictions depend on the quality, representativeness, and labeling of the training abstracts.
3. **Language and text limitations:** The preprocessing retains English alphabetic characters and is designed for English-language abstracts. It may not work well for other languages, mathematical notation, or specialized symbols.
4. **Prediction uncertainty:** A model probability is not a guarantee of correctness and should not be interpreted as the model's overall accuracy.
5. **No paper-quality assessment:** ResearchLens classifies abstract text. It does not evaluate research validity, originality, credibility, methodology quality, or publication quality.
6. **No independent domain detector:** The current application does not independently determine whether an abstract is in-domain or out-of-domain.
7. **Preprocessing consistency:** Training and inference must use matching text-cleaning rules. Changes to preprocessing may affect predictions.
8. **Evaluation scope:** The reported metrics are from the recorded notebook execution. They are not guarantees of performance on future or external datasets.

ResearchLens should be treated as an assistive categorization tool. Important academic or research decisions should not rely solely on its predictions.

---

## 15. Future Improvements

Potential extensions include:

- Add more academic subject categories.
- Evaluate the model on a larger, independently sourced external dataset.
- Tune classifier hyperparameters systematically.
- Calibrate predicted probabilities and report calibration metrics.
- Add more detailed error analysis for misclassified abstracts.
- Add model versioning and reproducible experiment tracking.
- Improve handling of multilingual text and domain-specific terminology.
- Add semantic search and similar-paper recommendations.
- Provide downloadable evaluation reports for verified labeled datasets.

These are possible future directions and are not represented as features already implemented.

---

## 16. Author / Academic Information

**Project:** ResearchLens — Research Paper Abstract Classification  
**Author:** Raunak Priyadarshi Yadav  
**Academic Programme:** M.Tech in Data Science  
**Institution:** Indian Institute of Technology Roorkee

---

## Acknowledgements and References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [NLTK Documentation](https://www.nltk.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Joblib Documentation](https://joblib.readthedocs.io/)

---

*ResearchLens — Turning Research Abstracts into Organized Insights.*
