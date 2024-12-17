---

# Data Science Project: **IOWA Dream 🌴 - Ames Housing Price Prediction**

A modular, production-ready data science pipeline for analyzing and modeling the **Ames Housing Dataset**. This project follows a modular design with a clear separation of data loading, preprocessing, feature engineering, model training, and evaluation steps.

---

## **Table of Contents**

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Getting the Data](#getting-the-data)
5. [Pipeline Workflow](#pipeline-workflow)
6. [Usage](#usage)
7. [Configuration](#configuration)
8. [Development Tools](#development-tools)
9. [Example Outputs](#example-outputs)
10. [Future Work](#future-work)
11. [Contributors](#contributors)
12. [License](#license)

---

## **Overview**

The **IOWA Dream** project is a comprehensive implementation of a data science workflow designed to streamline housing price prediction for the Ames dataset. It includes:

- **Data Loading:** Automated import of data from Kaggle or local sources.
- **Preprocessing:** Consistent splitting, cleaning, and data preparation.
- **Feature Engineering:** Automated generation, transformation, and selection of key features.
- **Model Training:** Hyperparameter tuning and training of predictive models like GLM and LightGBM.
- **Evaluation:** Interpretable metrics and visualizations for in-depth analysis of model performance.

---

## **Repository Structure**

```
ames_project/
├── .gitignore                 # Git ignore rules
├── environment.yml            # Conda environment configuration
├── .pre-commit-config.yaml     # Pre-commit hooks configuration
├── .flake8                     # Linting configuration
├── .setup.cfg                  # Packaging configuration
├── pyproject.toml             # Build system metadata
├── LICENSE                    # Licensing information
├── README.md                  # Project documentation
├── iowa_dream/
│   ├── config/
│   │   ├── config.yaml            # Centralized configuration file
│   │   └── validation.py          # Pydantic validation of configuration
│   ├── data/
│   │   ├── data_loader.py         # Data import and loading
│   │   └── data_splitter.py       # Train/test splitting
│   ├── preprocessing/
│   │   ├── data_cleaner.py        # Handles missing values, transformations
│   │   └── encoder.py             # Encoding categorical variables
│   ├── feature_engineering/
│   │   ├── feature_generator.py   # Creation of new features
│   │   └── feature_selector.py    # Feature selection logic
│   ├── analysis/
│   │   ├── model_training.py      # Training pipeline for models
│   │   └── evaluation.py          # Model evaluation and visualizations
└── notebooks/
│   └── eda_report.ipynb       # Exploratory Data Analysis notebook
└── tests/
    ├── test_data_loader.py    # Tests for data loading
    └── test_preprocessing.py  # Tests for preprocessing
```

---

## **Installation**

### **1. Clone the Repository**
```bash
git clone https://github.com/congminh-nguyen/ames_project.git
cd ames_project
```

### **2. Set Up the Environment**
Create and activate the Conda environment:
```bash
conda env create -f environment.yaml
conda activate iowa_dream
```

### **3. Install Pre-commit Hooks (Optional)**
Install pre-commit hooks for consistent formatting:
```bash
pre-commit install
pre-commit run --all-files
```

---

# Getting the data

Follow the instructions below to set up and download the dataset using the provided `data_loader.py` script.

#### **Step 1: Set Up Kaggle Credentials**
To authenticate with Kaggle, you need an API key:

1. Log in to your Kaggle account at [https://www.kaggle.com](https://www.kaggle.com).
2. Go to your profile and click **Settings**.
3. Scroll down to the **API** section and click **Create New API Token**.
4. A file named `kaggle.json` will be downloaded. This file contains your **username** and **key**.

#### **Step 2: Place Credentials**
You have three options for providing Kaggle credentials:

1. **Use the `kaggle.json` File (Recommended)**:
   - Place the `kaggle.json` file in the default location: `~/.kaggle/kaggle.json`.

2. **Set Environment Variables**:
   - Export your credentials using:
     ```bash
     export KAGGLE_USERNAME="your_username"
     export KAGGLE_KEY="your_api_key"
     ```

3. **Provide Credentials via Command-Line Arguments**:
   - Pass your Kaggle username and API key directly when running the script (see Step 4).

---

#### **Step 3: Install Requirements**
Ensure you have the required Python libraries installed:

1. Create and activate your Python environment using the provided `environment.yml`:
   ```bash
   conda env create -f environment.yml
   conda activate your_env_name
   ```

2. Install dependencies manually (if needed):
   ```bash
   pip install kaggle pyyaml pydantic
   ```

---

#### **Step 4: Run the Script**

Use the following commands to download the dataset:

1. **Using Default Configuration**:
   If you’ve placed `kaggle.json` in the default location:
   ```bash
   python iowa_dream/data/data_loader.py
   ```

2. **Specifying Credentials via Command-Line**:
   If you prefer not to use `kaggle.json`:
   ```bash
   python iowa_dream/data/data_loader.py -u "your_username" -k "your_api_key" -d "shashanknecrothapa/ames-housing-dataset"
   ```

3. **Customizing the Download Path**:
   To save the dataset to a specific directory:
   ```bash
   python iowa_dream/data/data_loader.py --download-path "./custom_rawfile"
   ```

---

#### **Step 5: Verify the Download**
The dataset will be downloaded and extracted into the `_rawfile` directory by default (or your specified directory).

Example structure after download:
```
iowa_dream/
├── _rawfile/
│   ├── AmesHousing.csv
```

### **(Step 6:) You can directly get the cleaned data from the `_rawfile` directory or run the `load_and_clean.py` script to get the cleaned data.**

```bash
python iowa_dream/integration_scripts/load_and_clean.py
```

---

### **Additional Notes**

- The default dataset is **Ames Housing Dataset** (`shashanknecrothapa/ames-housing-dataset`). You can override this using the `-d` or `--dataset` argument.
- Ensure your credentials are kept secure. Avoid exposing `kaggle.json` or API keys in public repositories.
- For help, run:
  ```bash
  python iowa_dream/data/data_loader.py --help
  ```

---

## **Pipeline Workflow**

1. **Data Loading**
   - Automated loading from local sources or Kaggle.

2. **Exploratory Data Analysis (EDA)**
   - Perform data visualization and summarize key statistics using Jupyter notebooks.

3. **Data Cleaning**
   - Handle missing values, outliers, and perform transformations.

4. **Feature Engineering**
   - Generate domain-specific features, encode categorical variables, and scale numerical ones.

5. **Model Training**
   - Train GLM and LightGBM models with hyperparameter optimization.

6. **Evaluation and Interpretation**
   - Assess models using metrics
   - Visualize feature importance and residuals.

---

## **Usage**

### Run the pipeline end-to-end:
```bash
python main.py --config config/config.yaml
```

---

## **Configuration**

All settings are centralized in `config/config.yaml` to ensure reproducibility and ease of experimentation.

#### Example `config.yaml`:
```yaml
data:
  raw_path: "data/raw/train.csv"
  processed_path: "data/processed/cleaned_data.parquet"
  features_columns:
    - "LotArea"
    - "OverallQual"
    - "YearBuilt"
    - "GrLivArea"
    - "GarageCars"
    - "TotalBsmtSF"
  target_column: "SalePrice"

split:
  primary_key: "Id"
  test_ratio: 0.2

model:
  glm:
    params:
      alpha: 1.0
  lgbm:
    params:
      learning_rate: 0.01
      n_estimators: 100
      max_depth: 6
      min_child_weight: 0.1
```

---

## **Development Tools**

### **Testing**
Run all unit tests with `pytest`:
```bash
pytest tests/
```

### **Code Quality**
Format and check code style using `black` and `isort`:
```bash
black .
isort .
```

---

## **Example Outputs**

- Visualizations: Feature importance graphs, residual plots.
- Metrics: RMSE, MAE, and R² on test data.
- Reports: Automated model performance summaries.

---

## **Future Work**

- **Incorporate deep learning models** for non-linear feature interactions.
- **Extend feature engineering** with interaction terms and polynomial features.
- **Deploy as an API** using FastAPI or Flask.
- **Integrate CI/CD pipelines** for automated testing and deployment.

---

## **Contributors**

- **Blind Grading Number (BGN): 3387G**

---

## **License**

This project is licensed under the MIT License. See `LICENSE` for more details.

---
