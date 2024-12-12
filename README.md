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
iowa_dream/
├── config/
│   ├── config.yaml            # Centralized configuration file
│   ├── validation.py          # Pydantic validation of configuration
├── data/                      # Data handling and loading modules
│   ├── data_loader.py         # Data import and loading
│   ├── data_splitter.py       # Train/test splitting
├── preprocessing/             # Data preprocessing modules
│   ├── data_cleaner.py        # Handles missing values, transformations
│   ├── encoder.py             # Encoding categorical variables
├── feature_engineering/       # Feature engineering modules
│   ├── feature_generator.py   # Creation of new features
│   ├── feature_selector.py    # Feature selection logic
├── analysis/                  # Model training and evaluation
│   ├── model_training.py      # Training pipeline for models
│   ├── evaluation.py          # Model evaluation and visualizations
├── notebooks/                 # Jupyter notebooks for EDA and analysis
│   ├── eda_report.ipynb       # Exploratory Data Analysis notebook
├── tests/                     # Unit tests for all modules
│   ├── test_data_loader.py    # Tests for data loading
│   ├── test_preprocessing.py  # Tests for preprocessing
├── .gitignore                 # Git ignore rules
├── environment.yml            # Conda environment configuration
├── pre-commit-config.yaml     # Pre-commit hooks configuration
├── flake8                     # Linting configuration
├── setup.cfg                  # Packaging configuration
├── pyproject.toml             # Build system metadata
├── LICENSE                    # Licensing information
├── README.md                  # Project documentation
```

---

## **Installation**

### **1. Clone the Repository**  
```bash
git clone https://github.com/congminh-nguyen/ames_project.git
cd iowa_dream
```

### **2. Set Up the Environment**  
Create and activate the Conda environment:  
```bash
conda env create -f environment.yml
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

1. create a Kaggle account: https://www.kaggle.com/
2. get API keys and user name
   - Go to your account, and click on your profile in the top right
   - Then click on "Settings"
   - Scroll down to the API section and click on "Create New Token"
   - Get your username and API key from there

We have written a data loader function for you in the "iowa_dream/data/data_loader.py".This allows you to
download the data with by running the script from the terminal. Run the following command in the
terminal being at the root of the repository.

```bash
python nba/data_loader.py -u "your_user_name" -k "your_api_key" -d "shashanknecrothapa/ames-housing-dataset"
```

Replace "your_user_name" and "your_api_key" with your username and API key. This creates a json
file at "~/.kaggle/kaggle.json" with your username and API key, which is used to authenticate your
account and download the data.

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
