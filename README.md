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
├── .pre-commit-config.yaml     # Pre-commit hooks configuration
├── .flake8                     # Linting configuration
├── LICENSE                    # Licensing information
├── README.md                  # Project documentation
├── config.yaml                # Centralized configuration file
├── data_dict.md               # Data dictionary
├── environment.yaml           # Conda environment configuration
├── iowa_dream/                # Main project folder
│   ├── __init__.py
│   ├── __pycache__/           # Cached Python files
│   ├── _cleanfile/            # Processed/cleaned data files
│   │   └── cleaned_AmesHousing.parquet
│   ├── _rawfile/              # Raw data files
│   │   └── AmesHousing.csv
│   ├── data/                  # Data handling modules
│   │   ├── __init__.py
│   │   ├── cleaner.py         # Data cleaning scripts
│   │   ├── importer.py        # Data importing scripts
│   │   └── loader.py          # Data loading scripts
│   ├── evaluation/            # Model evaluation modules
│   │   ├── __init__.py
│   │   └── metrics_plot.py    # Evaluation and plotting scripts
│   ├── feature_engineering/   # Feature engineering modules
│   │   ├── __init__.py
│   │   ├── add_drop_features.py
│   │   ├── categotical_transformer.py
│   │   ├── lot_frontage_imputer.py
│   │   └── numerical_transformer.py
│   ├── integration_scripts/   # Scripts to integrate various components
│   │   ├── evaluation.py
│   │   ├── glm_pipeline.py
│   │   ├── lgbm_pipeline.py
│   │   └── load_and_clean.py
│   ├── models/                # Model training and optimization
│   │   ├── __init__.py
│   │   ├── custom_obj_lgbm.py # Custom LightGBM objective
│   │   └── optuna_objective.py # Optuna optimization objective
│   ├── utils/                 # Utility scripts
│   │   ├── describer.py       # Data describing scripts
│   │   ├── inconsistency_check.py
│   │   ├── plotting_EDA.py    # Exploratory Data Analysis plotting
│   │   └── sample_split.py    # Data splitting utilities
│   └── version.py             # Version information
├── notebooks/                 # Jupyter notebooks
│   ├── eda_cleaning.ipynb     # EDA and data cleaning
│   ├── engineering_explorer.ipynb
│   ├── evaluation.ipynb       # Model evaluation
│   └── model.ipynb            # Model training notebook
├── pyproject.toml             # Build system metadata
└── tests/                     # Test cases
    ├── test_add_drop_features.py
    ├── test_categorical_transformer.py
    ├── test_lot_frontage_imputer.py
    └── test_numeric_transformer.py

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
To run the the integrated scripts, simply add interactive windows in VSCODE and choose the kernel to iowa_dream. 
Another note is that for interactive plots in notebooks for EDA, type in to get the feature types, and type 'exit' to quit the interactive window. 

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

### **Run the pipeline end-to-end:**

1. **Navigate to the Project Directory**:
   Ensure you are in the root directory of the project:
   ```bash
   cd /ames_project
   ```

2. **Activate the Conda Environment**:
   Make sure the Conda environment is activated:
   ```bash
   conda activate iowa_dream
   ```

3. **Run the Integration Scripts**:
   Execute the following scripts in sequence to run the entire pipeline:

   - **Load and Clean Data**:
   
     ```bash
     python iowa_dream/integration_scripts/load_and_clean.py
     ```

   - **Run GLM Pipeline**:
     ```bash
     python iowa_dream/integration_scripts/glm_pipeline.py
     ```

   - **Run LGBM Pipeline**:
     ```bash
     python iowa_dream/integration_scripts/lgbm_pipeline.py
     ```

   - **Evaluate Models**:
     ```bash
     python iowa_dream/integration_scripts/evaluation.py
     ```

This will execute the full data science pipeline, from data loading and cleaning to model training and evaluation.

---

## **Configuration**

All settings are centralized in `config/config.yaml` to ensure reproducibility and ease of experimentation.

This saves models' specifications, features types and dictionaries

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

**Example Outputs**
KEY Pipeline Scripts and Notebooks
The pipeline scripts are located in the `integration_scripts` directory. These scripts include:

- `load_and_clean.py`: For loading and cleaning data.
- `glm_pipeline.py`: For running the GLM pipeline.
- `lgbm_pipeline.py`: For running the LGBM pipeline.
- `evaluation.py`: For evaluating models.

The complementary notebooks for analysis are located in the `notebooks` directory. These notebooks include:

- `eda_cleaning.ipynb`: For EDA and data cleaning.
- `engineering_explorer.ipynb`: For exploring feature engineering.
- `evaluation.ipynb`: For model evaluation.
---

## **Contributors**

- **Blind Grading Number (BGN): 3387G**

---

## **License**

This project is licensed under the MIT License. See `LICENSE` for more details.

---
