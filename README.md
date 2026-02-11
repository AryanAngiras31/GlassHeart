# GlassHeart

This project implements **GlassHeart**, a Clinical Decision Support System (CDSS) that predicts the likelihood of heart failure mortality based on clinical records. Beyond standard modeling, it features an interactive dashboard that allows clinicians to assess patient risk in real-time, visualize feature importance using SHAP, and validate the model's rigorous statistical foundations. This is deployed at [GlassHeart](https://glassheart.streamlit.app/).

## Table of Contents

- [Features](#-features)
- [Dataset](#-dataset)
- [Project Structure](#️-project-structure)
- [Installation](#️-installation)
- [Model Development](#-model-development)
- [Results](#-results)
- [License](#-license)

## Features

- **Multiple Model Support**: Implements and compares Logistic Regression, Random Forest, and XGBoost classifiers
- **Advanced Calibration**: Implements Platt Scaling and Isotonic Regression for probability calibration
- **Feature Engineering**: Includes feature selection and transformation techniques
- **Comprehensive Evaluation**: Multiple metrics including ROC AUC, PR AUC, and Brier Score
- **Optimal Threshold Selection**: Implements Youden's J statistic for optimal decision thresholding
- **Statistical Testing**: Includes likelihood ratio tests and DeLong's test for model comparison

## Dataset

The project uses the UCI Heart Failure dataset containing patient records with the following features:

### Input Features:

- **Demographics**:
  - Age
  - Sex
- **Medical History**:
  - Anaemia
  - Diabetes
  - High Blood Pressure
  - Smoking Status
- **Lab Results**:
  - Creatinine Phosphokinase (CPK) levels
  - Ejection Fraction
  - Platelets
  - Serum Creatinine
  - Serum Sodium
- **Clinical Status**:
  - Time to follow-up
  - Death Event (target variable: `0 = survived`, `1 = death`)

### Data Preprocessing:

- Handling missing values
- Feature scaling and normalization
- Outlier detection and treatment
- Feature selection based on statistical significance

## Project Structure

```
GlassHeart/
├── data/                                   # Data file
│   └── heart_failure_clinical_records.csv
├── notebooks/                              # Jupyter notebooks
│   ├── EDA_and_Preprocessing.ipynb         # Exploratory Data Analysis
│   ├── Model_Building_and_Evaluation.ipynb # Model development
│   └── utils.py                            # Utility functions
├── artifacts/                              # Saved pipelines and test splits
├── models/                                 # Calibrated model binaries (.joblib)
├── reports/                                # Generated analysis
│   ├── figures/                            # Saved plots (SHAP, Calibration curves)
│   └── tables/                             # Statistical test results
├── app.py                                  # Streamlit Dashboard application
├── train_models.py                         # Script to retrain and calibrate models
├── docker-compose.yml                      # Dashboard orchestration
├── Dockerfile                              # Docker configuration for notebooks
├── requirements.txt                        # Project dependencies
└── README.md                               # Project documentation
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/AryanAngiras31/Heart-Failure-Prediction-Model.git
   cd Heart-Failure-Prediction-Model
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Using Docker

1. **Build the Docker image**:

   ```bash
   docker build -t heart-failure-prediction .
   ```

2. **Run the Docker container**:

   ```bash
   docker run -p 8888:8888 -v "$(pwd):/home/jovyan/work" heart-failure-prediction
   ```

   This will:

   - Map port 8888 from the container to your host machine
   - Mount your current directory to the container's working directory
   - Start Jupyter Notebook with no password or token required

3. **Access Jupyter Notebook**:
   Open your web browser and navigate to:

   ```
   http://localhost:8888
   ```

   You should now be able to access and run the notebooks from your browser.

4. **Stopping the container**:
   Press `Ctrl+C` in the terminal where the container is running, or use:
   ```bash
   docker stop $(docker ps -q --filter ancestor=heart-failure-prediction)
   ```

## Model Development

The model development process is structured as follows:

1.  **Feature Set Engineering**: Feature sets are constructed incrementally to test specific clinical hypotheses:

    *   **Primary Hypothesis Validation**: To test if lower ejection fraction and higher serum creatinine are dominant predictors, the following sets are used:
        *   `baseline_fs`: Includes age, sex, creatinine phosphokinase, and platelets.
        *   `ef_features`: Baseline + Ejection Fraction.
        *   `creatinine_features`: Baseline + Serum Creatinine.
        *   `ef_creatinine_features`: Baseline + Ejection Fraction + Serum Creatinine.

    *   **Secondary Hypothesis Validation**: To test if serum sodium and age nonlinearity add value, the following sets are used:
        *   `ef_creatinine_sodium_features`: Adds Serum Sodium.
        *   `ef_creatinine_age_squared_features`: Adds Age Squared.
        *   `ef_creatinine_sodium_age_squared_features`: Adds both Serum Sodium and Age Squared.

    By comparing model performance across these sets, the clinical importance of each feature is systematically evaluated.

2.  **Model Training**: Three models are trained on each feature set:
    *   **Logistic Regression**: A linear model with L2 regularization.
    *   **Random Forest**: An ensemble of 100 decision trees.
    *   **XGBoost**: A gradient boosting framework.

3.  **Model Calibration**: To ensure the predicted probabilities are reliable, the models are calibrated:
    *   **Platt Scaling**: Applied to the Logistic Regression model as traditionally prescribed. It is also applied to Random Forest and XGBoost models since Isotonic calibration would cause them to overfit on this small dataset

4.  **Evaluation**: The models are evaluated using a comprehensive set of metrics:
    *   **PR AUC**: The primary metric for this imbalanced dataset.
    *   **ROC AUC**: To assess the model's ability to distinguish between classes.
    *   **Brier Score**: To measure the accuracy of probability predictions.
    *   **Calibration Plots**: To visually inspect the calibration of the models.

5.  **Statistical Significance**: Likelihood Ratio test is performed to check if the expanded model significantly outperforms the baseline model. DeLong's test is used to determine if the differences in ROC AUC scores between the models are statistically significant.

## Results

The project partially validates the hypothesis by confirming that models built upon Ejection Fraction and Serum Creatinine are the most stable and discriminative. However, the models fail to conclusively prove the utility of the hypothesized incremental variables (serum_sodium and age_nonlinearity).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
