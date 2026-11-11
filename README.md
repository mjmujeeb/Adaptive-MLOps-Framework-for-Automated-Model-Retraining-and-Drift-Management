Adaptive MLOps Framework for Automated Model Retraining and Drift Management

A notebook-based MLOps project exploring how machine-learning models can be monitored for changes in incoming data and retrained when their predictive performance begins to degrade.

The project demonstrates a model lifecycle workflow using activity data, regression models, statistical drift checks, and before-and-after performance evaluation.

Overview

A machine-learning model is trained using a particular dataset and learns patterns from that data. Over time, new data may differ from the original training distribution.

This change, commonly referred to as data drift, can affect model reliability.

This project explores a practical workflow for identifying such changes and evaluating whether retraining improves model performance.

Workflow

Historical Dataset
       |
       v
Data Preparation
       |
       v
Model Training
       |
       v
Model Evaluation
       |
       v
New Incoming Data
       |
       v
Drift Detection
       |
       v
Retraining Decision
       |
       v
Updated Model
       |
       v
Performance Comparison

What the Project Covers

1. Data Preparation

The notebook loads activity data from "dailyActivity_merged.csv" and prepares the features and target variable for model training.

The data is divided into training and testing sets using an 80:20 split.

2. Model Training

The project compares three regression approaches:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

Each model is trained and evaluated using:

- Mean Squared Error (MSE)
- R² Score

The model with the strongest R² score is selected for the subsequent workflow.

3. Drift Detection

The notebook explores statistical approaches for comparing historical and incoming data.

It includes:

Population Stability Index (PSI)

PSI is calculated to compare distributions across historical and new observations.

Kolmogorov–Smirnov Test

The KS test is used to compare the distributions of selected features.

The notebook uses a PSI threshold of 0.25 and a KS-test p-value threshold of 0.05 in its example retraining decision.

These thresholds are configurable examples rather than universal standards.

4. Model Retraining

When the drift condition is met, the notebook combines the historical and new datasets and retrains the XGBoost model.

The updated model is saved as:

model_v2.pkl

5. Performance Evaluation

The previous and updated models are evaluated on the new dataset.

The notebook compares:

- Mean Squared Error
- R² Score
- Actual versus predicted values

This makes it possible to inspect whether retraining improved the model rather than assuming that new training data automatically produces a better result.

Technology Stack

Area| Technology
Language| Python
Data processing| Pandas, NumPy
Machine learning| scikit-learn
Gradient boosting| XGBoost
Statistical testing| SciPy
Visualization| Matplotlib
Model persistence| Joblib
Development environment| Jupyter Notebook

Repository Contents

Adaptive-MLOps-Framework/
│
├── MLOPS_.ipynb    # Model training, drift analysis and retraining workflow
├── README.md       # Project documentation
└── LICENSE

Getting Started

1. Clone the repository

git clone https://github.com/mjmujeeb/Adaptive-MLOps-Framework-for-Automated-Model-Retraining-and-Drift-Management.git
cd Adaptive-MLOps-Framework-for-Automated-Model-Retraining-and-Drift-Management

2. Install the required libraries

The notebook uses the following Python packages:

pip install pandas numpy scikit-learn xgboost scipy matplotlib joblib jupyter

3. Prepare the dataset

Place the "dailyActivity_merged.csv" dataset in the notebook's working directory.

The dataset is not currently included in the repository, so it must be obtained separately.

4. Open the notebook

jupyter notebook MLOPS_.ipynb

Run the cells in sequence, beginning with data loading and preparation.

Expected Outputs

The notebook produces:

- Regression model evaluation results
- A comparison of model performance
- PSI and KS-test drift results
- A retraining decision
- A saved model artifact when the retraining branch executes
- Visual comparisons of predictions and data distributions

Important Implementation Notes

The current repository demonstrates the workflow in a notebook. It does not yet contain a deployed monitoring service, scheduled retraining pipeline, Docker deployment configuration, or a complete AWS-based model serving system.

The project title describes the intended direction; the current code should be understood as an experimental MLOps workflow.

Future Improvements

- Automate data ingestion and drift monitoring.
- Add experiment tracking with MLflow.
- Introduce model versioning and a model registry.
- Package training and inference using Docker.
- Add automated tests and CI/CD.
- Implement a deployment pipeline for approved model versions.
- Monitor production model performance over time.

License

See the "LICENSE" file in this repository.
