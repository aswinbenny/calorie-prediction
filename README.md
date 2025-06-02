# Calories Burn Prediction Project

This project presents a machine learning ensemble to predict the calories burned during workouts from input features like biometric attributes (age, gender, weight, height) and workout-specific features (duration, heart rate, body temperature), along with engineered features derived from them.  
This code was originally built for a Kaggle competition and then refined into a modular, standalone project.

## Dataset Overview
The dataset used in this project was originally released as part of a Kaggle competition and is publicly available for download.  
You can access the dataset from the official competition page here:  
 [Playground Series - Season 5, Episode 5](https://www.kaggle.com/competitions/playground-series-s5e5/data)

 **Note:** The dataset is subject to Kaggle’s terms of use. Make sure to log in to your Kaggle account to download the files.

## Project Structure and Tools



### Utility Modules

- **`wrangle.py`** – Contains all feature engineering logic, including interaction terms, statistical features, and features derived from residual analysis.
- **`data_prep.py`** – Loads, splits, and prepares the data for training and validation.
- **`pipeline_create.py`** – Builds model-specific pipelines to scale or provide tailored features for each model.
- **`model_eval.py`** – Defines the `ModelEvaluator` class, which evaluates models on segmented data (based on target values), generates permutation importance plots, compares residuals, and computes model-wise error correlations.

### Notebooks

- **EDA Notebook** – Serves as the starting point to explore the dataset through basic exploratory data analysis (EDA), including the distribution of the target variable and other numeric features. Helped to form an initial understanding of the data.
  
- **optuna-hyperparameter_tuning** – Demonstrates how Optuna, a powerful Bayesian optimization library, is used to tune several base models and the meta-model. The hyperparameter optimization significantly improved model performance.

- **model_selection Notebook** – Applies the evaluation utilities to compare models at different stages. The permutation importance plots were especially helpful for feature selection and further feature engineering.

- **residual_analysis Notebook** – Trains a Random Forest classifier to identify samples with high residual errors (formulated as a binary classification task). This helped pinpoint features contributing to poor predictions, which were then refined to improve the model’s accuracy.

- **05_ensemble_training Notebook** – Trains the final stacked ensemble model and makes predictions on the test set.  
  Final model performance:
  - **RMSE**: 0.0598  
  - **MAE**: 0.0338  
  - **R²**: 0.9961  

## Steps to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/calorie-burn-predictor.git
cd calorie-burn-predictor
```

### 2. Set Up Your Environment

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download the Dataset
Download the dataset from the link provided above and place the files inside the data/raw/ directory.

### 4. Explore the Project Notebooks
Run the notebooks individually based on your interest. Each notebook focuses on a specific stage of the workflow.

## License

This project is released under the MIT License.

## Author

Developed by [Aswin Benny](https://github.com/aswinbenny)