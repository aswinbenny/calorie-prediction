import pandas as pd
import numpy as np
# Helper function to convert all column names to lowercase
def lower_case_columns(df):
    df.columns = df.columns.str.lower()
    return df

# Main data wrangling function
def wrangle(df):
    """
    Perform data wrangling on the input DataFrame.
    This includes feature engineering, normalization, and transformation of the dataset.
    """
    df = lower_case_columns(df)

    # Calculate Body Mass Index (BMI)
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

    # Deviation from normal body temperature (assumed to be 37°C)
    df['temp_deviation'] = df['body_temp'] - 37

    # Combined cardiovascular load
    df['cardio_load'] = df['duration'] * df['heart_rate']

    # Ratio of body temperature to heart rate
    df['temp_hr_ratio'] = df['body_temp'] / (df['heart_rate'] + 1e-5) 

    # Heart rate squared (to capture non-linear effects)
    df['heart_rate_squared'] = df['heart_rate'] ** 2

    # Weight per unit duration (intensity proxy)
    df['weight_duration_ratio'] = df['weight'] / (df['duration'] + 1e-5)

    # Product of BMI and heart rate
    df['bmi_hr_product'] = df['bmi'] * df['heart_rate']

    # Heart rate normalized per minute of activity
    df['hr_per_minute'] = df['heart_rate'] / (df['duration'] + 1e-5)

    # Product of body temperature and activity duration
    df['temp_duration_product'] = df['body_temp'] * df['duration']

    # Age-weighted cardiovascular load
    df['age_cardio_load'] = df['age'] * df['cardio_load']

    # Statistical features (row-wise mean and standard deviation)
    original_features = ['age', 'height', 'weight', 'duration', 'heart_rate', 'body_temp']
    df['row_mean'] = df[original_features].mean(axis=1)
    df['row_std'] = df[original_features].std(axis=1)
    
    # Reverse Feature Engineering

    # Log-transformations
    cols = ['age_cardio_load', 'weight_duration_ratio']
    for col in cols:
        df[f'log_{col}'] = np.log1p(df[col])
    

    
    # Binning 
    df['temp_duration_product_bin'] = pd.cut(df['temp_duration_product'], bins=10, labels=[f'bin_{i}' for i in range(10)])
    df['temp_hr_ratio_bin'] = pd.qcut(df['temp_hr_ratio'], q=10, duplicates='drop', labels=[f'bin_{i}' for i in range(10)])
    df['age_bin'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 65, 100], labels=[f'bin_{i}' for i in range(6)])
  
    
    # set categorical features
    cols = ['sex', 'temp_duration_product_bin', 'temp_hr_ratio_bin', 'age_bin']
    for col in cols:
        df[col] = df[col].astype('category')


    # Drop ID column (not useful for modeling)
    df = df.drop(columns=['id'])

    # Log-transform the target for more normal distribution
    df['calories'] = np.log1p(df['calories'])

    return df


