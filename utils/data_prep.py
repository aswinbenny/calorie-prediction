import pandas as pd
from sklearn.model_selection import train_test_split
from .wrangle import wrangle 

def load_and_prepare_data(data_dir='data'):
    """
    Load and prepare the dataset for training and validation.
    """
    df_train = pd.read_csv(f'{data_dir}/train.csv')
    df_calories = pd.read_csv(f'{data_dir}/calories.csv')

    # Apply wrangle
    df_train = wrangle(df_train).drop_duplicates()
    df_calories = wrangle(df_calories.rename(columns={'Gender': 'Sex', 'User_ID': 'id'})).drop_duplicates()

    # Merge train and calories
    df_train = pd.concat([df_train, df_calories], axis=0).drop_duplicates()

    # Split features and target
    X = df_train.drop(columns=['calories'])
    y = df_train['calories']

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    X_val1, X_val2, y_val1, y_val2 = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

    # Feature sets
    drop_for_linear_models = ['heart_rate', 'height', 'body_temp', 'duration']
    drop_for_other_models = ['temp_duration_product_bin', 'temp_hr_ratio_bin', 'age_bin']

    linear_model_features = X.columns.difference(drop_for_linear_models)
    other_model_features = X.columns.difference(drop_for_other_models)

    lin_num = X[linear_model_features].select_dtypes(include='number').columns.tolist()
    lin_cat = X[linear_model_features].select_dtypes(include='category').columns.tolist()
    oth_num = X[other_model_features].select_dtypes(include='number').columns.tolist()
    oth_cat = X[other_model_features].select_dtypes(include='category').columns.tolist()

    return {
        'X_train': X_train,
        'X_val1': X_val1,
        'X_val2': X_val2,
        'y_train': y_train,
        'y_val1': y_val1,
        'y_val2': y_val2,
        'features': {
            'linear': {'num': lin_num, 'cat': lin_cat},
            'other': {'num': oth_num, 'cat': oth_cat}
        }
    }