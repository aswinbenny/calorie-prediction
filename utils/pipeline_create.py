from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer

linear_models = ['ridge', 'lasso', 'elasticnet', 'bayesianridge', 'huberregressor']
tree_models = ['randomforest', 'extratrees', 'gradientboosting', 'xgboost']

def create_pipeline(model_name, model, params, features):
    """
    Create a machine learning pipeline based on the model type and features.
    """
    lin_num = features['linear']['num']
    lin_cat = features['linear']['cat']
    oth_num = features['other']['num']
    oth_cat = features['other']['cat']
    other_model_features = oth_num + oth_cat

    # Pipeline for linear models: scale selected numeric features
    if model_name in linear_models:
        pipeline = Pipeline(steps=[
            ('select', ColumnTransformer(transformers=[
                ('num', StandardScaler(), lin_num),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), lin_cat)
            ], remainder='drop')),
            (model_name, model(**params))
        ])

    # Pipeline for tree-based models: select relevant features directly (no scaling)
    elif model_name in tree_models:
        pipeline = Pipeline(steps=[
            ('select', ColumnTransformer(transformers=[
                ('num', 'passthrough', oth_num),
                ('cat', OneHotEncoder(drop='first'), oth_cat)
            ], remainder='drop')),
            (model_name, model(**params))
        ])

    elif model_name == 'lightgbm':
        pipeline = Pipeline(steps=[
            ('select', FunctionTransformer(lambda X: X[other_model_features], validate=False)),
            (model_name, model(**params))
        ])

    elif model_name == 'catboost':
        pipeline = Pipeline(steps=[
            ('select', FunctionTransformer(lambda X: X[other_model_features], validate=False)),
            (model_name, model(**params, cat_features=oth_cat))
        ])
    else:
        raise ValueError(f"unknown model name: {model_name}")

    return pipeline