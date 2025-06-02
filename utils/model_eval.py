import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

class ModelEvaluator():
    def __init__(self, model_dict, X_val, y_val, feature_dict):
        # Store models and validation data
        self.model_dict = model_dict
        self.X_val = X_val
        self.y_val = y_val
        self.feature_dict = feature_dict

    def segment_evaluation(self, n_segments=3):
        # Sort validation data by true values
        sorted_index = np.argsort(self.y_val.values)
        X_val_sorted = self.X_val.iloc[sorted_index].reset_index(drop=True)
        y_val_sorted = self.y_val.iloc[sorted_index].reset_index(drop=True)
        
        model_order = list(self.model_dict.keys())
        result = []
        segment_size = len(self.y_val) // n_segments
        
        # Evaluate each model on segments of data split by target values
        for model_name, model in self.model_dict.items():
            for i in range(n_segments):
                start = i * segment_size
                end = (i + 1) * segment_size if i < n_segments - 1 else len(self.y_val)
                X_seg = X_val_sorted[start:end]
                y_seg = y_val_sorted[start:end]
                y_pred = model.predict(X_seg)
                
                # Calculate metrics per segment
                rmse = np.sqrt(mean_squared_error(y_seg, y_pred))
                mae = mean_absolute_error(y_seg, y_pred)
                r2 = r2_score(y_seg, y_pred)
                
                result.append({
                    'model': model_name,
                    'segment_no': i,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                })
        
        result_df = pd.DataFrame(result)
        result_df['model'] = pd.Categorical(result_df['model'], categories=model_order, ordered=True)
        return result_df.sort_values(by=['segment_no', 'model']).reset_index(drop=True)

    def permutation_importance(self, n_repeats=10):
        rows = []
        
        # Calculate permutation importance for each model using its features
        for model_name, model in self.model_dict.items():
            all_features = self.X_val.columns.to_list()
            features = self.feature_dict[model_name]
            result = permutation_importance(model, self.X_val, self.y_val, n_repeats=n_repeats, random_state=42)
            
            # Fill importance scores; keep NaN for unused features
            importance_row = {f: np.nan for f in all_features}
            for feat, score in zip(features, result.importances_mean):
                importance_row[feat] = score
            
            importance_row['model'] = model_name
            rows.append(importance_row)
        
        importance_df = pd.DataFrame(rows).set_index('model')
        return importance_df

    def error_analysis(self):
        rows = []
        error_df = {}
        
        # Plot residuals for each model
        import math

        ncols = 3
        nrows = math.ceil(len(self.model_dict) / ncols)
        fig1, ax1 = plt.subplots(figsize=(15, 4 * nrows), nrows=nrows, ncols=ncols, sharex=True, sharey=True)
        ax1 = ax1.flatten()
        
        for i, (model_name, model) in enumerate(self.model_dict.items()):
            y_pred = model.predict(self.X_val)
            mae = mean_absolute_error(self.y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_val, y_pred))
            r2 = r2_score(self.y_val, y_pred)
            
            error = self.y_val - y_pred
            error_df[model_name] = error.values
            
            sns.scatterplot(x=self.y_val, y=error, ax=ax1[i], s=20, alpha=0.6)
            ax1[i].axhline(0, color='red', linestyle='--', linewidth=1)
            ax1[i].set_title(model_name)
            ax1[i].set_xlabel('True value')
            ax1[i].set_ylabel('Residuals')
            
            rows.append({
                'model': model_name,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            })

        error_df = pd.DataFrame(error_df)
        
        # Plot correlation between model errors
        fig2, ax2 = plt.subplots(figsize=(12, 10))
        ax2.set_title('Error Correlation')
        fig2.tight_layout()
        sns.heatmap(error_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax2)

        fig1.tight_layout()
        fig1.suptitle('Residual Comparison')

        eval_df = pd.DataFrame(rows)
        return eval_df, fig1, fig2