import argparse
import os
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml
from catboost import CatBoostRegressor
from scipy.stats import mstats
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from xgboost import XGBRegressor


def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config


def load_data(snp_file, str_file):
    X = pd.read_csv(snp_file, index_col="animal_id")
    y = pd.read_csv(str_file, index_col="animal_id").dropna(axis=0, how="any")

    common_ids = X.index.intersection(y.index)
    X = X.loc[common_ids]
    y = y.loc[common_ids]

    return X, y


# Методы обработки выбросов
def handle_outliers(df, col, method):
    if method == "simple":
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    elif method == "winsorization":
        df[col] = mstats.winsorize(df[col], limits=[0.05, 0.05])
        return df
    elif method == "iterative":
        prev_len = len(df)
        while True:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df_cleaned = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            if len(df_cleaned) == prev_len:
                break
            prev_len = len(df_cleaned)
        return df_cleaned
    elif method == "zscore":
        mean = df[col].mean()
        std = df[col].std()
        return df[(df[col] >= mean - 3 * std) & (df[col] <= mean + 3 * std)]
    else:
        return df


def train_model(config, iterations, depth, learning_rate):
    X, y = load_data(config["data"]["snp_file"], config["data"]["str_file"])
    outlier_methods = config["outliers"]["methods"]
    model_type = config["model"]["type"]

    total_rmse = 0
    total_r2 = 0
    n_models = len(y.columns)

    for method in outlier_methods:
        for target in tqdm(y.columns, desc=f"Training models with {method}"):
            y_target = y[target].dropna()
            combined = pd.concat([X.loc[y_target.index], y_target], axis=1)

            combined_cleaned = handle_outliers(combined, target, method)
            X_cleaned = combined_cleaned.drop(columns=[target])
            y_cleaned = combined_cleaned[target]

            if len(y_cleaned.unique()) == 1:
                print(f"Skipping {target} due to all equal targets after {method}.")
                continue

            X_train, X_val, y_train, y_val = train_test_split(
                X_cleaned, y_cleaned, test_size=0.2, random_state=42
            )

            if model_type == "CatBoost":
                model = CatBoostRegressor(
                    iterations=iterations,
                    depth=depth,
                    learning_rate=learning_rate,
                    verbose=0,
                    task_type="GPU",
                )
            elif model_type == "XGBoost":
                model = XGBRegressor(
                    n_estimators=iterations,
                    max_depth=depth,
                    learning_rate=learning_rate,
                    tree_method="hist",
                    device="cuda",
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)

            total_rmse += rmse
            total_r2 += r2

    avg_rmse = total_rmse / n_models
    avg_r2 = total_r2 / n_models

    return avg_rmse


def objective(trial):
    config = load_config(args.config_path)

    iterations = trial.suggest_int("iterations", 50, 500)
    depth = trial.suggest_int("depth", 3, 10)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.5, log=True)

    avg_rmse = train_model(config, iterations, depth, learning_rate)

    return avg_rmse


class SaveIntermediateResultsCallback:
    def __init__(self, output_file):
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, study, trial):
        with open(self.output_file, "a") as f:
            f.write(
                f"Trial {trial.number}: RMSE = {trial.value}, Params = {trial.params}\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization with Optuna."
    )
    parser.add_argument("config_path", type=str, help="Path to the configuration file.")

    args = parser.parse_args()

    config = load_config(args.config_path)

    # Определяем директорию и файл для логирования
    output_dir = config["logging"]["output_dir"]
    output_file = os.path.join(
        output_dir, f"optuna_results_{config['model']['type'].lower()}.txt"
    )

    save_callback = SaveIntermediateResultsCallback(output_file)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, callbacks=[save_callback])

    print("Best hyperparameters: ", study.best_params)
    print("Best RMSE: ", study.best_value)
