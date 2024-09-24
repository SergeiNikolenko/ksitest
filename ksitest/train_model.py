import argparse
import os

import numpy as np
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
        return df  # Если метод "none", просто возвращаем исходный df


def train_model(config):
    X, y = load_data(config["data"]["snp_file"], config["data"]["str_file"])
    outlier_methods = config["outliers"]["methods"]
    model_type = config["model"]["type"]
    model_save_dir = config["model"]["save_dir"]

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    for method in outlier_methods:
        results = []
        total_rmse = 0
        total_r2 = 0
        n_models = len(y.columns)

        for target in tqdm(y.columns, desc=f"Training models with {method}"):
            y_target = y[target].dropna()
            combined = pd.concat([X.loc[y_target.index], y_target], axis=1)

            # Обработка выбросов
            combined_cleaned = handle_outliers(combined, target, method)
            X_cleaned = combined_cleaned.drop(columns=[target])
            y_cleaned = combined_cleaned[target]

            if len(y_cleaned.unique()) == 1:
                print(
                    f"Skipping {target} due to all equal targets after processing with {method}."
                )
                continue

            X_train, X_val, y_train, y_val = train_test_split(
                X_cleaned, y_cleaned, test_size=0.2, random_state=42
            )

            # Выбор модели
            if model_type == "CatBoost":
                model = CatBoostRegressor(
                    iterations=config["model"]["iterations"],
                    depth=config["model"]["depth"],
                    learning_rate=config["model"]["learning_rate"],
                    verbose=config["model"]["verbose"],
                    task_type="GPU",
                )
                model_file = f"{model_save_dir}/{target}_CatBoost_{method}.cbm"
            elif model_type == "XGBoost":
                model = XGBRegressor(
                    n_estimators=config["model"]["iterations"],
                    max_depth=config["model"]["depth"],
                    learning_rate=config["model"]["learning_rate"],
                    tree_method="hist",
                    device="cuda",
                )
                model_file = f"{model_save_dir}/{target}_XGBoost_{method}.json"
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)

            # Сохранение модели
            if model_type == "CatBoost":
                model.save_model(model_file)
            elif model_type == "XGBoost":
                model.save_model(model_file)

            results.append(f"{target:<20} | RMSE = {rmse:>8.4f} | R2 = {r2:>8.4f}")
            total_rmse += rmse
            total_r2 += r2

        # Логирование результатов
        log_dir = config["logging"]["output_dir"]
        log_file_prefix = config["logging"]["log_file_prefix"]
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Формирование имени выходного файла с указанием типа модели и метода обработки выбросов
        output_file = os.path.join(
            log_dir, f"{log_file_prefix}_{model_type}_{method}.txt"
        )
        avg_rmse = total_rmse / n_models
        avg_r2 = total_r2 / n_models

        with open(output_file, "w") as f:
            for result in results:
                f.write(result + "\n")
            f.write(f"\n**Average RMSE**: {avg_rmse:.4f}\n")
            f.write(f"**Average R2**: {avg_r2:.4f}\n")

        print(f"\nResults saved to {output_file}")
        print(f"Average RMSE: {avg_rmse:.4f}, Average R2: {avg_r2:.4f}")


if __name__ == "__main__":
    # Используем argparse для получения пути к конфигурационному файлу
    parser = argparse.ArgumentParser(
        description="Train model with specified configuration."
    )
    parser.add_argument("config_path", type=str, help="Path to the configuration file.")

    args = parser.parse_args()

    config = load_config(args.config_path)
    train_model(config)
    print("Training complete.")
