import os

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from tqdm import tqdm
from xgboost import XGBRegressor

# Загрузка данных
print("Loading SNP and STR test data...")
snp_data = pd.read_csv("data/raw/FinalReport.csv", sep=";")
str_test = pd.read_csv("data/raw/STR_test.csv", sep=";")


# Предобработка SNP данных
def preprocess_snp(df):
    print("Preprocessing SNP data...")
    df = df.pivot_table(
        index="animal_id", columns="SNP Name", values="Allele1 - AB", aggfunc="first"
    )
    df = df.fillna("0")
    print("Filled missing values with '0'.")

    allele_mapping = {"A": 0, "B": 1, "0": np.nan}
    df = df.replace(allele_mapping)
    return df


snp_pivot = preprocess_snp(snp_data)
print(f"SNP pivot data shape: {snp_pivot.shape}")


# Функция для загрузки моделей
def load_models(model_dir, model_type):
    print(f"Loading {model_type} models from {model_dir}...")
    models = {}
    for file_name in os.listdir(model_dir):
        # Извлекаем имя аллеля и маркера
        if file_name.endswith(".cbm") and model_type == "CatBoost":
            model_name = "_".join(
                file_name.split("_")[:2]
            )  # Извлекаем имя как 'Allele1_TGLA227'
            model = CatBoostRegressor()
            model.load_model(os.path.join(model_dir, file_name))
            models[model_name] = model
            print(f"Loaded CatBoost model: {model_name}")
        elif file_name.endswith(".json") and model_type == "XGBoost":
            model_name = "_".join(
                file_name.split("_")[:2]
            )  # Извлекаем имя как 'Allele1_TGLA227'
            model = XGBRegressor()
            model.load_model(os.path.join(model_dir, file_name))
            models[model_name] = model
            print(f"Loaded XGBoost model: {model_name}")
    return models


# Определяем, какой тип модели использовать
model_type = (
    "XGBoost"  # или "CatBoost", в зависимости от того, какие модели использовать
)
model_dir = f"models/{model_type.lower()}"


models = load_models(model_dir, model_type)


if not models:
    print(f"No models found in {model_dir}. Exiting...")
    exit()

print(f"Total models loaded: {len(models)}")

str_markers = str_test["STR Name"].unique()

imputed_data = []
missing_models = 0
for marker in tqdm(str_markers, desc="Predicting alleles for STR markers"):
    for allele in ["Allele1", "Allele2"]:
        target = f"{allele}_{marker}"
        if target in models:
            model = models[target]
            X_pred = snp_pivot.loc[str_test["animal_id"].unique()].fillna(0)
            X_pred = X_pred.apply(pd.to_numeric, errors="coerce").fillna(0)
            X_pred = X_pred.drop(columns=["BovineHD1900015984", "BovineHD3100000210"])
            y_pred = model.predict(X_pred)
            temp_df = pd.DataFrame(
                {"animal_id": X_pred.index, "STR Name": marker, allele: y_pred}
            )
            imputed_data.append(temp_df)
        else:
            print(f"Model for {target} not found.")
            missing_models += 1

if missing_models > 0:
    print(f"{missing_models} models were missing and skipped.")

# Объединение предсказанных данных
if imputed_data:
    imputed_df = pd.concat(imputed_data, axis=0)

    str_test = str_test.drop(columns=["Allele1", "Allele2"])
    str_test = str_test.merge(imputed_df, on=["animal_id", "STR Name"], how="left")

    output_path = "data/processed/STR_test_imputed.csv"
    str_test.to_csv(output_path, index=False, sep=";")
    print(f"Imputed STR test data saved to {output_path}.")
else:
    print("No imputed data to save. Exiting...")
