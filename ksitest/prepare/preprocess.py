import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


def clean_snp_column(column):
    allele_mapping = {"A": 0, "B": 1}
    column = column.replace("-", np.nan)
    column = column.replace(allele_mapping)
    return column


def preprocess_snp(snp_data_path):
    print("Preprocessing SNP data...")
    snp_data = pd.read_csv(snp_data_path, sep=";")
    df = snp_data.pivot_table(
        index="animal_id", columns="SNP Name", values="Allele1 - AB", aggfunc="first"
    )
    threshold = len(df) * 0.2
    df_cleaned = Parallel(n_jobs=-1)(
        delayed(clean_snp_column)(df[col])
        for col in tqdm(df.columns, desc="Cleaning SNP columns")
    )
    df = pd.concat(df_cleaned, axis=1)
    df = df.dropna(thresh=threshold, axis=1)
    df = df.fillna(df.mean())
    print(f"SNP data shape after cleaning: {df.shape}")
    return df


def preprocess_str(str_data_path):
    print("Preprocessing STR data...")
    str_data = pd.read_csv(str_data_path, sep=";")
    df = str_data.pivot_table(
        index="animal_id",
        columns="STR Name",
        values=["Allele1", "Allele2"],
        aggfunc="first",
    )
    df.columns = ["_".join(col).strip() for col in df.columns.values]
    for col in tqdm(df.columns, desc="Cleaning STR columns"):
        df[col] = df[col].where(df[col] < 600)
    print(f"STR data shape after cleaning: {df.shape}")
    return df


def save_processed_data(snp_data, str_data):
    snp_data.to_csv("data/processed/snp_pivot.csv", index=True)
    str_data.to_csv("data/processed/str_pivot.csv", index=True)


if __name__ == "__main__":
    snp_pivot = preprocess_snp("data/raw/FinalReport.csv")
    str_pivot = preprocess_str("data/raw/STR_train.csv")

    save_processed_data(snp_pivot, str_pivot)
    print("Preprocessing complete.")
