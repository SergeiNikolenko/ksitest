import pandas as pd
import numpy as np
import pickle

# Загрузка данных
snp_data = pd.read_csv('data/raw/FinalReport.csv', sep=';')
str_test = pd.read_csv('data/raw/STR_test.csv', sep=';')

# Предобработка SNP данных
def preprocess_snp(df):
    df = df.pivot_table(index='animal_id', columns='SNP Name', values='Allele1 - AB', aggfunc='first')
    df = df.fillna('0')
    allele_mapping = {'A': 0, 'B': 1, '0': np.nan}
    df = df.replace(allele_mapping)
    return df

snp_pivot = preprocess_snp(snp_data)

# Загрузка обученных моделей
with open('models/str_imputation_models.pkl', 'rb') as f:
    models = pickle.load(f)

# Получение списка необходимых STR маркеров
str_markers = str_test['STR Name'].unique()

# Предсказание аллелей для каждого маркера
imputed_data = []
for marker in str_markers:
    for allele in ['Allele1', 'Allele2']:
        target = f'{allele}_{marker}'
        if target in models:
            print(f'Предсказание для {target}')
            model = models[target]
            X_pred = snp_pivot.loc[str_test['animal_id'].unique()].fillna(0)
            y_pred = model.predict(X_pred)
            temp_df = pd.DataFrame({
                'animal_id': X_pred.index,
                'STR Name': marker,
                allele: y_pred
            })
            imputed_data.append(temp_df)

# Объединение предсказанных данных
imputed_df = pd.concat(imputed_data, axis=0)

# Объединение с исходным STR тестовым файлом
str_test = str_test.drop(columns=['Allele1', 'Allele2'])
str_test = str_test.merge(imputed_df, on=['animal_id', 'STR Name'], how='left')

# Сохранение проимпутированного файла
str_test.to_csv('data/processed/STR_test_imputed.csv', index=False, sep=';')
