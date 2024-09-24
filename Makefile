# Полный процесс (от установки до импутации)
all: setup_environment download_data prepare_data train_catboost train_xgboost impute_str



# Создание окружения и активация
setup_environment:
	mamba env create -f environment.yml
	conda activate ksitest

# Загрузка данных и разархивирование
download_data:
	gdown https://drive.google.com/uc?id=1t5mAz9s6xu40J7_QUSnBqcLmCX-T8L56 -O data/raw/data.zip
	unzip data/raw/data.zip -d data/raw/

# Обработка данных
prepare_data:
	python ksitest/prepare/preprocess.py

# Обучение моделей CatBoost и XGBoost
train_catboost:
	python ksitest/train_model.py config/config_catboost.yaml

train_xgboost:
	python ksitest/train_model.py config/config_xgboost.yaml

# Импутация STR данных
impute_str:
	python ksitest/impute_str.py




optimize_all:
	make optimize_catboost
	make optimize_xgboost

optimize_catboost:
	python ksitest/optuna_hyperparameter_optimization.py config/config_catboost.yaml

optimize_xgboost:
	python ksitest/optuna_hyperparameter_optimization.py config/config_xgboost.yaml
