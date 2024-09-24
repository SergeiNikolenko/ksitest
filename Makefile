# Обработка данных
prepare_data:
	python ksitest/prepare/preprocess.py

# Запуск моделей через конфигурационные файлы

train_catboost:
	python ksitest/train_model.py config/config_catboost.yaml

train_xgboost:
	python ksitest/train_model.py config/config_xgboost.yaml


optimize_all:
	make optimize_catboost
	make optimize_xgboost

optimize_catboost:
	python ksitest/optuna_hyperparameter_optimization.py config/config_catboost.yaml

optimize_xgboost:
	python ksitest/optuna_hyperparameter_optimization.py config/config_xgboost.yaml
