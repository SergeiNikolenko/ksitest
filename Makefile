# Обработка данных
prepare_data:
	python ksitest/prepare/preprocess.py

# Запуск моделей через конфигурационные файлы

train_catboost:
	python ksitest/train_model.py config/config_catboost.yaml

train_xgboost:
	python ksitest/train_model.py config/config_xgboost.yaml


# SHAPEIT, BEAGLE, IMPUTE2

run_shapeit:
	python ksitest/shapeit/run_shapeit.py

run_beagle:
	python ksitest/shapeit/run_beagle.py

run_impute2:
	python ksitest/shapeit/run_impute2.py

run_all:
	make run_shapeit
	make run_beagle
	make run_impute2
