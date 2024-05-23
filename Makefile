clean:
	rm -rf ./logs*
	rm -rf ./mlruns/*
	rm -rf ./models/*
	rm -rf ./output/*

install:
	python3 -m venv myenv && \
	. myenv/bin/activate && \
	pip install --no-cache-dir -r requirements.txt

run:
	. myenv/bin/activate && python main.py

tune:
	. myenv/bin/activate && python hyp-tuning/hyperparameter_tuning.py

test:
	. myenv/bin/activate && pytest tests/

run-mlflow:
	. myenv/bin/activate && \
	mlflow server --host 127.0.0.1 --port 8080