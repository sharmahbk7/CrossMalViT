.PHONY: install train eval test lint format clean docker reproduce

install:
	pip install -e ".[dev]"
	pre-commit install

train:
	python scripts/train.py --config configs/experiment/main_experiment.yaml

eval:
	python scripts/evaluate.py --checkpoint outputs/best_model.ckpt

test:
	pytest tests/ -v --cov=crossmal_vit

lint:
	ruff check crossmal_vit/ scripts/ tests/
	mypy crossmal_vit/

format:
	black crossmal_vit/ scripts/ tests/
	ruff check --fix crossmal_vit/ scripts/ tests/

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache/ .mypy_cache/ __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} +

docker:
	docker build -t crossmal-vit:latest -f docker/Dockerfile .

reproduce:
	python scripts/reproduce_paper.py --experiment all
