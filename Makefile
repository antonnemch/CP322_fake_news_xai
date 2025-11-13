.PHONY: setup data train eval xai layers figures

setup:
	python -m venv .venv && . .venv/Scripts/activate || . .venv/bin/activate; \
	pip install --upgrade pip && pip install -r requirements.txt

data:
	bash scripts/prepare_data.sh

train:
	bash scripts/run_train.sh

eval:
	bash scripts/run_eval.sh

xai:
	bash scripts/run_xai.sh

layers:
	bash scripts/run_layers.sh

figures:
	jupyter nbconvert --to notebook --execute notebooks/10_figures.ipynb --output artifacts/figures/10_figures.out.ipynb
