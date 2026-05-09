.PHONY: setup install test notebook ui clean

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements-dev.txt

install:
	pip install -e .

test:
	pytest

notebook:
	jupyter notebook notebooks/01_news_macro_agent_demo.ipynb

ui:
	streamlit run app.py

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} +
