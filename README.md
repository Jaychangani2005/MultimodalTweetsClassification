# Multimodal Tweets Classification

project for classifying crisis-related tweets using both text and image inputs from the CrisisMMD dataset.

## What is in this repo

- Training and experiment notebooks for informative and humanitarian classification
- Pretrained model checkpoints in the models folder
- Local BERT files in bert_model and bert_local
- Streamlit frontend in the frontend folder

## Quick setup

1. Create and activate a Python virtual environment (Python 3.10 or 3.11 recommended).
2. Install dependencies:

```bash
pip install -r requirement.txt
```

## Dataset

Download CrisisMMD from the official source:
https://crisisnlp.qcri.org/crisismmd

Place the extracted data under the data folder.

## Run notebooks

Open any notebook at the project root or in sample_data and run cells in order.


## Tech stack

- Language: Python
- Deep Learning: PyTorch, Torchvision
- NLP: Hugging Face Transformers (BERT)
- Vision Backbone: ResNet-50
- Data/ML Utilities: NumPy, Pandas, scikit-learn, SciPy
- Visualization: Matplotlib, Seaborn
- App UI: Streamlit
- Experiment Format: Jupyter Notebook

## Workflow

1. Prepare CrisisMMD data in the data folder.
2. Run preprocessing and feature preparation scripts/notebooks.
3. Train or load multimodal models (text + image fusion).
4. Evaluate informative/humanitarian classification performance.
5. Save best checkpoints in the models folder.
6. Run Streamlit app for interactive inference.

## Main folders

- data: dataset files and splits
- exp: preprocessing and model utility scripts
- models: saved model checkpoints
- frontend: Streamlit app and app-specific requirements
- sample_data: additional notebooks for experiments

## Notes

- Large model and dataset files are not bundled completely in Git history.
- If you use this project in research, cite CrisisMMD and the related model references.
