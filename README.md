# GC-CoDiff
GC-CoDiffContrastive Diffusion Models with Graph Construction for Tabular Data Synthesizing
##  Requirements

To set up the environment, run:

```bash
conda env create --file environment.yaml
conda activate [your_env_name]  # Replace with the actual environment name specified in environment.yaml

## Training

To train CoDi from scratch on the **Adult** dataset, run:

```bash
python main.py --data adult

## Evaluation

We evaluate the quality of synthetic data from multiple aspects:

### 1. Density Estimation of Single Column and Pair-wise Correlation

Evaluates the similarity between real and synthetic data in terms of column-wise distributions and correlations.

```bash
python eval/eval_density.py \
    --dataname [NAME_OF_DATASET] \
    --model [METHOD_NAME] \
    --path [PATH_TO_SYNTHETIC_DATA]


