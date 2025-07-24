# Scaling and Data Saturation in Protein Language Models

This repository contains all inference and training scripts, along with relevant data processing utilities, used in:

**Spinner, A., DeBenedictis, E., & Hudson, C. M. (2025).**  
*Scaling and Data Saturation in Protein Language Models.*  
*Proceedings of the ICML GenBio Workshop 2025.*

---

## Repository Structure

- `Analysis/` – notebooks and scripts for analysis and visualization  
- `Data/` – input datasets used in the study  
- `Results/` – model outputs, figures, and metrics  

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Data

We used the Substitution DMS dataset from the [ProteinGym benchmark](https://github.com/OATML-Markslab/ProteinGym) for all supervised experiments.

Specifically, the following files in `Data/` were downloaded from the ProteinGym repository:

- [`DMS_substitutions.csv`](https://github.com/OATML-Markslab/ProteinGym/blob/bb685245f8f4bb95a5b54b470396c23826cd6284/reference_files/DMS_substitutions.csv)  
  Protein-level metadata for all substitution DMS experiments.

- [`DMS_substitutions_Spearman_DMS_level.csv`](https://github.com/OATML-Markslab/ProteinGym/blob/bb685245f8f4bb95a5b54b470396c23826cd6284/benchmarks/DMS_zero_shot/substitutions/Spearman/DMS_substitutions_Spearman_DMS_level.csv)  
  Spearman correlations between unsupervised model predictions and experimental data.

For full dataset access, download instructions are available on the [ProteinGym Resources page](https://github.com/OATML-Markslab/ProteinGym/tree/main?tab=readme-ov-file#resources).

