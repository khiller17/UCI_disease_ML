# Breast Cancer & Heart Disease Prediction

## Overview

This folder contains two well-documented machine learning projects focused on predicting breast cancer malignancy and the presence of heart disease using real clinical datasets from the UCI repository.

Each notebook walks through a complete modeling pipeline: from data cleaning and feature engineering to model training, evaluation, and interpretation. Notebooks are supported by markdown commentary, exploratory plots, and thorough evaluation metrics.

---

## Project Structure

```text
FemtoDx/
├── data/                                               # data necessary to run notebooks                    
│   ├── wdbc.data
│   ├── wdbc.names
│   └── Module 2 heart + disease/
│   	├── ...  
├── Hiller-Bittrolff_heart_disease_analysis.ipynb       # notebooks            
├── Hiller-Bittrolff_WDBC_analysis.ipynb  
├── Hiller-Bittrolff_heart_disease_analysis.html        # html files with saved code and output from notebooks (plots, markdown etc)                
├── Hiller-Bittrolff_WDBC_analysis.html                                    
└── environment.yml                                     # yml to create conda env if you want to run the code
```

## Quickstart Instructions

Install on your laptop if you don't already have it:

https://www.anaconda.com/docs/getting-started/miniconda/install

Then do the following:

1. **Create and activate the conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate sfl  # smart focused lightweight env
   ```

2. **Lauch Jupyter Lab**
	```
	jupyter lab  
	```
3. Open breast_cancer_analysis.ipynb or heart_disease_analysis.ipynb and run all cells.


Data Source: UCI Machine Learning Repository – Heart Disease Dataset
https://archive.ics.uci.edu/dataset/45/heart+disease
Data collected from: Cleveland, Hungarian, Long Beach VA, and Switzerland clinical sites.
Credit to investigators: Andras Janosi, William Steinbrunn, Matthias Pfisterer, Robert Detrano.

