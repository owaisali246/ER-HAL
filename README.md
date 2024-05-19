ER-HAL: Hybrid Active Learning for Active Learning for Entity Resolution
=============

ER-HAL is a Python package for performing entity and text matching using
hybrid active machine learning. It provides built-in Active learning and
utilities that enable you to train and apply hybrid active learning
models for entity matching with less training data.

# Paper and Data

For details on the architecture of the models used, take a look at our
paper [Enhancing Entity Resolution with a Hybrid Active Machine Learning
Framework: Strategies for Optimal Learning in Sparse Datasets](). All
public datasets used in can be downloaded from the [datasets page
1](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md)
and [datasets page
2](https://github.com/wbsg-uni-mannheim/UnsupervisedBootAL/tree/master/datasets).

# Datasets:
In the datasets folder you can find all data sets
used for experimenation:

## Structured datasets:
1. Amazon_Google 
2. BeerAdvo_RateBeer 
3. Fodors_Zagats 
4. iTunes_Amazon 
5. Walmart-Amazon

## Dirty Datasets :
6. iTunes_Amazon 
7. Walmart_Amazon 
8. wdc_headphones
9. wdc_phones

## Textual:

10. abt_buy
11. Amazon_Google

For every dataset pair we provide the initial datasets, feature vector
files and files including matching labels for the train and test sets.

Results: In the results folder you can find all result files for DPQ and
STQ methods.

# Quick Start: ER-HAL in 30 seconds

There are tow main steps in using ER-HAL:

### 1.  Data processing: Load data and create Feature Vector File.

1.1. choose dataset under ER-HAL/createFv/CTE.py

1.2. run ER-HAL/createFv/createFeatureVectorFile.py script

### 2.  Run active learning process:

2.1. choose dataset and number of AML iterations under ER-HAL/cte.py

2.2. run ER-HAL/main.py script.

# Installation

We currently support only Python versions 3.5+. Installing using pip is
recommended:

``` 
pip install -r requirements.txt
```

# The Team

ER-HAL was developed by sultan moulay slimane university Ph.D students
JABRANE Mourad and TABBAA Hiba, under the supervision of Prof. HADRI
Aissam and Prof. HAFIDI Imad.
