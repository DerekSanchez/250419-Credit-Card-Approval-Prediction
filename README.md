# Customer Credit Card Approval Prediction

## Description

**For an executive results deck, check 'results' folder**

This project builds and implements a binary classification model in order to predict customer approval for a bank using multiple numerical and categorical variables about the client

The original dataset can be found [here](https://www.kaggle.com/datasets/rohitudageri/credit-card-details?select=Credit_card.csv)

## Folder Structure
- data
    - raw_data.csv: original dataset
- notebooks
    - EDA.ipynb: Extensive Exploratory Data Analysis notebook using raw data
    - pipeline_check.ipynb: Notebook used for checking if each step of the preprocessing pipeline was applied correctly to the dataset
- results
    - Credit Card Approval Results.PDF: Executive presentation results deck
    - results_report.ipynb: Results of the models' predictions and model benchmarking. it uses the functions and classes defined in 'src'
    - log.txt: logs files used to keep track of important events (running a model, preprocessing information, etc.)
- src
    - config.py: file used to set global configuration variables
    - utils.py: file used to define global and reusable functions across the ecosystem
    - preprocess.py: file used to define the preprocessing pipeline through classes
    - train.py: file that contanins generic and reusable training functions for binary classification  models
    - evaluate.py: file that contanins generic and reusable testing functions for binary classification  models
