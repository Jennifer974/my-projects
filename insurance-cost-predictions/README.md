# Insurance cost prediction



## Objectives 🚀

Predict the price of yearly medical bills.

*Source* : Kaggle Challenge : [Medical Cost Personal Datasets](https://www.kaggle.com/mirichoi0218/insurance)



## Context

Insurance companies want to **determine the yearly insurance premium for a person** and they use information like a person's age, sex, BMI, no. of children and smoking habit **to predict the price of yearly medical bills**. Here, I use **Linear Regression model with `PyTorch`** to make prediction.



## Prerequisites

- I use `PyTorch` Library (**version 1.5.0**) to create model: `PyTorch` is an open source machine learning library based on the Torch library, developed by Facebook's AI Research lab. You can find documentation [here](https://pytorch.org/docs/stable/index.html).
- I use `Jovian.ml` platform for this project : `Jovian.ml` is a better place for your data science projects, Jupyter notebooks, machine learning models, experiment logs, results and more.



## Dataset

📥 I download the dataset on [Kaggle](https://www.kaggle.com/mirichoi0218/insurance).

The dataset contains personal informations :
- `age` : age of primary beneficiary
- `sex` : insurance contractor gender, female, male
- `bmi` : Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally `18.5` to `24.9`
- `children` : Number of children covered by health insurance / Number of dependents
- `smoker` : Smoking
- `region` : the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
- `charges` : Individual medical costs billed by health insurance



## Library Installation

    !conda install numpy pytorch torchvision cpuonly -c pytorch -y
    !pip install matplotlib --upgrade --quiet
    !pip install jovian --upgrade --quiet
    !pip install pandas --upgrade --quiet
    !pip install seaborn --upgrade --quiet


## Library Importation

    import jovian

    #Data Manipulation
    import pandas as pd

    #Data Visualization
    import matplotlib.pyplot as plt
    import seaborn as sns

    #PyTorch Library
    import torch
    import torchvision
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.datasets.utils import download_url
    from torch.utils.data import DataLoader, TensorDataset, random_split
    
    
I download and load Insurance Cost dataset in a pandas dataframe :  
    
    #Define dataset path :
    DATASET_URL = "https://hub.jovian.ml/wp-content/uploads/2020/05/insurance.csv"
    DATA_FILENAME = "insurance.csv"

    # Download insurance dataset
    download_url(DATASET_URL, '.')
    
    #Load csv file with pandas dataframe
    dataframe_raw = pd.read_csv(DATA_FILENAME)



## Authors

* **Jennifer LENCLUME** - *Data Scientist* - 

For more informations, you can contact me :

Jovian : [Jovian profile](https://jovian.ai/jennifer974)

LinkedIn : [LinkedIn profile](https://www.linkedin.com/in/jennifer-lenclume-a93728115/?locale=en_US)

Email : <a href="j.lenclume@epmistes.net">j.lenclume@epmistes.net</a>



## Acknowledgments

* Python
* Data Visualization
    - Matplotlib
    - Seaborn
* Exploration Data Analysis
    - Pandas
* Machine Learning : 
    - PyTorch
* Deep Learning : 
    - PyTorch
* GPU :
    - Jovian       