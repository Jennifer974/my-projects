# Telcom Customer Churn 🛰

![](../../images-library/stock-market-tracking-and-stocks.jpg)


## Objectives 🚀

Predict behavior to retain customers.

Source : Kaggle Challenge : [Telcom Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)


## Context

The project is about a telecom company 🛰 : they want to **predict if a customer is about to leave**. We can analyze all relevant customer data and develop focused customer retention programs.


## Getting Started

This project **is coded in Python**.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.


### Installing

You can create a virtual environment an install different packages with :

```
pip install -r requirements.txt
```

### Dataset

All important files used for my project saved [here](https://drive.google.com/drive/u/0/folders/1kKvppUSJ1ODAZqHWwiQ5vW-l2YN93loB)

📥 I download the dataset on [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn).

Each row represents a customer, each column contains customer’s attributes described on the column Metadata. The dataset contains about **7043 customers** with **21 features**. 

**Features** are the following:
- `customerID`: a unique ID for each customer
- `gender`: the gender of the customer
- `SeniorCitizen`: whether the customer is a senior (i.e. older than 65) or not
- `Partner`: whether the customer has a partner or not
- `Dependents`: whether the customer has people to take care of or not
- `tenure`: the number of months the customer has stayed
- `PhoneService`: whether the customer has a phone service or not
- `MultipleLines`: whether the customer has multiple telephonic lines or not
- `InternetService`: the kind of internet services the customer has (DSL, Fiber optic, no)
- `OnlineSecurity`: what online security the customer has (Yes, No, No internet service)
- `OnlineBackup`: whether the customer has online backup file system (Yes, No, No internet service)
- `DeviceProtection`: Whether the customer has device protection or not (Yes, No, No internet service)
- `TechSupport`: whether the customer has tech support or not (Yes, No, No internet service)
- `StreamingTV`: whether the customer has a streaming TV device (e.g. a TV box) or not (Yes, No, No internet service)
- `StreamingMovies`: whether the customer uses streaming movies (e.g. VOD) or not (Yes, No, No internet service)
- `Contract`: the contract term of the customer (Month-to-month, One year, Two year)
- `PaperlessBilling`: Whether the customer has electronic billing or not (Yes, No)
- `PaymentMethod`: payment method of the customer (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
- `MonthlyCharges`: the amount charged to the customer monthly
- `TotalCharges`: the total amount the customer paid

And the **Target** :
- `Churn`: whether the customer left or not (Yes, No)


### Import

```
#System library
import os

#Data manipulation
import pandas as pd, numpy as np

#Data visualization
import matplotlib.pyplot as plt, seaborn as sns

#Data Preprocessing
from sklearn.model_selection import train_test_split

#Supervised Machine Learning Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

#Metrics computed for Classification
from sklearn.metrics import classification_report, f1_score, accuracy_score
```

I load Telcom customers dataset in a pandas dataframe :
```
#Define dataset path :
filepath = os.path.join('data','WA_Fn-UseC_-Telco-Customer-Churn.csv')

#Load csv file with pandas dataframe
telcom_customers = pd.read_csv(filepath)
```



## Authors

* **Jennifer LENCLUME** - *Data Scientist* - 

For more informations, you can contact me :

LinkedIn : [LinkedIn profile](https://www.linkedin.com/in/jennifer-lenclume-a93728115/?locale=en_US)

Email : <a href="j.lenclume@epmistes.net">j.lenclume@epmistes.net</a>



## Acknowledgments

* Python
* Data Visualization
    - Matplotlib
    - Seaborn
* Exploration Data Analysis
    - Pandas
* Machine Learning : Churn
    - Scikit-Learn