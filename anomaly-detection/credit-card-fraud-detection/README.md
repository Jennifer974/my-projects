# Credit Card Fraud Detection 💳

<img src='../../images-library/computer-security-lock-and-payment.jpg'>
          
Photo by <a href="https://burst.shopify.com/@ndekhors?utm_campaign=photo_credit&amp;utm_content=Free+Stock+Photo+of+Computer+Security+Lock+And+Payment+%E2%80%94+HD+Images&amp;utm_medium=referral&amp;utm_source=credit">Nicole De Khors</a> from <a href="https://burst.shopify.com/technology?utm_campaign=photo_credit&amp;utm_content=Free+Stock+Photo+of+Computer+Security+Lock+And+Payment+%E2%80%94+HD+Images&amp;utm_medium=referral&amp;utm_source=credit">Burst</a>


## Objectives 🚀

Identify fraudulent credit card transactions 💳.


## Context

Fraud detection is a billion dollars business 💰: according to the [Nilson Report](https://nilsonreport.com/), credit card fraud adds up to 24 billion dollars in 2018 ! 

Every bank and insurance company has some fraud detection algorithms. They are working hard to find out the fraudulent transactions amongst a huge number of valid ones.

Some companies are doing really good. For instance, Paypal has developed really complicated and efficient algorithms to perform fraud detection.


## Getting Started

This project is coded in Python.
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.


### Installing

You can create a virtual environment an install different packages with :

```
pip install -r requirements.txt
```

### Dataset

All important files used for my project saved [here](https://drive.google.com/drive/u/0/folders/12q21VCLoatz58Nr45ATdJmZi33Z2xMot).

📥 I download the dataset on [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).

The dataset contains transactions made by credit cards in September 2013 by european cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284 807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.17% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. The only features which have not been transformed with PCA are `Time` and `Amount`:
- Features `V1`, `V2`, … `V28` are the principal components obtained with PCA
- Feature `Time` contains the seconds elapsed between each transaction and the first transaction in the dataset. 
- The feature `Amount` is the transaction Amount (euros), this feature can be used for example-dependant cost-senstive learning. 
- Feature `Class` is the response variable and it takes value 1 in case of fraud and 0 otherwise

### Import

```
#System library
import os

#Data manipulation
import pandas as pd, numpy as np

#Data storage
import pickle

#Data visualization
import matplotlib.pyplot as plt, seaborn as sns
from mpl_toolkits import mplot3d                                                           #3D visualization
%matplotlib notebook

#Data Preprocessing
from sklearn.preprocessing import StandardScaler                                           #Data scaling
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV     #Data split and hyperparameter                       
from imblearn.over_sampling import SMOTE                                                   #Data oversampling

#Supervised Machine Learning Model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklear.svm import SVC

#Unsupervised Machine Learning Model
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

#Metrics computed for Classification
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.metrics import roc_curve
```

## 1. Exploration Data Exploration

I load credit card dataset in pandas dataframe :
```
#Define dataset path :
filepath = os.path.join('data', 'creditcard.csv')

#Load csv file with pandas dataframe
creditcard = pd.read_csv(filepath)
```

### 1.1 Data Cleaning and Exploration

As we can see, the data has thirty one columns as follows:
- `V1`, `V2`, … `V28` : principal components obtained with PCA
- `Time`: seconds elapsed between each transaction and the first transaction in the dataset. 
- `Amount`: transaction Amount (euros), this feature can be used for example-dependant cost-senstive learning. 
- `Class`: response variable which takes value 1 in case of fraud and 0 otherwise
```
creditcard.shape

#see the first five rows of data
creditcard.head()
```
It contains continue values but my target (`Class`) is discrete values : It's a supervised Machine Learning case (Classification).
```
creditcard.info()
```
Next, I display some statistical summaries of the numerical columns below :
- `Amount` mean = 88€
- `Amount` max = 25 691€
- `Time` max = 172 792s (i.e. 48h)

```
#Statistical summaries of the numerical columns
creditcard.describe()
```
Some relevant informations with data cleaning are :
- No null values : 
```
#Numbers of null values
creditcard.isna().sum()
```
- 1081 duplicates :
```
#Numbers of duplicates
creditcard.duplicated().sum()

#Drop duplicates
creditcard.drop_duplicates(inplace=True)
```

### 1.2 Data Analysis

According to Kaggle documentation, `V1` is a result of a PCA Dimensionality reduction to protect user identities and sensitive features (`V1`to `V8`) so I select `Time`, `Amount` and `V1` to visualize my data :

<img src='graph/credit-card-fraud-dist-amount-time.jpg'>

<img src='graph/credit-card-fraud-dist-V1-time.jpg'>

<img src='graph/credit-card-fraud-dist-amount-V1.jpg'>

#### Fraudulent/Non-fraudulent transactions distribution :
    
The dataset is highly **unbalanced**, the positive class (frauds) account for **0.17%** of all transactions :

<img src='graph/credit-card-fraud-transactions-distributions.jpg'>

Later, I use **oversampling or undersampling method** to balance my dataset to build Supervised Machine Learning model.

#### Amount transactions distribution :

Most of amount transactions is **lower than 2 000€**, however there is amount higher than 25 000€. I suppose that fraudulent transactions is lower than 2 000€. 

<img src='graph/credit-card-transactions-amount-distributions.jpg'>

I plot *Transaction Amount* for fraudulent and non-fraudulent transactions to have more precision :

<img src='graph/credit-card-transactions-amout-nonfraud-fraud-distributions.jpg'>

**Fraudulent transactions** amount is **150€ in average**:

<img src='graph/credit-card-transactions-amout-fraud-distributions.jpg'>

#### Time transactions distribution :

Transactions are on **two days** that corresponds to **48 hours** (1 hour = 3600s):  I convert **hour** to **second**.
I constat that transactions time is globaly seasonal.

<img src='graph/credit-card-transactions-time-distributions.jpg'>

I notice that fraudulent transactions increase overnight.

<img src='graph/credit-card-transactions-time-fraud-distributions.jpg'>

#### Features correlation :

Correlation matrix graphically gives us an idea of how features correlate with each other and can help us predict what are the features that are most relevant for the prediction : 

<img src='graph/credit-card-transactions-correlation-heatmap.jpg'>

I can clearly see that most of the features **don't correlate to other features** but there are some features that either has a **positive or a negative correlation** with each other. For example, `V12` and `V17` are negatively correlated. 

<img src='graph/credit-card-fraud-dist-v12-v17.jpg'>

<img src='graph/credit-card-fraud-dist-v1-v3.jpg'>

## 2. Anomaly Detection

Firstly, I build Supervised Machine Learning model to identify fraudulent credit card transactions and secondly I use Unsupervised Machine Learning model to compare results.

### 2.1. Using Supervised Machine Learning 🤖

#### 2.1.1. Features and labels definitions

I select this following features to build `X` :
- `V1`, `V2`, … `V28` : principal components obtained with PCA
- `Time` (in hours): contains the seconds elapsed between each transaction and the first transaction in the dataset. 
- `Amount` (in euros) : transaction amount

```
X = creditcard.iloc[:, :30]
```

Labels are `Class` : it takes value 1 in case of fraud and 0 otherwise : it represents `y` 

```
y = creditcard['Class']
```
I **split** data into **train** and **test** to build model with `train_test_split` function from scikit-learn :

```
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=.2,        #train represents 80% of dataset and test represents 20% of dataset
                                                    random_state=0,      #keep the same random split     
                                                    stratify=y)          #conserve the same distributions for labels
```

#### 2.1.2 Scaling and oversampling to improve model performance

##### Data scaling

I **scale** data, otherwise one feature will be more relevant than the other, for example transaction `Amount` is higher than `V1` (numerically).

```
#Instanciate a scaler
scaler = StandardScaler()

#Scale train and test data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
##### Data oversampling

Credit card dataset is highly unbalanced. I use **oversampling** method to balance dataset and improve model performance: **SMOTE** : it consists to create data for fradulent transactions to balance labels for **train data**.

```
#Instanciate a SMOTE
oversampling = SMOTE()

#Balance train data
X_train, y_train = oversampling.fit_resample(X_train, y_train)
```
<img src='graph/credit-card-fraud-transactions-distributions-oversampled.jpg'>

#### 2.1.3 Modeling

Logistic Regression and Random Forest are models (from scikit-learn) I used to predict fraudulent transactions.

I build this function to draw prediction later :
```
def graph_prediction(X, y_true, y_pred, model):
    '''This function draw true distribution and predicted distribution transactions for V2 and V1
    
    Parameters
    ------------
    X : array of float
        data to draw
    y_true: array of int
        contains Class transactions
    y_pred: array of int
        contains predictions for fraudulent of non-fraudlent transactions
   
    Returns
    ------------
    Predictions graph
    '''
    fig = plt.figure(figsize=(14, 6))

    fig.add_subplot(121)
    #Plot True distribution V2=f(V1)
    plt.title('True distribution for credit card fraud detection : V2=f(V1)', fontsize=14)
    plt.scatter(X[:, 1], X[:, 2], c=y_true)
    plt.xlabel('V1')
    plt.ylabel('V2')

    fig.add_subplot(122)
    #Plot Predicted distribution V2=f(V1)
    plt.title('Predicted distribution for credit card fraud detection V2=f(V1)', fontsize=14)
    plt.scatter(X[:, 1], X[:, 2], c=y_pred)
    plt.xlabel('V1')
    plt.ylabel('V2')
    
    #Save the graph with plt.savefig
    filepath_prediction = os.path.join('graph', f'credit-card-fraud-prediction-{model}.jpg')
    plt.savefig(filepath_prediction,                                                     #Image path
            format='jpg',                                                                #Image format to save
            bbox_inches='tight')  
    
    plt.show()
```

##### Logistic Regression

I define Logistic Regression function to change hyperparameter :

```
def get_logistic_regression(C=1.0):
    '''
    This function predicts Class transactions with Logistic Regression model
    
    Parameters
    ------------
    C : float, default=1.0
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization.
        
    Returns
    ------------
    y_pred : array of int
        contains predictions for fraudulent of non-fraudlent transactions
    y_pred_proba : array of int
        contains the probability of the predicted class
    '''
    #Instanciate model
    lr = LogisticRegression(C=C)
    
    #Model fitting
    print('Logistic Regression time to fit :')
    %time
    lr.fit(X_train, y_train)

    #Model predictions
    print('\n Logistic Regression time to predict y :')
    %time
    y_pred = lr.predict(X_test)
    print('\n Logistic Regression time to predict y proba :')
    %time
    y_pred_proba = lr.predict_proba(X_test)

    return y_pred, y_pred_proba

```
I predict and evaluate my model with `accuracy_score` and `classification_report` from scikit-learn :

1. Prediction for `C=1.0`:
```
#Prediction for C=1.0
y_pred_lr, y_pred_proba_lr = get_logistic_regression(C=1.0)
```
<img src='result/credit-card-fraud-detection-lr-c-10.png'>

```
#Compute accuracy score C=1.0
acc_lr = round(accuracy_score(y_test, y_pred_lr), 4) * 100
acc_lr
```
`accuracy_score = 97.23%`

2. Prediction for `C=0.1`:

<img src='result/credit-card-fraud-detection-lr-c-01.png'>

`accuracy_score = 97.24%`

3. Prediction for `C=0.01`:

<img src='result/credit-card-fraud-detection-lr-c-001.png'>

`accuracy_score = 97.26%`

###### ROC Curve

I plot ROC Curve to choose the best model :

```
#Compute ROC characteristic
fpr_lr, tpr_lr, threshold_lr = roc_curve(y_test, y_pred_proba_lr[:, 1])
fpr_lr_c_01, tpr_lr_c_01, threshold_lr_c_01 = roc_curve(y_test, y_pred_proba_lr_c_01[:, 1])
fpr_lr_c_001, tpr_lr_c_001, threshold_lr_c_001 = roc_curve(y_test, y_pred_proba_lr_c_001[:, 1])
```

```
plt.figure(figsize=(10, 8))

#Plot ROC Curve for Logistic Regression
plt.title('ROC Curve for Logistic Regession', fontsize=14)                                #ROC Curve graph Title

#Plot ROC Curve for Logistic Regression for C=1.0
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression Classifier Score for Logistic Regression for C=1.0: {acc_lr}%')

#Plot ROC Curve for Logistic Regression for C=0.1
plt.plot(fpr_lr_c_01, tpr_lr_c_01, label=f'Logistic Regression Classifier Score for Logistic Regression for C=0.1: {acc_lr_c_01}%')   

#Plot ROC Curve for Logistic Regression for C=0.01
plt.plot(fpr_lr_c_001, tpr_lr_c_001, label=f'Logistic Regression Classifier Score for Logistic Regression for C=0.01: {acc_lr_c_001}%')

#Plot ROC Curve limited score
plt.plot([0, 1], [0, 1], 'k--')
plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)',           #Plot arrow legend
             xy=(0.5, 0.5),                                                              #The point *(x,y)* to annotate.
             xytext=(0.6, 0.3),                                                          #The position *(x,y)* to place the text at.
             arrowprops=dict(facecolor='#6E726D', shrink=0.05))                          #The properties used to draw a arrow between the positions *xy* and *xytext*.     

#Graph property
plt.xlabel('False Positive Rate', fontsize=12)                                           #Abscissa label
plt.ylabel('True Positive Rate', fontsize=12)                                            #Ordinate label
plt.legend(loc='lower right')                                                            #Graph Legend

#Save the graph with plt.savefig
filepath_roc_curve_lr = os.path.join('graph', 'credit-card-fraud-roc-curve-lr.jpg')
plt.savefig(filepath_roc_curve_lr,                                                       #Image path
            format='jpg',                                                                #Image format to save
            bbox_inches='tight')                                                         #Keep abscissa legend


plt.show()
```
<img src='graph/credit-card-fraud-roc-curve-lr.jpg'>

I validate Logistic Regression model for `C = 0.01` because it has a **best accuracy_score : 97.26%**.

<img src='graph/credit-card-fraud-prediction-lr.jpg'>

##### Random Forest Classifier

I define Random Forest Classifier function to change hyperparameter :

```
def get_random_forest_classifier(n_estimators=100, max_depth=50):
    '''
    This function predicts Class transactions with Random Forest Classifier model
    
    Parameters
    ------------
    n_estimators : integer, optional (default=100)
        The number of trees in the forest.
    max_depth : integer or None, optional (default=50)
        The maximum depth of the tree. 
        
    Returns
    ------------
    y_pred : array of int
        contains predictions for fraudulent of non-fraudlent transactions
    y_pred_proba : array of int
        contains the probability of the predicted class
    '''
    #Instanciate model
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    
    #Model fitting
    print('Random Forest Classifier time to fit :')
    %time
    rf.fit(X_train, y_train)

    #Model predictions
    print('\n Random Forest Classifier time to predict y :')
    %time
    y_pred = rf.predict(X_test)
    print('\n Random Forest Classifier time to predict y proba :')
    %time
    y_pred_proba = rf.predict_proba(X_test)

    return y_pred, y_pred_proba
```
I predict and evaluate my model with `accuracy_score` and `classification_report` from scikit-learn :

1. Prediction for `n_estimators=100, max_depth=50` :
```
#Prediction for n_estimators=100, max_depth=50
y_pred_rf, y_pred_proba_rf = get_random_forest_classifier(n_estimators=100, max_depth=50)
```
<img src='result/credit-card-fraud-detection-rf-default.png'>

`accuracy_score = 99.96%`

2. Prediction for `n_estimators=100, max_depth=None` :

<img src='result/credit-card-fraud-detection-rf-depth-none.png'>

`accuracy_score = 99.96%`

3. Prediction for `n_estimators=50, max_depth=None` :

<img src='result/credit-card-fraud-detection-rf-est-50.png'>

`accuracy_score = 99.96%`

###### ROC Curve

I plot ROC Curve to choose the best model :

<img src='graph/credit-card-fraud-roc-curve-rf.jpg'>

I validate Random Forest model for `n_estimators = 100, max_depth = None` because the ROC score is better.

<img src='graph/credit-card-fraud-prediction-rf.jpg'>

##### Conclusion

I plot ROC Curve for Supervised Machine Learning models and I validate Random Forest model for `n_estimators = 100, max_depth = None` because the ROC score is better.

<img src='graph/credit-card-fraud-roc-supervisedML.jpg'>

<img src='result/credit-card-fraud-detection-rf-depth-none.png'>


## Authors

* **Jennifer LENCLUME** - *Data Scientist* - 

For more informations, you can contact me :

LinkedIn : [LinkedIn profile](https://www.linkedin.com/in/jennifer-lenclume-a93728115/?locale=en_US)

Email : <a href="j.lenclume@epmistes.net">j.lenclume@epmistes.net</a>


## Acknowledgments

* Python 
* Machine Learning
* Anomaly Detection