# Quora Insincere Questions Classification

![logo](images/quora.jpg)


## Objectives

Detect toxic content to improve online conversations

*Source : Kaggle Challenge* : [Quora insincere questions](https://www.kaggle.com/c/quora-insincere-questions-classification)


## Context

Quora is a popular website where anyone can ask and/or answer a question. There are more than 100 millions unique visitors per month.

Like any other forum, Quora is facing a problem: toxic questions and comments.

As you can imagine, Quora teams cannot check all of the Q&A by hand. So they decided to ask the data science community to help them to perform automatically insincere questions classification.


## Getting Started

This project **is coded in Python**.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.


### Installing

You can create a virtual environment an install different packages with :

```
pip install -r requirements.txt
```

### Dataset

All important files used for my project saved [here](https://drive.google.com/drive/u/0/folders/16ISw88BaAWbCt8d0RZtgZ6biLszrtatS)

📥 The Kaggle dataset is quite heavy and it may be too difficult for my laptop to perform the computations. Therefore, I provide **the train dataset** (to be sampled) and also **light word embeddings**, which I download [here](https://www.kaggle.com/c/quora-insincere-questions-classification/data).

Quora provided a dataset of questions with a label, and the features are the following:

- `qid`: a unique identifier for each question, an hexadecimal number
- `question_text`: the text of the question
- `target`: either 1 (for insincere question) or 0

🔦 In this project, the metric used for performance evaluation is the **F1-score**.


### Import

```
#System library
import os

#Data manipulation
import pandas as pd, numpy as np
from sklearn.utils import resample                                                #Select sample on dataset

#Data storage
import pickle
from joblib import dump, load                                                     #Joblib is more persistent than Pickle for large data, in particular large numpy arrays.

#Data visualization
import matplotlib.pyplot as plt, seaborn as sns

#Text Preprocessing
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import operator

#Data Preprocessing
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

#Supervised Machine Learning Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#Metrics computed for Classification
from sklearn.metrics import classification_report, f1_score
```

I load Quora questions dataset in a pandas dataframe :
```
#Define dataset path :
filepath = os.path.join('data','quora_train.csv')

#Load csv file with pandas dataframe
quora_questions = pd.read_csv(filepath)
```


## Authors

* **Jennifer LENCLUME** - *Data Scientist* - 

For more informations, you can contact me :

LinkedIn : [LinkedIn profile](https://www.linkedin.com/in/jennifer-lenclume-a93728115/?locale=en_US)

Email : <a href="j.lenclume@epmistes.net">j.lenclume@epmistes.net</a>


## Acknowledgments

* Python
* Natural Language Processing
    - NLTK
* Text Processing
* Sentiment Analysis : 
    - TextBlob
* Word Embedding :
    - Glove