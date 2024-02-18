import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_iris
from scipy.stats import chi2_contingency
import re
from datetime import datetime
from collections import Counter
from sklearn.preprocessing import MinMaxScaler


df = pd.read_pickle(r"C:\Users\tamar\Downloads\XY_train.pkl")
print(df)

#--Missing values--
# Calculate missing values before deletion
missing_values_before = df.isnull().sum()
df = df.dropna(thresh=df.shape[1]-2)
missing_values_after = df.isnull().sum()
missing_values_table = pd.DataFrame({
    'Before': missing_values_before,
    'After': missing_values_after,
    'Difference': missing_values_before - missing_values_after
})
print("Missing Values Before and After Deletion:")
print(missing_values_table)

#--filling email iwith the value unknowm--
df['email'] = df['email'].fillna('unknown')
missing_values_count = df['email'].isnull().sum()

#--Filling embedded_content and platform according to their probability--
embedded_content_prob = df['embedded_content'].value_counts(normalize=True)
platform_prob = df['platform'].value_counts(normalize=True)

def impute_missing_values(row, prob_dist):
    if pd.isnull(row):
        return np.random.choice(prob_dist.index, p=prob_dist.values)
    else:
        return row
df['embedded_content'] = df['embedded_content'].apply(lambda x: impute_missing_values(x, embedded_content_prob))
df['platform'] = df['platform'].apply(lambda x: impute_missing_values(x, platform_prob))

#--Fill missing values for email_verified and blue_tick--
df['email_verified'].fillna(df['blue_tick'], inplace=True)
df['blue_tick'].fillna(df['email_verified'], inplace=True)

#--filling gender randomly--
df['gender'].replace('None', np.nan, inplace=True)
gender_counts = df['gender'].value_counts()
df['gender'].fillna(pd.Series(np.random.choice(gender_counts.index,
                                               size=len(df.index),
                                               p=(gender_counts / gender_counts.sum()))),
                    inplace=True)

#drop the rows with missing values
df = df.dropna()
print(df)


