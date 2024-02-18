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

#--Droping the rows with missing values--
df = df.dropna()

#Converting message_date to categorical
df['message_date'] = pd.to_datetime(df['message_date'])
df['hour'] = df['message_date'].dt.hour
morning_interval = range(6, 12)  # 6:00 AM to 11:59 AM
noon_interval = range(12, 18)     # 12:00 PM to 5:59 PM
evening_interval = range(18, 24)  # 6:00 PM to 11:59 PM
def categorize_hour(hour):
    if hour in morning_interval:
        return 'Morning'
    elif hour in noon_interval:
        return 'Noon'
    elif hour in evening_interval:
        return 'Evening'
    else:
        return 'Night'
df['message_time_category'] = df['hour'].apply(categorize_hour)
df.drop(columns=['hour'], inplace=True)

#1.Create a new column based on the length of messages
df['message_length'] = df['text'].apply(lambda x: len(x))
#2.Creating the number of messages sent by the user
df['num_messages_sent'] = df['previous_messages_dates'].apply(len)
#3.creating num of followers and following
df['follower_count'] = df['date_of_new_follower'].apply(lambda x: len(x))
df['following_count'] = df['date_of_new_follow'].apply(lambda x: len(x))

#4.creating the difference between followers and following
df['follower_following_diff'] = abs(df['following_count'] - df['follower_count'])
new_columns_df = df[['follower_count', 'following_count', 'follower_following_diff']]

#5.N-GRAM
ngram_range = (2, 2)  # Set N-gram range to bi-grams
vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=20)
X_text_features = vectorizer.fit_transform(df['text'])
df = pd.concat([df.reset_index(drop=True), pd.DataFrame(X_text_features.toarray())], axis=1)

#6.Extracting email domain endings
def extract_email_domain_ending(email):
    if pd.isnull(email):
        return 'Missing'
    match = re.search(r'\.(\w+)$', email)
    if match:
        return match.group(1)
    else:
        return 'Unknown'
df['email_domain_ending'] = df['email'].apply(extract_email_domain_ending)
email_domain_ending_counts = df['email_domain_ending'].value_counts()

#7.Creating seniority in years
df['account_creation_date'] = pd.to_datetime(df['account_creation_date'])
current_date = datetime.now()
df['seniority'] = (current_date - df['account_creation_date']).dt.days / 365.25

#8.Creating the average time difference between messages
def calculate_average_time_difference(message_dates_array):
    message_dates = [datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S') for date_str in message_dates_array]
    time_diffs = []
    for i in range(1, len(message_dates)):
        time_diff = (message_dates[i] - message_dates[i - 1]).total_seconds()
        if time_diff < 0:
            time_diff = (message_dates[i - 1] - message_dates[i]).total_seconds()
        time_diffs.append(time_diff)
    if len(time_diffs) > 0:
        average_time_difference = sum(time_diffs) / len(time_diffs)
        average_time_difference = int(average_time_difference)
        return average_time_difference
    else:
        return None
df['average_time_difference'] = df['previous_messages_dates'].apply(calculate_average_time_difference)

#4.Feature representation--------------------------------------------------->
#1.Normalized values by dividing each column by its maximum value
columns_to_normalize = ['message_length', 'num_messages_sent', 'follower_count', 'following_count',
                        'seniority', 'average_time_difference', 'follower_following_diff']
for column in columns_to_normalize:
    max_value = df[column].max()
    df[f'normalized_{column}'] = df[column] / max_value
for column in columns_to_normalize:
    normalized_column_name = f'normalized_{column}'
    print(f"Normalized {column}:")
    print(df[normalized_column_name])

#2.One-hot encoding
email_domain_ending_onehot = pd.get_dummies(df['email_domain_ending'], prefix='email_ending')
embedded_content_onehot = pd.get_dummies(df['embedded_content'], prefix='embedded_content')
platform_onehot = pd.get_dummies(df['platform'], prefix='platform')
message_time_category_onehot = pd.get_dummies(df['message_time_category'], prefix='message_time')
df = pd.concat([df, email_domain_ending_onehot, embedded_content_onehot, platform_onehot, message_time_category_onehot], axis=1)
df.drop(['email_domain_ending', 'embedded_content', 'platform', 'message_time_category'], axis=1, inplace=True)

#3.Converting to binary values
bool_to_binary = {True: 1, False: 0}
df['email_verified'] = df['email_verified'].map(bool_to_binary)
df['blue_tick'] = df['blue_tick'].map(bool_to_binary)

for column in df.columns:
    if df[column].dtype == bool:
        df[column] = df[column].astype(int)

gender_to_binary = {'F': 1, 'M': 0}
df['gender'] = df['gender'].map(gender_to_binary)

#4. Normalization the n-gram features
scaler = MinMaxScaler()
X_text_features_normalized = scaler.fit_transform(X_text_features.toarray())
df_normalized = pd.DataFrame(X_text_features_normalized, columns=[f'feature_{i}' for i in range(X_text_features_normalized.shape[1])])
df = pd.concat([df.reset_index(drop=True), df_normalized], axis=1)

#6.Data arrangement
sentiment_mapping = {'positive': 1, 'negative': -1}
df['sentiment'] = df['sentiment'].map(sentiment_mapping)
df = df.drop(columns=['text', 'previous_messages_dates', 'message_date',
                      'email', 'date_of_new_follower', 'date_of_new_follow',
                      'account_creation_date', 'message_length', 'num_messages_sent',
                      'follower_count', 'following_count', 'follower_following_diff',
                      'seniority', 'average_time_difference'])

ngram_column_indices = [ 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
df = df.drop(df.columns[ngram_column_indices], axis=1)
print(df)
#5.Feature selection-------------------------------------------------------->
#6.dimensionality reduction------------------------------------------------->

