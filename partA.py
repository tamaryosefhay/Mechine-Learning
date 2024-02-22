# Machine Learning Project - PART A
# %% md
## Read Data
# %%
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

df = pd.read_pickle(r'/Users/maya/Documents/Information Systems/שנה ג׳/למידת מכונה/פרוייקט קורס/XY_train.pkl')
df.head()
# %% md
## EDA Code
# %% md
## Dataset Creation
# %% md
### 1. Pre-Processing
# %%
# ------------ HANDLE MISSING VALUES ---------------#

# Calculate missing values before deletion
missing_values_before = df.isnull().sum()
df = df.dropna(thresh=df.shape[1] - 2)
missing_values_after = df.isnull().sum()
missing_values_table = pd.DataFrame({
    'Before': missing_values_before,
    'After': missing_values_after,
    'Difference': missing_values_before - missing_values_after
})
print("Missing Values Before and After Deletion:")
print(missing_values_table)

# fill email with unknown
df['email'] = df['email'].fillna('unknown')
missing_values_count = df['email'].isnull().sum()

# fill embedded_content and platform by to their probability
embedded_content_prob = df['embedded_content'].value_counts(normalize=True)
platform_prob = df['platform'].value_counts(normalize=True)


def impute_missing_values(row, prob_dist):
    if pd.isnull(row):
        return np.random.choice(prob_dist.index, p=prob_dist.values)
    else:
        return row


df['embedded_content'] = df['embedded_content'].apply(lambda x: impute_missing_values(x, embedded_content_prob))
df['platform'] = df['platform'].apply(lambda x: impute_missing_values(x, platform_prob))

# fill missing values for email_verified and blue_tick
df['email_verified'].fillna(df['blue_tick'], inplace=True)
df['blue_tick'].fillna(df['email_verified'], inplace=True)

# fill gender randomly
df['gender'].replace('None', np.nan, inplace=True)
gender_counts = df['gender'].value_counts()
df['gender'].fillna(pd.Series(np.random.choice(gender_counts.index,
                                               size=len(df.index),
                                               p=(gender_counts / gender_counts.sum()))),
                    inplace=True)

# delete the remaining rows with missing values
df = df.dropna()

# ------------ HANDLE DATA CONVERSIONS ---------------#

# convert message_date to categorical
df['message_date'] = pd.to_datetime(df['message_date'])
df['hour'] = df['message_date'].dt.hour
morning_interval = range(6, 12)  # 6:00 AM to 11:59 AM
noon_interval = range(12, 18)  # 12:00 PM to 5:59 PM
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

# ------------ HANDLE CHANGES FOR N-GRAM  ---------------#

# lower case
lower = df['text'].apply(str.lower)

# stemming
from nltk.stem import SnowballStemmer

stem = SnowballStemmer('english')
stemmed = lower.apply(lambda x: ' '.join(stem.stem(word) for word in str(x).split()))

# remove punctuation
import re

rem_punc = stemmed.apply(lambda x: re.sub(r'[^\w\s]', ' ', x))

# remove numbers
rem_num = rem_punc.apply(lambda x: " ".join(x for x in x.split() if not x.isdigit()))

# remove words with 1 letter
rem_length1 = rem_num.apply(lambda x: re.sub(r'\b\w{1}\b', ' ', x))

# remove top 0.05% of most common or not common words
h_pct = 0.05
l_pct = 0.05

# remove the top $h_pct of the most frequent words
high_freq = pd.Series(''.join(rem_length1).split()).value_counts()[
            :int(pd.Series(' '.join(rem_length1).split()).count() * h_pct / 100)]

rem_high = rem_length1.apply(lambda x: " ".join(x for x in x.split() if x not in high_freq))

# remove the top $l_pct of the least frequent words
low_freq = pd.Series(''.join(rem_high).split()).value_counts()[
           :-int(pd.Series(' '.join(rem_high).split()).count() * l_pct / 100):-1]

rem_low = rem_high.apply(lambda x: " ".join(x for x in x.split() if x not in low_freq))

# remove double spaces
rem_punc = rem_low.apply(lambda x: re.sub(r'[^\w\s]', ' ', x))

df['clean_text'] = rem_punc

# Calculate missing values before deletion
missing_values_before = df.isnull().sum()
df = df.dropna(thresh=df.shape[1] - 2)
missing_values_after = df.isnull().sum()
missing_values_table = pd.DataFrame({
    'Before': missing_values_before,
    'After': missing_values_after,
    'Difference': missing_values_before - missing_values_after
})
print("Missing Values Before and After Deletion:")
print(missing_values_table)

# %% md
### 3. Feature Extraction
# %%
# 1. create a new column based on the length of messages
df['message_length'] = df['text'].apply(lambda x: len(x))

# 2. create the number of messages sent by the user
df['num_messages_sent'] = df['previous_messages_dates'].apply(len)

# 3. create num of followers and following
df['follower_count'] = df['date_of_new_follower'].apply(lambda x: len(x))
df['following_count'] = df['date_of_new_follow'].apply(lambda x: len(x))

# 4. append new columns created to the df
new_columns_df = df[['follower_count', 'following_count']]

# 5. N-GRAM

X = df['clean_text']
Y = df['sentiment']

ngram_vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=100)
X_ngrams = ngram_vectorizer.fit_transform(X).toarray()

# Check words frequency
# sum_of_words = X_ngrams.sum(axis = 0)
# words_freq = [(word, sum_of_words[i]) for word, i in ngram_vectorizer.vocabulary_.items()]
# words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
#
# print(words_freq)

df_output = pd.DataFrame(data=X_ngrams, columns=ngram_vectorizer.get_feature_names_out())


# 6. extract email domain endings
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

# 7. create seniority in years
df['account_creation_date'] = pd.to_datetime(df['account_creation_date'])
current_date = datetime.now()
df['seniority'] = (current_date - df['account_creation_date']).dt.days / 365.25


# 8. create the average time difference between messages
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
# %% md
### 4. Feature Representation
# %%
# 1. normalize the values by dividing each column by its maximum value
columns_to_normalize = ['message_length', 'num_messages_sent', 'follower_count', 'following_count', 'seniority']
for column in columns_to_normalize:
    max_value = df[column].max()
    df[f'normalized_{column}'] = df[column] / max_value
for column in columns_to_normalize:
    normalized_column_name = f'normalized_{column}'
    # print(f"Normalized {column}:")
    # print(df[normalized_column_name])

# 2. one-hot coding
email_domain_ending_onehot = pd.get_dummies(df['email_domain_ending'], prefix='email_ending')
embedded_content_onehot = pd.get_dummies(df['embedded_content'], prefix='embedded_content')
platform_onehot = pd.get_dummies(df['platform'], prefix='platform')
message_time_category_onehot = pd.get_dummies(df['message_time_category'], prefix='message_time')
df = pd.concat([df, email_domain_ending_onehot, embedded_content_onehot, platform_onehot, message_time_category_onehot],
               axis=1)
df.drop(['email_domain_ending', 'embedded_content', 'platform', 'message_time_category'], axis=1, inplace=True)

# 3. convert one hot coding to binary values
bool_to_binary = {True: 1, False: 0}
df['email_verified'] = df['email_verified'].map(bool_to_binary)
df['blue_tick'] = df['blue_tick'].map(bool_to_binary)

for column in df.columns:
    if df[column].dtype == bool:
        df[column] = df[column].astype(int)

gender_to_binary = {'F': 1, 'M': 0}
df['gender'] = df['gender'].map(gender_to_binary)

# 4. normalization for the n-gram features
scaler = MinMaxScaler()
X_ngrams_normalized = scaler.fit_transform(X_ngrams)
df_normalized = pd.DataFrame(X_ngrams_normalized, columns=ngram_vectorizer.get_feature_names_out())
df = pd.concat([df.reset_index(drop=True), df_normalized], axis=1)

# 6. arrange data again
sentiment_mapping = {'positive': 1, 'negative': -1}
df['sentiment'] = df['sentiment'].map(sentiment_mapping)
df = df.drop(columns=['text', 'previous_messages_dates', 'message_date',
                      'email', 'date_of_new_follower', 'date_of_new_follow',
                      'account_creation_date', 'message_length', 'num_messages_sent',
                      'follower_count', 'following_count',
                      'seniority', 'clean_text', 'average_time_difference'])

df.head()

# %%
# nan_rows = df[df['normalized_average_time_difference'].isna()]
# nan_rows

# %%
# Separate the dataset into positive and negative sentiment groups
positive_sentiment = df[df['sentiment'] == 1]
negative_sentiment = df[df['sentiment'] == -1]

# Remove 'sentiment', 'textID' columns as we are not considering it as a feature
features = df.drop(columns=['sentiment', 'textID']).columns

# print(features)

fisher_scores = {}

# Calculate Fisher score for each feature
for feature in features:
    mean_positive = positive_sentiment[feature].mean()
    mean_negative = negative_sentiment[feature].mean()
    var_positive = positive_sentiment[feature].var()
    var_negative = negative_sentiment[feature].var()

    # print(var_negative)
    # print(var_positive)

    fisher_score = ((mean_positive - mean_negative) ** 2) / (var_positive + var_negative)
    # print(feature)
    fisher_scores[feature] = round(fisher_score, 3)  # Round to 3 decimal places

# print(fisher_scores)

fisher_df = pd.DataFrame.from_dict(fisher_scores, orient='index', columns=['Fisher Score'])

# sort the features by highest to lowest
fisher_df = fisher_df.sort_values(by='Fisher Score', ascending=False)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

fisher_df
# %%
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np

# Assuming 'X_cat' is your feature matrix and 'Y' is your target variable

# Extract the target variable 'sentiment'
Y = df['sentiment']

# Drop the 'textID' column from the feature DataFrame
X_cat = df.drop(columns=['textID', 'sentiment'])

# Initialize SelectKBest with chi-squared as the scoring function and k=15
chi2_features = SelectKBest(chi2, k=15)

# Fit SelectKBest to the data and transform it
X_cat_kbest = chi2_features.fit_transform(X_cat, Y)

# Get the p-values of the features and round to 4 digits
p_values = chi2_features.pvalues_
p_values_rounded = np.round(p_values, 6)

# Get the selected features
selected_features = X_cat.columns[chi2_features.get_support()]

# Convert the array of rounded p-values into a DataFrame
p_values_df = pd.DataFrame({'Feature': X_cat.columns, 'P-value': p_values_rounded})

# Filter the selected features based on the support
selected_features_df = pd.DataFrame({'Feature': selected_features})

# Print the DataFrames
print("P-values DataFrame:")
print(p_values_df)

print("\nSelected Features DataFrame:")
selected_features_df

