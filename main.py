import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nltk import SnowballStemmer
import re
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_pickle(r"C:\Users\tamar\Downloads\XY_train.pkl")
print(df)

# 1.Preprocessing ----------------------------------------------------------------------------------------------->
def preprocess_data(df):
    # Drop rows with missing values exceeding a threshold
    df = df.dropna(thresh=df.shape[1] - 2)

    # Fill missing values for 'email' with 'unknown'
    df['email'] = df['email'].fillna('unknown')

    # Fill missing values for 'embedded_content' and 'platform' based on their probability distributions
    embedded_content_prob = df['embedded_content'].value_counts(normalize=True)
    platform_prob = df['platform'].value_counts(normalize=True)

    def impute_missing_values(row, prob_dist):
        if pd.isnull(row):
            return np.random.choice(prob_dist.index, p=prob_dist.values)
        else:
            return row

    df['embedded_content'] = df['embedded_content'].apply(lambda x: impute_missing_values(x, embedded_content_prob))
    df['platform'] = df['platform'].apply(lambda x: impute_missing_values(x, platform_prob))

    # Fill missing values for 'email_verified' and 'blue_tick'
    df['email_verified'].fillna(df['blue_tick'], inplace=True)
    df['blue_tick'].fillna(df['email_verified'], inplace=True)

    # Fill missing values for 'gender' randomly
    df['gender'].replace('None', np.nan, inplace=True)
    gender_counts = df['gender'].value_counts()
    df['gender'].fillna(pd.Series(np.random.choice(gender_counts.index, size=len(df.index),
                                                   p=(gender_counts / gender_counts.sum()))), inplace=True)

    # Delete the remaining rows with missing values
    df = df.dropna()

    # Convert 'message_date' to categorical and create 'message_time_category' column
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

    # handle changes for ngram
    # Make a copy of the DataFrame to avoid modifying the original
    df_processed = df.copy()

    # Lowercase
    df_processed['clean_text'] = df_processed['text'].str.lower()

    # Stemming
    stemmer = SnowballStemmer('english')
    df_processed['clean_text'] = df_processed['clean_text'].apply(
        lambda x: ' '.join(stemmer.stem(word) for word in x.split()))

    # Remove punctuation
    df_processed['clean_text'] = df_processed['clean_text'].apply(lambda x: re.sub(r'[^\w\s]', ' ', x))

    # Remove numbers
    df_processed['clean_text'] = df_processed['clean_text'].apply(
        lambda x: ' '.join(word for word in x.split() if not word.isdigit()))

    # Remove words with 1 letter
    df_processed['clean_text'] = df_processed['clean_text'].apply(lambda x: re.sub(r'\b\w{1}\b', '', x))

    # Remove top 0.05% of most common or not common words
    h_pct = 0.05
    l_pct = 0.05

    # Remove the top $h_pct of the most frequent words
    high_freq = pd.Series(' '.join(df_processed['clean_text']).split()).value_counts()[
                :int(pd.Series(' '.join(df_processed['clean_text']).split()).count() * h_pct / 100)]
    df_processed['clean_text'] = df_processed['clean_text'].apply(
        lambda x: ' '.join(word for word in x.split() if word not in high_freq))

    # Remove the top $l_pct of the least frequent words
    low_freq = pd.Series(' '.join(df_processed['clean_text']).split()).value_counts()[
               :-int(pd.Series(' '.join(df_processed['clean_text']).split()).count() * l_pct / 100):-1]
    df_processed['clean_text'] = df_processed['clean_text'].apply(
        lambda x: ' '.join(word for word in x.split() if word not in low_freq))

    # Remove double spaces
    df_processed['clean_text'] = df_processed['clean_text'].apply(lambda x: re.sub(r'\s+', ' ', x))

    return df_processed



# 2.Feature Extraction ---------------------------------------------------------------------------------------------->

def extract_features(df):
    # 1. Create a new column based on the length of messages
    df['message_length'] = df['text'].apply(lambda x: len(x))

    # 2. Create the number of messages sent by the user
    df['num_messages_sent'] = df['previous_messages_dates'].apply(len)

    # 3. Create the number of followers and following
    df['follower_count'] = df['date_of_new_follower'].apply(lambda x: len(x))
    df['following_count'] = df['date_of_new_follow'].apply(lambda x: len(x))

    # 4. Append new columns created to the df
    new_columns_df = df[['follower_count', 'following_count']]

    # 5. N-GRAM
    X = df['clean_text']
    ngram_vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=100)
    X_ngrams = ngram_vectorizer.fit_transform(X).toarray()
    df_output = pd.DataFrame(data=X_ngrams, columns=ngram_vectorizer.get_feature_names_out())

    # 6. Extract email domain endings
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

    # 7. Create seniority in years
    df['account_creation_date'] = pd.to_datetime(df['account_creation_date'])
    current_date = datetime.now()
    df['seniority'] = (current_date - df['account_creation_date']).dt.days / 365.25

    # 8. Create the average time difference between messages
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

    return df, new_columns_df, df_output


def test_extract_features(df, ngram_vectorizer):
    # 1. Create a new column based on the length of messages
    df['message_length'] = df['text'].apply(lambda x: len(x))

    # 2. Create the number of messages sent by the user
    df['num_messages_sent'] = df['previous_messages_dates'].apply(len)

    # 3. Create the number of followers and following
    df['follower_count'] = df['date_of_new_follower'].apply(lambda x: len(x))
    df['following_count'] = df['date_of_new_follow'].apply(lambda x: len(x))

    # 4. Append new columns created to the df
    new_columns_df = df[['follower_count', 'following_count']]

    # 5. N-GRAM
    X = df['clean_text']
    X_ngrams = ngram_vectorizer.transform(X).toarray()
    df_output = pd.DataFrame(data=X_ngrams, columns=ngram_vectorizer.get_feature_names_out())

    # 6. Extract email domain endings
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

    # 7. Create seniority in years
    df['account_creation_date'] = pd.to_datetime(df['account_creation_date'])
    current_date = datetime.now()
    df['seniority'] = (current_date - df['account_creation_date']).dt.days / 365.25

    # 8. Create the average time difference between messages
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

    return df, new_columns_df, df_output




# 3.Feature Representation -------------------------------------------------------------------------------------------->

def feature_representation(df, new_columns_df, df_output):
    # Normalize the values by dividing each column by its maximum value
    columns_to_normalize = ['message_length', 'num_messages_sent', 'follower_count', 'following_count', 'seniority']
    for column in columns_to_normalize:
        max_value = df[column].max()
        df[f'normalized_{column}'] = df[column] / max_value

    # One-hot coding
    email_domain_ending_onehot = pd.get_dummies(df['email_domain_ending'], prefix='email_ending')
    embedded_content_onehot = pd.get_dummies(df['embedded_content'], prefix='embedded_content')
    platform_onehot = pd.get_dummies(df['platform'], prefix='platform')
    message_time_category_onehot = pd.get_dummies(df['message_time_category'], prefix='message_time')
    df = pd.concat(
        [df, email_domain_ending_onehot, embedded_content_onehot, platform_onehot, message_time_category_onehot],
        axis=1)
    df.drop(['email_domain_ending', 'embedded_content', 'platform', 'message_time_category'], axis=1, inplace=True)

    # Convert one-hot coding to binary values
    bool_to_binary = {True: 1, False: 0}
    df['email_verified'] = df['email_verified'].map(bool_to_binary)
    df['blue_tick'] = df['blue_tick'].map(bool_to_binary)
    for column in df.columns:
        if df[column].dtype == bool:
            df[column] = df[column].astype(int)

    gender_to_binary = {'F': 1, 'M': 0}
    df['gender'] = df['gender'].map(gender_to_binary)

    # Normalization for the n-gram features
    scaler = MinMaxScaler()
    X_ngrams_normalized = scaler.fit_transform(df_output)
    df_normalized = pd.DataFrame(X_ngrams_normalized, columns=df_output.columns)
    df = pd.concat([df.reset_index(drop=True), df_normalized], axis=1)

    # Arrange data again
    sentiment_mapping = {'positive': 1, 'negative': -1}
    df['sentiment'] = df['sentiment'].map(sentiment_mapping)
    df = df.drop(columns=['text', 'previous_messages_dates', 'message_date',
                          'email', 'date_of_new_follower', 'date_of_new_follow',
                          'account_creation_date', 'message_length', 'num_messages_sent',
                          'follower_count', 'following_count',
                          'seniority', 'clean_text', 'average_time_difference'])

    return df,scaler , max_value

def test_feature_representation(df, new_columns_df, df_output,scaler , max_value):
    # Normalize the values by dividing each column by its maximum value
    columns_to_normalize = ['message_length', 'num_messages_sent', 'follower_count', 'following_count', 'seniority']
    for column in columns_to_normalize:
        df[f'normalized_{column}'] = df[column] / max_value

    # One-hot coding
    email_domain_ending_onehot = pd.get_dummies(df['email_domain_ending'], prefix='email_ending')
    embedded_content_onehot = pd.get_dummies(df['embedded_content'], prefix='embedded_content')
    platform_onehot = pd.get_dummies(df['platform'], prefix='platform')
    message_time_category_onehot = pd.get_dummies(df['message_time_category'], prefix='message_time')
    df = pd.concat(
        [df, email_domain_ending_onehot, embedded_content_onehot, platform_onehot, message_time_category_onehot],
        axis=1)
    df.drop(['email_domain_ending', 'embedded_content', 'platform', 'message_time_category'], axis=1, inplace=True)

    # Convert one-hot coding to binary values
    bool_to_binary = {True: 1, False: 0}
    df['email_verified'] = df['email_verified'].map(bool_to_binary)
    df['blue_tick'] = df['blue_tick'].map(bool_to_binary)
    for column in df.columns:
        if df[column].dtype == bool:
            df[column] = df[column].astype(int)

    gender_to_binary = {'F': 1, 'M': 0}
    df['gender'] = df['gender'].map(gender_to_binary)

    # Normalization for the n-gram features

    X_ngrams_normalized = scaler.transform(df_output)
    df_normalized = pd.DataFrame(X_ngrams_normalized, columns=df_output.columns)
    df = pd.concat([df.reset_index(drop=True), df_normalized], axis=1)

    # Arrange data again

    df = df.drop(columns=['text', 'previous_messages_dates', 'message_date',
                          'email', 'date_of_new_follower', 'date_of_new_follow',
                          'account_creation_date', 'message_length', 'num_messages_sent',
                          'follower_count', 'following_count',
                          'seniority', 'clean_text', 'average_time_difference'])

    return df


# 4.Feature Selection -------------------------------------------------------------------------------------------->

def feature_selection(df, k=20):
    # Extract the target variable 'sentiment'
    Y = df['sentiment']

    # Drop the 'sentiment' column
    X_cat = df.drop(columns=['sentiment'])

    # Save 'textID' column for later
    textID_column = X_cat['textID']

    # Drop 'textID' and 'sentiment' columns from the feature DataFrame
    X_cat = X_cat.drop(columns=['textID'])

    # Initialize SelectKBest with chi-squared as the scoring function and k=k
    chi2_features = SelectKBest(chi2, k=k)

    # Fit SelectKBest to the data and transform it
    X_cat_kbest = chi2_features.fit_transform(X_cat, Y)

    # Get the selected features
    selected_features = X_cat.columns[chi2_features.get_support()]

    # Create a DataFrame with selected features
    selected_features_df = pd.DataFrame(X_cat_kbest, columns=selected_features)

    # Add 'textID' column back to the selected features DataFrame
    selected_features_df['sentiment'] = df['sentiment']

    return selected_features_df

#----------------------------------PART B------------------------------------------------------------------------>
# Define dataset (X, y)
X = df.drop(columns=['sentiment'])
y = df['sentiment']


# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Merge X_train and y_train into one DataFrame for feature extraction
train_df = pd.concat([X_train, y_train], axis=1)

# Step 1: Preprocess the training data
train_processed = preprocess_data(train_df)
# Step 2: Extract features from the training data
train_features,new_columns_train, df_output_train = extract_features(train_processed)
ngram_vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=100)
X_train_ngrams = ngram_vectorizer.fit_transform(train_features['clean_text']).toarray()
# Step 3: Perform feature representation on training data
train_represented, scaler , max_value = feature_representation(train_features,new_columns_train, df_output_train )
# Step 4: Perform feature selection on training data
train_selected = feature_selection(train_represented)


# Step 1: Preprocess the test data
test_processed = preprocess_data(X_test)
# Step 2: Extract features from the test data
test_features,new_columns_test, df_output_test = test_extract_features(test_processed, ngram_vectorizer)
# Step 3: Perform feature representation on test data
test_represented = test_feature_representation(test_features,new_columns_test, df_output_test ,scaler , max_value)
# Step 4: Perform feature selection on test data based on the selected features from the training data
train_selected_columns_without_sentiment = train_selected.columns[train_selected.columns != 'sentiment']
test_selected = test_represented[train_selected_columns_without_sentiment]


# 5.Model Training and Evaluation ------------------------------------------------------------------------------------>

model = DecisionTreeClassifier(criterion='entropy')
model.fit(train_selected[train_selected.columns[train_selected.columns != 'sentiment']], train_selected['sentiment'])
plt.figure(figsize=(12, 10))
plot_tree(model, filled=True, feature_names=train_selected.columns[train_selected.columns != 'sentiment'])
plt.show()








