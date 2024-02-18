import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

df = pd.read_pickle(r"C:\Users\tamar\Downloads\XY_train.pkl")
print(df)

#______________________________________________________________________________________________
# for text
message_content = df['text'].values
# print(message_content[:5])
message_sizes = [len(message) for message in message_content]
# print(message_sizes[:5])
plt.hist(message_sizes, bins=50, color='blue', alpha=0.7)
plt.title('Distribution of Message Lengths')
plt.xlabel('Message Number')
plt.ylabel('Message Length')
plt.show()

#______________________________________________________________________________________________
# for sentiment
sentiment_counts = df['sentiment'].value_counts()
plt.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red'])
plt.title('Distribution of Message Sentiments')
plt.xlabel('Sentiment')
plt.ylabel('Number of Messages')
plt.xticks(sentiment_counts.index, ['Positive', 'Negative'])
plt.show()

#______________________________________________________________________________________________
#The amount of positive and negative messages that men sent versus the amount of positive and negative messages that women sent
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.countplot(x='gender', hue='sentiment', data=df, palette='pastel')
plt.title('Distribution of Positive and Negative Messages by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

#______________________________________________________________________________________________
#How many monthly users were created each month for each of the years 2013,2014,2015
# Convert 'account_creation_date' to datetime
df['account_creation_date'] = pd.to_datetime(df['account_creation_date'], errors='coerce')
# Check for any missing values after conversion
print(df['account_creation_date'].isnull().sum())
# Drop rows with missing dates if necessary
df = df.dropna(subset=['account_creation_date'])
# Extract the year and month from the 'account_creation_date'
df['year'] = df['account_creation_date'].dt.year
df['month'] = df['account_creation_date'].dt.month_name()
# Plot the number of new users created over the year, broken down by month
plt.figure(figsize=(12, 6))
sns.countplot(x='month', hue='year', data=df)
plt.title('Number of New Users Created Over the Year')
plt.xlabel('Month')
plt.ylabel('Number of New Users')
plt.xticks(rotation=45)
plt.legend(title='Year', loc='upper right')
plt.show()

#______________________________________________________________________________________________
# Assuming 'sentiment' and 'platform' are the columns in your DataFrame
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# Assuming 'sentiment' contains 'positive' and 'negative' values
sns.countplot(x="platform", hue="sentiment", data=df, palette="pastel")

plt.title("Number of Positive and Negative Messages for Each Social Network")
plt.xlabel("Social Network")
plt.ylabel("Count")
plt.show()

#_________________________________________________________________________

# Assuming 'sentiment' and 'text' are the columns in your DataFrame
positive_messages = df[df['sentiment'] == 'positive']
negative_messages = df[df['sentiment'] == 'negative']

# Plotting for positive messages
plt.figure(figsize=(10, 6))
plt.hist(positive_messages['text'].apply(len), bins=30, color='green', alpha=0.7)
plt.title('Distribution of Message Length for Positive Sentiments')
plt.xlabel('Message Length')
plt.ylabel('Number of Messages')
plt.show()

# Plotting for negative messages
plt.figure(figsize=(10, 6))
plt.hist(negative_messages['text'].apply(len), bins=30, color='red', alpha=0.7)
plt.title('Distribution of Message Length for Negative Sentiments')
plt.xlabel('Message Length')
plt.ylabel('Number of Messages')
plt.show()
#_______________________________________________________________________________________________

# Assuming these are your categorical variables
categorical_variables = ['sentiment', 'gender', 'email_verified', 'blue_tick']

# Load your dataset (assuming it has the columns mentioned)
# df = pd.read_pickle(r"C:\Users\tamar\Downloads\XY_train.pkl")

# Create a DataFrame with the cross-tabulation results
cross_tabulations = pd.DataFrame(index=categorical_variables, columns=categorical_variables)

for var1 in categorical_variables:
    for var2 in categorical_variables:
        if var1 != var2:
            crosstab_result = pd.crosstab(df[var1], df[var2])
            strength = crosstab_result.iloc[1, 1] / crosstab_result.sum().sum()
            cross_tabulations.loc[var1, var2] = strength  # Normalize by total count

# Convert the DataFrame to numeric values
cross_tabulations = cross_tabulations.apply(pd.to_numeric)

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cross_tabulations, annot=True, cmap='Blues', vmin=0, vmax=1)
plt.title('Strength of Relationships Between Categorical Variables')
plt.show()

#_______________________________________________________________________________________________________

# Create a cross-tabulation for sentiment and email_verified
cross_tab_sentiment_email = pd.crosstab(df['sentiment'], df['email_verified'])

# Plot the grouped bar plot
plt.figure(figsize=(8, 6))
sns.countplot(x='sentiment', hue='email_verified', data=df, palette='Set2')
plt.title('Relationship between Sentiment and Email Verification')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

#_______________________________________________________________________


# Encode 'text' and 'previous_messages_dates' into numerical values based on array size
df['text'] = df['text'].apply(lambda x: len(str(x)))
df['previous_messages_dates'] = df['previous_messages_dates'].apply(lambda x: len(x) if isinstance(x, list) else 0)

# Encode 'date_of_new_follower' and 'date_of_new_follow' into numerical values based on array size
df['date_of_new_follower'] = df['date_of_new_follower'].apply(lambda x: len(x) if isinstance(x, list) else 0)
df['date_of_new_follow'] = df['date_of_new_follow'].apply(lambda x: len(x) if isinstance(x, list) else 0)

# Encode 'sentiment' into numerical values (1 for positive, 0 for negative)
df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Print the updated DataFrame
print(df)

# Create a correlation matrix
correlation_matrix = df[['text', 'previous_messages_dates', 'date_of_new_follower', 'date_of_new_follow', 'sentiment']].corr()

# Set a larger figure size
plt.figure(figsize=(10, 8))

# Plot the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".5f")
plt.title('Correlation Matrix')
plt.show()

#_________________________________________________________________________________________________
#Gender Distribution:
# Replace None values with 'Missing'
df['gender'].fillna('Missing', inplace=True)

# Count the values for each gender
gender_counts = df['gender'].value_counts()

# Plotting the pie chart
plt.figure(figsize=(8, 6))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Gender Distribution')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

#_______________________________________________________________________________________________________
#FOT TEXT LENGTH

# Define a function to categorize message length
def categorize_message_length(message):
    if len(message) < 50:
        return 'Short'
    elif 50 <= len(message) <= 100:
        return 'Medium'
    else:
        return 'Long'

# Apply the function to categorize message length and create a new column
df['message_length_category'] = df['text'].apply(categorize_message_length)

# Plotting the count plot for message length categories and sentiment
plt.figure(figsize=(8, 6))
sns.countplot(x='message_length_category', hue='sentiment', data=df)
plt.title('Distribution of Sentiment across Message Length Categories')
plt.xlabel('Message Length Category')
plt.ylabel('Count')
plt.legend(title='Sentiment')
plt.show()
#_______________________________________________________________________________________
#previous message


# Assuming df is your DataFrame containing the data
# First, convert previous_messages_dates to the count of messages on each date
df['previous_messages_count'] = df['previous_messages_dates'].apply(len)

# Create a new DataFrame for positive sentiment
positive_df = df[df['sentiment'] == 'positive']

# Create a new DataFrame for negative sentiment
negative_df = df[df['sentiment'] == 'negative']

# Define colors
positive_color = 'skyblue'
negative_color = 'salmon'

# Plotting for positive sentiment
plt.figure(figsize=(10, 6))
sns.boxplot(x=positive_df['previous_messages_count'], color=positive_color)
plt.title('Range of Previous Messages Count for Positive Sentiment')
plt.xlabel('Count of Previous Messages Sent')
plt.ylabel('Frequency')
plt.show()

# Plotting for negative sentiment
plt.figure(figsize=(10, 6))
sns.boxplot(x=negative_df['previous_messages_count'], color=negative_color)
plt.title('Range of Previous Messages Count for Negative Sentiment')
plt.xlabel('Count of Previous Messages Sent')
plt.ylabel('Frequency')
plt.show()
#___________________________________________________________________________________________________

# Filter data for positive sentiment and create a copy
positive_df = df[df['sentiment'] == 'positive'].copy()

# Convert date column to datetime objects if they are not already
positive_df['account_creation_date'] = pd.to_datetime(positive_df['account_creation_date'])

# Extract year and month from account creation date
positive_df['year_month'] = positive_df['account_creation_date'].dt.to_period('M')

# Group by year-month and count the number of user creations
positive_counts = positive_df.groupby('year_month').size()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(positive_counts.index.astype(str), positive_counts.values, label='Positive Sentiment', color='green')
plt.title('Number of User Creations with Positive Sentiment Over Time')
plt.xlabel('Year-Month')
plt.ylabel('Number of User Creations')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# Filter data for negative sentiment and create a copy
negative_df = df[df['sentiment'] == 'negative'].copy()

# Convert date column to datetime objects if they are not already
negative_df['account_creation_date'] = pd.to_datetime(negative_df['account_creation_date'])

# Extract year and month from account creation date
negative_df['year_month'] = negative_df['account_creation_date'].dt.to_period('M')

# Group by year-month and count the number of user creations
negative_counts = negative_df.groupby('year_month').size()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(negative_counts.index.astype(str), negative_counts.values, label='Negative Sentiment', color='red')
plt.title('Number of User Creations with Negative Sentiment Over Time')
plt.xlabel('Year-Month')
plt.ylabel('Number of User Creations')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

#______________________________________________________________________________________


# Group by sentiment and blue_tick and count the number of occurrences
sentiment_blue_tick_counts = df.groupby(['sentiment', 'blue_tick']).size().unstack(fill_value=0)

# Plotting
sentiment_blue_tick_counts.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Distribution of Sentiment by User Authentication (Blue Tick)')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Blue Tick', loc='upper right')
plt.grid(axis='y')
plt.show()

#--------for the email domain ending--------
# Define a function to extract the domain ending from an email address
def extract_email_domain_ending(email):
    if pd.isnull(email):
        return 'Missing'
    # Extract the domain ending using regex
    match = re.search(r'\.(\w+)$', email)
    if match:
        return match.group(1)
    else:
        return 'Unknown'

# Apply the function to extract email domain endings
df['email_domain_ending'] = df['email'].apply(extract_email_domain_ending)

# Now count the unique email domain endings
email_domain_ending_counts = df['email_domain_ending'].value_counts()

print(email_domain_ending_counts)

# Adjust Pandas display settings to show all rows
pd.set_option('display.max_rows', None)

# Print email type counts
print(email_domain_ending_counts)

# Filter DataFrame for positive sentiment
positive_df = df[df['sentiment'] == 'positive']

# Filter DataFrame for negative sentiment
negative_df = df[df['sentiment'] == 'negative']

# Count unique email domain endings for positive sentiment
positive_email_domain_ending_counts = positive_df['email_domain_ending'].value_counts()

# Count unique email domain endings for negative sentiment
negative_email_domain_ending_counts = negative_df['email_domain_ending'].value_counts()

print("Positive Email Domain Endings:")
print(positive_email_domain_ending_counts)

print("\nNegative Email Domain Endings:")
print(negative_email_domain_ending_counts)


