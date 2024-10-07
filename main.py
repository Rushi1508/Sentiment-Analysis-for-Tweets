import random
import nltk
import streamlit as st
from nltk import classify, NaiveBayesClassifier
from nltk.corpus import twitter_samples
from nltk.tokenize import word_tokenize
from preprocessing import remove_noise, get_tweets_for_model  # Ensure these functions are defined in preprocessing.py

# Download necessary NLTK resources if not already downloaded
nltk.download('twitter_samples')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load data
positive_tweets = twitter_samples.tokenized('positive_tweets.json')
negative_tweets = twitter_samples.tokenized('negative_tweets.json')

# Preprocess the datasets
stop_words = nltk.corpus.stopwords.words('english')

def prepare_datasets():
    positive_cleaned_tokens_list = [remove_noise(tokens, stop_words) for tokens in positive_tweets]
    negative_cleaned_tokens_list = [remove_noise(tokens, stop_words) for tokens in negative_tweets]

    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

    positive_dataset = [(tweet_dict, "Positive") for tweet_dict in positive_tokens_for_model]
    negative_dataset = [(tweet_dict, "Negative") for tweet_dict in negative_tokens_for_model]

    dataset = positive_dataset + negative_dataset
    random.shuffle(dataset)

    return dataset

# Train the model
def train_model(dataset):
    train_data = dataset[:5000]
    classifier = NaiveBayesClassifier.train(train_data)
    return classifier

# Classify a custom tweet
def classify_tweet(custom_tweet, classifier):
    custom_tokens = remove_noise(word_tokenize(custom_tweet))
    features = dict([token, True] for token in custom_tokens)
    return classifier.classify(features)

# Prepare the datasets and train the model
dataset = prepare_datasets()
classifier = train_model(dataset)

# Streamlit app
def main():
    st.title("Sentiment Analysis of Tweets")

    # Input Text Area
    custom_tweet = st.text_area("Enter a tweet:")

    # Analyze Button
    if st.button("Analyze"):
        if custom_tweet:
            sentiment = classify_tweet(custom_tweet, classifier)
            st.write(f"Sentiment: **{sentiment}**")
            
            # Display Classifier Accuracy after analysis
            accuracy = classify.accuracy(classifier, dataset[5001:])
            st.write(f"Classifier Accuracy: **{accuracy:.2%}**")
        else:
            st.write("Please enter a tweet for analysis.")

from nltk import classify
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Assuming you have X_test (features) and y_test (true labels) ready
# Convert X_test into the format required for NLTK classifier
def prepare_data_for_classification(data):
    return [dict([(token, True) for token in tokens]) for tokens in data]

# Prepare your test data
X_test_prepared = prepare_data_for_classification(X_test)

# Make predictions for each instance in the test set
y_pred = [classifier.classify(features) for features in X_test_prepared]

# You might need to map back your predictions if they are not already in the correct format
# For example, if your classifier predicts strings and y_test contains 0 and 1:
# y_pred_numeric = [1 if label == "Positive" else 0 for label in y_pred]

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")

# Classification Report
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)



if __name__ == "__main__":
    main()
