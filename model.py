import random
from nltk import classify, NaiveBayesClassifier
from nltk.corpus import twitter_samples
from nltk.tokenize import word_tokenize  
from preprocessing import remove_noise, get_tweets_for_model 
def load_data():
    positive_tweets = twitter_samples.tokenized('positive_tweets.json')
    negative_tweets = twitter_samples.tokenized('negative_tweets.json')

    return positive_tweets, negative_tweets

def prepare_datasets(positive_tweet_tokens, negative_tweet_tokens, stop_words):
    positive_cleaned_tokens_list = [remove_noise(tokens, stop_words) for tokens in positive_tweet_tokens]
    negative_cleaned_tokens_list = [remove_noise(tokens, stop_words) for tokens in negative_tweet_tokens]

    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)  
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

    positive_dataset = [(tweet_dict, "Positive") for tweet_dict in positive_tokens_for_model]
    negative_dataset = [(tweet_dict, "Negative") for tweet_dict in negative_tokens_for_model]

    dataset = positive_dataset + negative_dataset
    random.shuffle(dataset)

    return dataset

def train_model(dataset):
    train_data = dataset[:5000]
    classifier = NaiveBayesClassifier.train(train_data)

    return classifier

def classify_tweet(custom_tweet, classifier):
    custom_tokens = remove_noise(word_tokenize(custom_tweet))  # word_tokenize is now imported
    features = dict([token, True] for token in custom_tokens)
    return classifier.classify(features)

def evaluate_model(classifier, dataset):
    test_data = dataset[5001:]
    return classify.accuracy(classifier, test_data)
