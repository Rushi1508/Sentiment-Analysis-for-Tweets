import re
import string
import tkinter as tk
import random

import nltk
from nltk import classify, NaiveBayesClassifier
from nltk.corpus import twitter_samples
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.tag import pos_tag
from tkinter import scrolledtext

# Download necessary NLTK resources if not already downloaded
#nltk.download('all')
#nltk.download('wordnet')

positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tag(tokens)]
    return lemmatized_sentence

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('N'):
        return 'n'
    elif pos_tag.startswith('V'):
        return 'v'
    elif pos_tag.startswith('R'):
        return 'r'
    else:
        return 'a'

def remove_noise(tweet_tokens, stop_words=()):
    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

stop_words = stopwords.words('english')

positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

positive_cleaned_tokens_list = [remove_noise(tokens, stop_words) for tokens in positive_tweet_tokens]
negative_cleaned_tokens_list = [remove_noise(tokens, stop_words) for tokens in negative_tweet_tokens]

def classify_tweet(custom_tweet):
    custom_tokens = remove_noise(word_tokenize(custom_tweet))
    features = dict([token, True] for token in custom_tokens)
    sentiment = classifier.classify(features)
    return sentiment

def analyze():
    custom_tweet = input_text.get("1.0", "end-1c")
    sentiment = classify_tweet(custom_tweet)
    result_text.config(state=tk.NORMAL)
    result_text.delete("1.0", tk.END)
    result_text.insert(tk.END, f"Sentiment: {sentiment}\n")
    result_text.config(state=tk.DISABLED)

def get_all_words(cleaned_tokens_list):
    return (token for tokens in cleaned_tokens_list for token in tokens)

all_pos_words = get_all_words(positive_cleaned_tokens_list)
freq_dist_pos = FreqDist(all_pos_words)

def get_tweets_for_model(cleaned_tokens_list):
    return (dict([token, True] for token in tweet_tokens) for tweet_tokens in cleaned_tokens_list)

positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

positive_dataset = [(tweet_dict, "Positive") for tweet_dict in positive_tokens_for_model]
negative_dataset = [(tweet_dict, "Negative") for tweet_dict in negative_tokens_for_model]

dataset = positive_dataset + negative_dataset
random.shuffle(dataset)

train_data = dataset[:5000]
test_data = dataset[5001:]

classifier = NaiveBayesClassifier.train(train_data)

print("Accuracy is:", classify.accuracy(classifier, test_data))
print(classifier.show_most_informative_features(5))

def display_accuracy():
    accuracy = classify.accuracy(classifier, test_data)
    accuracy_text.config(state=tk.NORMAL)
    accuracy_text.delete("1.0", tk.END)
    accuracy_text.insert(tk.END, f"Accuracy: {accuracy:.2%}\n")
    accuracy_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    window = tk.Tk()
    window.title("Sentiment Analysis GUI")

    # Input Text Area
    input_label = tk.Label(window, text="Enter a tweet:")
    input_label.pack(pady=10)
    input_text = scrolledtext.ScrolledText(window, height=5, wrap=tk.WORD)
    input_text.pack(pady=5)

    # Analyze Button
    analyze_button = tk.Button(window, text="Analyze", command=analyze)
    analyze_button.pack(pady=5)

    # Result Text Area
    result_label = tk.Label(window, text="Sentiment Analysis Result:")
    result_label.pack()
    result_text = scrolledtext.ScrolledText(window, height=3, wrap=tk.WORD)
    result_text.pack(pady=5)
    result_text.config(state=tk.DISABLED)

    accuracy_button = tk.Button(window, text="Display Accuracy", command=display_accuracy)
    accuracy_button.pack(pady=5)

    # Accuracy Text Area
    accuracy_label = tk.Label(window, text="Classifier Accuracy:")
    accuracy_label.pack()
    accuracy_text = scrolledtext.ScrolledText(window, height=1, wrap=tk.WORD)
    accuracy_text.pack()
    accuracy_text.config(state=tk.DISABLED)

    # Run the GUI loop
    window.mainloop()
