from nltk import FreqDist
from preprocessing import get_all_words

def display_most_informative_features(classifier, n=5):
    return classifier.show_most_informative_features(n)

def get_word_frequencies(cleaned_tokens_list):
    all_words = get_all_words(cleaned_tokens_list)
    return FreqDist(all_words)

