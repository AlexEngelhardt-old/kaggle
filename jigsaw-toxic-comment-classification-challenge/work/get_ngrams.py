from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from scipy.sparse import hstack

def get_ngrams(train_text, test_text, Tfidf = False, chars = False):
    all_text = pd.concat([train_text, test_text])

    if Tfidf:
        vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{2,}',
            ngram_range=(1, 2),  # min/max ngram
            max_features=10000)  # the top 10000 *sorted by appearance*
        char_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='char',
            token_pattern=r'\w{2,}',
            ngram_range=(2, 5),
            max_features=20000)
    else:
        vectorizer = CountVectorizer(
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{2,}',
            ngram_range=(1, 2),
            max_features=10000)
        char_vectorizer = CountVectorizer(
            strip_accents='unicode',
            analyzer='char',
            token_pattern=r'\w{2,}',
            ngram_range=(2, 5),
            max_features=20000)

    vectorizer.fit(all_text)
    train_features = vectorizer.transform(train_text)
    test_features = vectorizer.transform(test_text)
    vectorizers = vectorizer
    
    if chars:
        char_vectorizer.fit(all_text)
        train_char_features = char_vectorizer.transform(train_text)
        test_char_features = char_vectorizer.transform(test_text)
        train_features = hstack([train_features, train_char_features])
        test_features = hstack([test_features, test_char_features])
        vectorizers = [vectorizer, char_vectorizer]

    return [train_features, test_features, vectorizers]
