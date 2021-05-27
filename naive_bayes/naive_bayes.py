import nltk
from nltk.stem.porter import PorterStemmer
import pandas as pd
import pdb
import pickle
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


FILE = '../datasets/train_formatted.csv'

# Classes
NEGATIVE = 1
POSITIVE = 2

MAX_FEATURES = 2500


def split_dataset(df):
    """ Divide the dataset into training and test data """

    X = df.text
    y = df.polarity

    return train_test_split(X, y, test_size=0.2)


def get_x_y(df):
    """ Get features (X) and labels (y) """

    X = df.text
    y = df.polarity

    return X, y


def tokenize(text):
    # Apply Porter stemmer to review text
    words = re.sub(r"[^A-Za-z0-9\-]", " ", text).lower().split()
    words = [PorterStemmer().stem(word) for word in words]
    return words


def get_number_of_features(X, vect):
    """ Get number of unique words used in training the dataset """

    # num_features = len(list(zip(X, vect.get_feature_names())))
    num_features = len(vect.get_feature_names())
    print("Total number of features (words): %s" % num_features)

    # Get vocabulary
    vocab = len(vect.vocabulary_)
    print("Number of words in vocab: %s" % vocab)


def count_vectorizer(X_train, y_train, naive_bayes):
    """ Get count matrix for words in review text """

    # Default params tokenizes and converts to lowercase
    # vect = CountVectorizer(max_features=MAX_FEATURES, binary=True)
    vect = CountVectorizer(binary=True)

    # Creates a csr_matrix
    X_vect_train = vect.fit_transform(X_train)

    naive_bayes.fit(X_vect_train, y_train)

    return vect, X_vect_train, naive_bayes


def tfidf_vectorizer(X_train, y_train, naive_bayes):
    """ Get normalised count matrix for words in review text """

    tfidf_vect = TfidfVectorizer(tokenizer=tokenize, sublinear_tf=True)

    X_vect_train = tfidf_vect.fit_transform(X_train)

    naive_bayes.fit(X_vect_train, y_train)

    return tfidf_vect, X_vect_train, naive_bayes


def pickle_result(result, filename):
    """ Pickle the result and save to a file """

    pickle.dump(result, open('roc_'+filename+".p", "wb"))


def get_results(X_test):
    """ Get predictions and the various accuracy scores/metrics for model """

    X_vect_test = vect.transform(X_test)
    y_pred = naive_bayes.predict(X_vect_test)

    y_scores = naive_bayes.predict_proba(X_vect_test)
    fpr, tpr, _ = metrics.roc_curve(y_test, y_scores[:, 1])
    auc = metrics.auc(fpr, tpr)

    print('AUC: %s' % auc)
    print(metrics.classification_report(y_test, y_pred))
    print("Accuracy: {:.2f}%".format(metrics.accuracy_score(y_test, y_pred) * 100))
    print("Confusion matrix:\n", metrics.confusion_matrix(y_test, y_pred))

    # If you want to save the results to a file uncomment lines belows
    # pickle_result(fpr, 'fpr')
    # pickle_result(tpr, 'tpr')
    # pickle_result(_, 'thresholds')


# Read in csv with formatted data
df = pd.read_csv(FILE, encoding='utf-8')
df.polarity.replace({1:0}, inplace=True)
df.polarity.replace({2:1}, inplace=True)

# Option 1: Split training dataset into train and test
# X_train, X_test, y_train, y_test = get_x_y(df)

# Option 2: Use full training dataset for training and test set for testing
X_train, y_train = get_x_y(df)

# Instantiate model
naive_bayes = MultinomialNB()

# Option 1: Create count matrix out of words in review texts
vect, X_vect_train, naive_bayes = count_vectorizer(X_train, y_train, naive_bayes)
# Option 2: Create normalised count matrix out of words in review texts
# vect, X_vect_train, naive_bayes = tfidf_vectorizer(X_train, y_train, naive_bayes)

get_number_of_features(X_train, vect)

# pickle.dump(naive_bayes, open('naive_bayes_tfidf_porter.sav', 'wb'))
pickle.dump(naive_bayes, open('naive_bayes.sav', 'wb'))

# Get final accuracy of model using test dataset
df_test = pd.read_csv('../datasets/test_formatted.csv', encoding='utf-8')
df_test.polarity.replace({1:0}, inplace=True)
df_test.polarity.replace({2:1}, inplace=True)
X_test, y_test = get_x_y(df_test)

get_results(X_test)
