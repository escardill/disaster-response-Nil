"""
Classifier Trainer
Project: Disaster Response Pipeline (Udacity - Data Science Nanodegree)
Sample Script Syntax:
> python train_classifier.py <path to sqllite destination db> <path to the pickle file>
Sample Script Execution:
> python train_classifier.py ../data/NilMessages.db classifier.pkl
Arguments:
    1) Path to SQLite destination database (e.g. NilMessages.db)
    2) Path to pickle file name where ML model needs to be saved (e.g. classifier.pkl)
"""


# import libraries
import sys
import numpy as np
import pandas as pd
import re
from sqlalchemy import create_engine

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score, fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV

import pickle

def load_data(database_filepath):
    """
    Load Data from the Database Function

    Arguments:
        database_filepath -> Path to SQLite destination database (e.g. NilMessages.db)
    Output:
        X -> a dataframe containing features
        Y -> a dataframe containing labels
        category_names -> List of categories name
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM NilMessages', con=engine)

    # Remove child alone as it has all zeros only
    df = df.drop(['child_alone'], axis=1)

    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = list(Y.columns.values)

    return X, Y, category_names


def tokenize(text):
    """
     Tokenize the text function

     Arguments:
         text -> Text message which needs to be tokenized
     Output:
         clean_tokens -> List of tokens extracted from the provided text
     """

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    clean = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words("english")]
    tokens = [lemmatizer.lemmatize(w, pos='v').strip() for w in clean]

    return tokens

def build_model():
    """
    Build model function

    Output:
        A Scikit ML Pipeline that process text messages and apply a classifier.

    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT
        model: The model that is to be evaluated
        X_test: Input features, testing set
        y_test: Label features, testing set
        category_names: List of the categories
    OUTPUT
        This method does nto specifically return any data to its calling method.
        However, it prints out the precision, recall and f1-score
    '''
    y_pred = pipeline.predict(X_test)
    for i, category in enumerate(category_names):
        print("Category:", category,"\n", classification_report(y_test.iloc[:, i].values, y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category, accuracy_score(y_test.iloc[:, i].values, y_pred[:,i])))


def save_model(model, model_filepath):
    '''
    Saves the model to disk
    INPUT
        model: The model to be saved
        model_filepath: Filepath for where the model is to be saved
    OUTPUT
        While there is no specific item that is returned to its calling method, this method will save the model as a pickle file.
    '''
    temp_pickle = open(model_filepath, 'wb')
    pickle.dump(model, temp_pickle)
    temp_pickle.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()