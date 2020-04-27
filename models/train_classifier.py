"""
TRAIN CLASSIFIER
Disaster Response Project
Udacity - Data Scientist Nanodegree
Script Example
> python train_classifier.py ../data/DisasterResponse.db classifier.pkl
Arguments:
    1) SQLite db path (containing pre-processed data)
    2) pickle file name to save ML model
"""


import sys
import pickle
import pandas as pd
from sqlalchemy import create_engine
from joblib import parallel_backend
import joblib
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, BaseEstimator, TfidfTransformer
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report, accuracy_score
import re
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words'])

stop_words = stopwords.words("english")


def load_data(database_filepath):
    
	"""
    Load Data Function
    
    Arguments:
        database_filepath -> path to SQLite db
    Output:
        X -> feature
        Y -> label
        category_names -> used for data visualization (app)
    """
	
    engine = create_engine('sqlite:///' + database_filepath)
    table_name=database_filepath[database_filepath.rfind("/")+1:-3]
    df= pd.read_sql_table(table_name, engine)
    
    X = df.message
    Y = df.iloc[:, 4:]
    
    category_names = list(Y.columns)
    print(category_names)
    
    return X, Y, category_names


def tokenize(text):
	"""
    Tokenize function
    
    Arguments:
        text -> list of text messages (english)
    Output:
        clean_tokens -> tokenized text, clean for ML modeling
    """
	
    #converts upper to lower
    # takes only alphabets and numbers
    #replaces urls with string "urlplaceholder"
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex,text)
    for url in detected_urls:
        text = text.replace(url,"urlplaceholder")
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    lemmatizer = WordNetLemmatizer()
    #stopswords are common words
    
    #clean will contain unique words . not common words
    clean = []
    for token in tokens:
        if token not in stop_words:
            clean.append(lemmatizer.lemmatize(token))
    return clean

def build_model():

	"""
    Build Model function
    
    This function builds a model for pipeline
    """
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [10, 20, 50],
        'clf__estimator__criterion': ['gini', 'entropy'],
        'clf__estimator__max_depth': [2, 5, None],
        'clf__estimator__min_samples_leaf':[1, 5, 10],
    }

    grid = GridSearchCV(estimator=pipeline, param_grid=parameters ,verbose=1)#, n_jobs=-1 , cv=2)

    return grid

def evaluate_model(model, X_test, Y_test, category_names):

	"""
    Evaluate Model function
    
    This function applies ML pipeline to a test set and prints out model performance (classification report)
	
    Arguments:
        model -> Scikit ML Pipeline
        X_test -> test features
        Y_test -> test labels
        category_names -> label names (multi-output)
    """

    Y_pred = model.predict(X_test)
    #print(classification_report(Y_test, Y_pred, target_names=category_names))
    #for i in range(Y_test.shape[1]):
    #    print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:, i].values, y_pred[:,i])))
    labels = category_names
    for i in range(36):
        print(category_names[i])
        print(classification_report(Y_test[:, i], Y_pred[:,i]))
    

def save_model(model, model_filepath):

	"""
    Save Model function
    
    This function saves trained model as Pickle file, to be used in real time scenario.
    
    Arguments:
        model -> GridSearchCV or Scikit Pipelin object
        model_filepath -> destination path to save .pkl file
    
    """
    
    file = open(model_filepath, 'wb')
    pickle.dump(model, file)
    file.close()



def main():

	"""
    Train Classifier Main function
    
    This function applies the Machine Learning Pipeline:
        1) Extract data from SQLite db
        2) Train ML model on training set
        3) Estimate model performance on test set
        4) Save trained model as Pickle
    
	Since the training takes long time , 
	if there is pre-trained variables (model) 
	after each step in the model directory
	, it takes those and skips the steps.
    """
	
	
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        intermediate_saving = model_filepath[:model_filepath.rfind("/")+1]

        
        save_after_build=intermediate_saving+"save_after_model_build.pkl"
        try:
            model = joblib.load(save_after_build)
            print('Loading model...')
        except:

            print('Building model...')
            model = build_model()

            save_after_build=intermediate_saving+"save_after_model_build.pkl"
            print('Saving model after building...\n    MODEL: {}'.format(save_after_build))
            save_model(model, save_after_build)


    
        save_after_fit=intermediate_saving+"save_after_fit.pkl"
        try:
            model = joblib.load(save_after_fit)
            print('Loading model...')

        except:
            print('Training model...')
            with parallel_backend('multiprocessing'):
                model.fit(X_train, Y_train)


            print('Saving model after fitting...\n    MODEL: {}'.format(save_after_fit))
            save_model(model, save_after_fit)


            
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test.values, category_names)

        print('Saving model after evaluating...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
    
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
