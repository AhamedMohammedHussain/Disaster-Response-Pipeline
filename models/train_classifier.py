import sys
import pickle
import pandas as pd
from sqlalchemy import create_engine
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
nltk.download('stopwords')

def load_data(database_filepath):
    
    engine = create_engine('sqlite:///' + database_filepath)
    table_name=database_filepath[database_filepath.rfind("/")+1:-3]
    df= pd.read_sql_table(table_name, engine)
    
    X = df.message
    Y = df.iloc[:, 4:]
    
    category_names = list(df.columns[4:])
    
    return X, Y, category_names


def tokenize(text):
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
    stop_words = stopwords.words("english")
    
    #clean will contain unique words . not common words
    clean = []
    for token in tokens:
        if token not in stop_words:
            clean.append(lemmatizer.lemmatize(token))
    return clean

def build_model():
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [10, 20, 50],
        'clf__estimator__criterion': ['gini', 'entropy'],
        'clf__estimator__max_depth': [2, 5, None],
#        'clf__estimator__min_samples_leaf':[1, 5, 10],
    }

    grid = GridSearchCV(estimator=pipeline, param_grid=parameters , n_jobs=-1 , cv=2)

    return grid


def evaluate_model(model, X_test, Y_test, category_names):

    y_pred = model.predict(X_test)
    
    df = pd.DataFrame(classification_report(y_test, y_pred1, target_names=labels, output_dict=True)).T.reset_index()

    

def save_model(model, model_filepath):
    
    file = open(model_filepath, 'wb')
    pickle.dump(cv, file)
    file.close()


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