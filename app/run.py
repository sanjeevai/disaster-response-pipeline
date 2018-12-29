# imports

from collections import Counter
import json, plotly
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
import numpy as np
import operator
from plotly.graph_objs import Bar
from pprint import pprint
import re
from sklearn.externals import joblib
from sqlalchemy import create_engine

# initializing Flask app
app = Flask(__name__)

def tokenize(text):
    """
    Tokenizes text data

    Args:
    text str: Messages as text data

    Returns:

    # clean_tokens list: Processed text after normalizing, tokenizing and lemmatizing
    words list: Processed text after normalizing, tokenizing and lemmatizing
    """
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    words = word_tokenize(text)
    
    # remove stop words
    stopwords_ = stopwords.words("english")
    words = [word for word in words if word not in stopwords_]
    
    # extract root form of words
    words = [WordNetLemmatizer().lemmatize(word, pos='v') for word in words]

    return words

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message'] # message count based\
                                                          # on genre
    genre_names = list(genre_counts.index)                # genre names
    cat_p = df[df.columns[4:]].sum()/len(df)              # proportion based on\
                                                          # categories
    cat_p = cat_p.sort_values(ascending = False)          # largest bar will be\
                                                          # on left
    cats = list(cat_p.index)                              # category names

    words_with_repetition=[]                              # will contain all\
                                                          # words words with\
                                                          # repetition
    for text in df['message'].values:
        tokenized_ = tokenize(text)
        words_with_repetition.extend(tokenized_)

    word_count_dict = Counter(words_with_repetition)      # dictionary\
                                                          # containing word\
                                                          # count for all words
    
    sorted_word_count_dict = dict(sorted(word_count_dict.items(),
                                         key=operator.itemgetter(1),
                                         reverse=True))   # sort dictionary by\
                                                          # values
    top, top_10 =0, {}

    for k,v in sorted_word_count_dict.items():
        top_10[k]=v
        top+=1
        if top==10:
            break
    words=list(top_10.keys())
    pprint(words)
    count_props=100*np.array(list(top_10.values()))/df.shape[0]
    # create visuals
    figures = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=cats,
                    y=cat_p
                )
            ],

            'layout': {
                'title': 'Proportion of Messages <br> by Category',
                'yaxis': {
                    'title': "Proportion",
                    'automargin':True
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -40,
                    'automargin':True
                }
            }
        },
        {
            'data': [
                Bar(
                    x=words,
                    y=count_props
                )
            ],

            'layout': {
                'title': 'Frequency of top 10 words <br> as percentage',
                'yaxis': {
                    'title': 'Occurrence<br>(Out of 100)',
                    'automargin': True
                },
                'xaxis': {
                    'title': 'Top 10 words',
                    'automargin': True
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["figure-{}".format(i) for i, _ in enumerate(figures)]
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly figures
    return render_template('master.html', ids=ids, figuresJSON=figuresJSON, data_set=df)

# web page that handles user query and displays model results
@app.route('/go')

def go():

    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template('go.html',
                            query=query,
                            classification_result=classification_results
                          )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()