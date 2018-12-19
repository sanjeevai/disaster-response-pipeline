# Data Scientist Nanodegree

## Data Engineering

## Project: Disaster Response Pipeline

### Project Overview
ETL pipeline combined with supervised learning to classify text messages sent during a disaster event

### Conclusion

### Files

<pre>
.
|
+-app
|  |
|  +-template
|  | |
|  | +-+master.html # main page of web app
|  | +-+go.html # classification result page of web app
|  +-+run.py # Flask file that runs app
+-data
| |
| +-+disaster_categories.csv # data to process
| +-+disaster_messages.csv # data to process
| +-+process_data.py # performs ETL process
| +-+DisasterResponse.db # database to save clean data to
+-models
| |
| +-+train_classifier.py # performs supervised learning
| +-+classifier.pkl # saved model
</pre>

### Libraries