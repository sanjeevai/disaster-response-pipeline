# Data Scientist Nanodegree

## Data Engineering

## Project: Disaster Response Pipeline

### Project Overview

In this project, I'll apply data engineering to analyze disaster data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages.

_data_ directory contains a data set which are real messages that were sent during disaster events. I will be creating a machine learning pipeline to categorize these events so that appropriate disaster relief agency can be reached out for help.

This project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

[Here](#eg) are a few screenshots of the web app.

### Project Components

There are three components of this project:

1. ETL Pipeline

File _data/process_data.py_ contains data cleaning pipeline that:

- Loads the `messages` and `categories` dataset
- Merges the two datasets
- Cleans the data
- Stores it in a **SQLite database**

2. ML Pipeline

File _models/train_classifier.py_ contains machine learning pipeline that:

- Loads data from the **SQLite database**
- Splits the data into training and testing sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs result on the test set
- Exports the final model as a pickle file

3. Flask Web App

<a id='eg'></a>

Running [this command](#com) **from app directory** will start the web app where users can enter their query, i.e., a request message sent during a natural disaster, e.g. _"Please, we need tents and water. We are in Silo, Thank you!"_.

**_Screenshot 1_**

![master](img/master.jpg)

What the app will do is that it will classify the text message into categories so that appropriate relief agency can be reached out for help.

**_Screenshot 2_**

![results](img/res.jpg)

### Running

There are three steps to get up and runnning with the web app if you want to start from ETL process.

1. Data Cleaning

**Go to the project directory** and the run the following command:

```bat
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

The first two arguments are input data and the third argument is the SQLite Database in which we want to save the cleaned data. The ETL pipeline is in _process_data.py_.

_DisasterResponse.db_ already exists in _data_ folder but the above command will still run and replace the file will same information. 

**_Screenshot 3_**
![process_data](img/process_data.png)
2. Training Classifier

After the data cleaning process, run this command **from the project directory**:

```bat
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```
This will use cleaned data to train the model, improve the model with grid search and saved the model to a pickle file (_classifer.pkl_).

_classifier.pkl_ already exists but the above command will still run and replace the file will same information.

It took me around **4 minutes** to train the classifier with grid search.

3. Starting the web app

Now that we have cleaned the data and trained our model. Now it's time to see the prediction in a user friendly way.

**Go the app directory** and run the following command:

<a id='com'></a>

```bat
python run.py
```

This will start the web app and will direct you to a URL where you can enter messages and get classification results for it.

An example is mentioned [above](#eg)

### Conclusion

Data is highly imbalanced. Though the accuracy metric is high (you will see the exact value after the model is trained by grid search, it is ~0.94), it has a poor value for recall (~0.6). So, take appropriate measures when using this model for decision-making process at a larger scale or in a production environment.

### Files

<pre>
.
|
+-app
| |
| |
| +-static
| | |
| | +-+favicon.ico---------------------# FAVICON FOR THE WEB APP
| |
| +-template
| | |
| | +-+go.html-------------------------# CLASSIFICATION RESULT PAGE OF WEB APP
| | +-+master.html---------------------# MAIN PAGE OF WEB APP
| |
| +-+run.py----------------------------# FLASK FILE THAT RUNS APP
|
+-data
| |
| +-+disaster_categories.csv-----------# DATA TO PROCESS
| +-+disaster_messages.csv-------------# DATA TO PROCESS
| +-+DisasterResponse.db---------------# DATABASE TO SAVE CLEAN DATA TO
| +-+process_data.py-------------------# PERFORMS ETL PROCESS
|
+-img
| |
| +-+master.jpg------------------------# A SCREENSHOT OF THE MAIN PAGE
| +-+res.jpg---------------------------# A SCREENSHOT OF THE CLASSIFICATION PAGE
|
+-models
| |
| +-+classifier.pkl--------------------# SAVED MODEL
| +-+train_classifier.py---------------# PERFORMS CLASSIFICATION TASK

</pre>

### Libraries

This project uses Python 3.6.6 and the necessary libraries are mentioned in _requirements.txt_.
The standard libraries which are not mentioned in _requirements.txt_ are _pickle_, _pprint_, _re_, _sys_, _time_ and _warnings_.

### Credits and Acknowledgements

Thanks [Udacity](https://www.udacity.com) for letting me use their logo as favicon for this web app.

Another [blog post](https://medium.com/udacity/three-awesome-projects-from-udacitys-data-scientist-program-609ff0949bed) was a great motivation to improve my documentation. This post shows some of the cool projects from [Data Scientist Nanodegree]() students. This really shows how far we can go if we dive deep into the project.