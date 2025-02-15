# Disaster Response Pipelines


### Table of Contents

1. [Installation](#installation)
2. [Project Descriptions](#descriptions)
3. [Files Descriptions](#files)
4. [Instructions](#instructions)

## Installation <a name="installation"></a>

All libraries are available in Anaconda distribution of Python. The used libraries are:

- pandas
- re
- sys
- json
- sklearn
- nltk
- sqlalchemy
- pickle
- Flask
- plotly
- sqlite3

The code should run using Python versions 3.*.


## Project Descriptions<a name = "descriptions"></a>
The project has three componants which are:

1. **ETL Pipeline:** `process_data.py` file contain the script to create ETL pipline which:

ETL Pipeline In a Python script, process_data.py, write a data cleaning pipeline that: Loads the messages and categories datasets Merges the two datasets Cleans the data Stores it in a SQLite database

2. **ML Pipeline:** `train_classifier.py` file contain the script to create ML pipline which:

ML Pipeline In a Python script, train_classifier.py, write a machine learning pipeline that: Loads data from the SQLite database Splits the dataset into training and test sets Builds a text processing and machine learning pipeline Trains and tunes a model using GridSearchCV Outputs results on the test set Exports the final model as a pickle file

3. **Flask Web App:** the web app enables the user to enter a disaster message, and then view the categories of the message.

The web app also contains one visualization that describe the data.



## Files Descriptions <a name="files"></a>

The files structure is arranged as below:

	- README.md: read me file

  - \app
			- run.py: flask file to run the app
  - \data
			- disaster_categories.csv: categories dataset
			- disaster_messages.csv: messages dataset
      - process_data.py: ETL process
	- \models
			- train_classifier.py: classification code

## Instructions <a name="instructions"></a>

To execute the app follow the instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
