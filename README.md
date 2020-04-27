# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ or http://localhost:3001/




#### NOTE: It takes almost 1 hr to train. Please train in local computer or in kaggle .
https://drive.google.com/open?id=1kXG1--5f2xtHzTkycy6mioRXkSx_6BOu
for skipping the training part. It works in kaggle. For me , it didn't work in local computer . It gives a warning when used in local computer.
