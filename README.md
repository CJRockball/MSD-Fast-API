# MSD-Fast-API
Using an XGBoost model to predict genre of a subset of the million song dataset. The main can be run as is, pkl-files of data pipeline and model are included. If you want to over-write the pickel file, just run the *xgb_model_train*. A small database is set up and preloaded with 100 songs from the training dataset. If you want to reset the database, run the *set_up_db file*. The genre prediction confidence for these songs will be 1 since we already know it. The api features predictions, a sample of the latest commited songs, a sample of a specified genre.

## Run
### To Run from File
* Clone the repo
* In a terminal go to folder and run `uvicorn main:app --reload`
* open browser choose song to predict from the data test file by adding a number between 1 and 428, `127.0.0.1:8000/predict/?q=88` the predicted song will be added to a database.
  
### To Run from Docker
* Clone the repo
* In a terminal go to folder and run `docker build -t name .` 
* Then run `docker run -dp 8000:8000 name`
* open browser choose song to predict from the data test file by adding a number between 1 and 428, `127.0.0.1:8000/predict/?q=88` the predicted song will be added to a database

## Functions
* `127.0.0.1:8000/` blank page
* `127.0.0.1:8000/predict/?q=88` change the number (between 1 and 428) to predict different songs and commit to database
* `127.0.0.1:8000/dataset` shows the 10 last songs commited to the database, genre and prediction confidence
* `127.0.0.1:8000/dataset/list` makes a list copy. This might open in the browser or you need to save it and open with notepad
* `127.0.0.1:8000/sample/punk` shows 10 samples from the db, from the punk category
