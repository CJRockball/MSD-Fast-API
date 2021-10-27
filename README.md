# MSD-Fast-API
Using an XGBoost model to predict genre of a subset of the million song dataset. The main can be run as is pkl-files of data pipeline and model are included. If you want to over-write the pickel file, just run the xgb_model_train. A small database is set up and preloaded with 100 songs from the training dataset. The genre prediction confidence for these songs will be 1 since we already know it. The api features predictions, a sample of the latest commited songs, a sample of a specified genre.

## Run
### To Run from File
* Clone the repo
* In terminal go to folder run `uvicorn main:app --reload`
* open browser goto choose song to predict from the data test file by adding a number between 1 and 428 `127.0.0.1:8000/predict/?q=88` the predicted file will be added to the database.
  
### To Run from Docker
* Clone the repo
* In terminal go to folder and run `docker build -t name .` 
* then run `docker run -dp 8000:8000 name`
* open browser goto choose song to predict from the data test file by adding a number between 1 and 428 `127.0.0.1:8000/predict/?q=88` the predicted file will be added to the database.

# Functions
