# MSD-Fast-API
Using an XGBoost model to predict genre of a subset of the million song dataset

## To Run from File
* Clone the repo
* In terminal go to folder run `uvicorn main:app --reload`
* open browser goto choose song to predict from the data test file by adding a number between 1 and 428 `127.0.0.1:8000/predict/?q=88` the predicted file will be added to the database.
  
## To Run from Docker
