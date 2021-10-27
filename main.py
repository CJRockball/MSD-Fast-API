# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 09:07:50 2021

@author: PatCa
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from typing import Optional
import pathlib
from pickle import load
import numpy as np
import cleaning_functions
import db_fcn


app = FastAPI()

BASE_DIR = pathlib.Path(__file__).resolve().parent
XGB_MODEL_PATH = BASE_DIR / "model_artifacts/xgb_pca.pkl" 


AI_MODEL = None
labels_legend = {'soul and reggae':0, 'pop':1, 'punk':2, 'jazz and blues':3, 
              'dance and electronica':4,'folk':5, 'classic pop and rock':6, 'metal':7}
labels_legend_inverted = {f"{v}": k for k,v in labels_legend.items()}



def predict(query:int):
    """
    Function getting genre for song from the test file, either from the database or by predicting. 
    If genre is predicted it is loaded to the database. Database is preloaded with 100 songs from
    the train dataset. Since genre is known confidence for them is 1.

    Parameters
    ----------
    query : int
        Row from test dataset

    Returns
    -------
    dict
        Genre prediction and probability
    song_title : str
        Song title 

    """
    global labels_legend, labels_legend_inverted, AI_MODEL
    # Preprocess data for genre prediction
    test_pipe, text_test, song_data_dict = cleaning_functions.get_test_data(query)  
    # Get metadata for db
    song_title = song_data_dict['song_title']
    song_duration = song_data_dict['song_duration']
    song_tempo = song_data_dict['song_tempo']
    # Check if chosen song is already in database.
    if db_fcn.check_song_in_db(song_title):
        # Get data from db
        genre_data = db_fcn.get_db_genre(song_title)
        # Move data to variables
        suggested_genre, num_genre, confidence = genre_data[0], genre_data[1], genre_data[2]
        # Make dict to return
        top_pred_dict = {'labels':suggested_genre,
                         'confidence': float(confidence)}
    # If song is not in db, predict and persist in db
    else:
        #Get model prediction
        preds_array = AI_MODEL.predict_proba(test_pipe)
        # Get first prediction
        preds = preds_array[0]
        # Get category with highest probability
        top_idx_val = np.argmax(preds)
        # Get prediction probability
        top_pred = float(preds[top_idx_val])
        # Make data dict
        top_pred_dict = {'labels':labels_legend_inverted[str(top_idx_val)],
                         'confidence': top_pred}
        # Commit to database
        db_fcn.load_song_to_db(str(song_title), int(song_tempo), int(song_duration), int(top_idx_val), top_pred)

    
    return {'top':top_pred_dict}, song_title

#Preload model
@app.on_event("startup")
def on_startup():
    global XGB_MODEL_PATH, AI_MODEL
    if XGB_MODEL_PATH.exists():
        AI_MODEL = load(open(XGB_MODEL_PATH,'rb'))

# Home page for predicting genre
@app.get("/")
def read_index(q:Optional[str] = None):
    return {"query":q} 

# Home page for predicting genre
@app.get("/predict")
def read_index(q:Optional[str] = None):
    preds_dict, song_title = predict(int(q))
    return {"query":q, "song_title": song_title, "results": preds_dict} 

# Get the 10 last songs commited to the db
@app.get("/dataset")
def dataset_sample():
    rows = db_fcn.get_db_sample()
    return list(rows)

# Support function to make generator
def fetch_rows():
    result_set = db_fcn.get_db_sample()
    yield "Song title, Genre, Confidence\n"
    for i in result_set:
        yield f"{i['Song title']},{i['Genre']},{['Confidence']}\n"

# Use generator to get csv of last 10 samples in the db
@app.get("/dataset/list")
def export_list():    
    return StreamingResponse(fetch_rows())   

# Get sample from a choosen genre
@app.get("/sample/{genre}")
def get_genre_sample(genre):
    global labels_legend
    num_genre = labels_legend[str(genre)]
    rows = db_fcn.get_genre_sample(str(num_genre))   
    return list(rows) 



























