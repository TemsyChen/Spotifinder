"""Machine learning functions"""
import logging
import random
from fastapi import APIRouter
import pandas as pd
from pydantic import BaseModel, Field, validator
import joblib
import pickle
from os.path import dirname
from sklearn.feature_extraction.text import TfidfVectorizer

from enum import Enum

log = logging.getLogger(__name__)
router = APIRouter()

DIR = dirname(__file__)
MODELS_DIR = DIR + '/../models/'
DATA_DIR = DIR + '/../data/'

data_filename = DATA_DIR + 'NLP_songs_data.zip'
model_filename = MODELS_DIR + 'nlp_model.pkl'
dtm_filename = MODELS_DIR + 'nlp_dtm.pkl'

df = None
loaded_model = None
dtm = None

def load_files():
    global df, loaded_model, dtm

    df = pd.read_csv(data_filename)
    loaded_model = pickle.load(open(model_filename, 'rb'))
    dtm = pickle.load(open(dtm_filename, 'rb'))

class Artist(str, Enum):
    PostMalone = "Post Malone"
    TheWeeknd = "The Weeknd"
    CeeLoGreen = "CeeLo Green"

class Song(str, Enum):
    Circles = "Circles"
    Heartless = "Heartless"
    BabyItsCold = "Baby It's Cold Outside (feat. Christina Aguilera)"

@router.post('/predict')
async def predict(artist: Artist, song: Song):
    if dtm is None:
        load_files()
    #translate artist, song into doc dtm.iloc[x].values
    artist_songs = df.loc[df['track_artist'] == artist]
    selected_song = artist_songs.loc[artist_songs['track_name'] == song]
    x = selected_song.index
    x = x[0]
    x = x.item()
    
    doc = dtm.loc[x].values
    result = loaded_model.kneighbors([doc], n_neighbors=6)

    rec_songs = {"artist": [], "song": []};

    for i in range(5):
        song = result[1][0][1 + i]

        # translate the loc into an artist and song title
        artist = df.loc[song]['track_artist']
        song = df.loc[song]['track_name']

        rec_songs['artist'].append(artist)
        rec_songs['song'].append(song)

    return rec_songs