import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Input, Output
import pandas as pd
import pickle
# from os.path import dirname

# DIR = dirname(__file__)
# MODELS_DIR = DIR + '/../models/'
# DATA_DIR = DIR + '/../data/'

# data_filename = DATA_DIR + 'NLP_songs_data.zip'
# model_filename = MODELS_DIR + 'nlp_model.pkl'
# dtm_filename = MODELS_DIR + 'nlp_dtm.pkl'

# df = None
# loaded_model = None
# dtm = None

# def load_files():
#     global df, loaded_model, dtm

#     df = pd.read_csv(data_filename)
#     loaded_model = pickle.load(open(model_filename, 'rb'))
#     dtm = pickle.load(open(dtm_filename, 'rb'))

# load_files()

data_filename = r'C:\Users\temsy\Documents\GitHub\Spotifinder\data\NLP_songs_data.zip'

df = pd.read_csv(data_filename)
loaded_model = pickle.load(open(r'C:\Users\temsy\Documents\GitHub\Spotifinder\models\nlp_model.pkl', 'rb'))
dtm = pickle.load(open(r'C:\Users\temsy\Documents\GitHub\Spotifinder\models\nlp_dtm.pkl', 'rb'))

#Plotly Dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, requests_pathname_prefix = '/dash/')

app.layout = html.Div([
    html.Label("Artist:", style={'fontSize':30, 'textAlign':'center'}),
    dcc.Dropdown(
        id='Artist',
        options=[{
            'label': c,
            'value': c}
            for c in df['track_artist']],
        value = df['track_artist'][0]
    ),
    html.Label("Songs:", style={'fontSize':30, 'textAlign':'center'}),
    dcc.Dropdown(id='Songs',
                 multi=False),
    html.Label("Recommendations:", style={'fontSize':30, 'textAlign':'center'}),
    html.Div(id='Recommendations')
])

@app.callback(
    Output('Songs', 'options'),
    [Input('Artist', 'value')]
)
def set_options(artist):
    dff = df[df.track_artist == artist]
    dicosongs = [{'label': c, 'value': c} for c in sorted(dff.track_name.unique())]
    return dicosongs

@app.callback(
    Output('Recommendations', 'dicorecs')
    [Input('Songs', 'value')],
    [Input('Artist', 'value')]
)
def predict(artist, song):
    # if dtm is None:
    #     load_files()
    #translate artist, song into doc dtm.iloc[x].values
    artist_songs = df.loc[df['track_artist'] == artist]
    selected_song = artist_songs.loc[artist_songs['track_name'] == song]
    x = selected_song.index
    x = x[0]
    x = x.item()
    
    doc = dtm.loc[x].values
    result = loaded_model.kneighbors([doc], n_neighbors=6)

    songs = []
    # rec_songs = {"artist": [], "song": []};

    for i in range(5):
        song = result[1][0][1 + i]

        # translate the loc into an artist and song title
        artist = df.loc[song]['track_artist']
        song = df.loc[song]['track_name']

        # rec_songs['artist'].append(artist)
        # rec_songs['song'].append(song)
        songs.append(song)

    return result[1][0]

if __name__ == '__main__':
    app.run_server(debug=True)