import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Input, Output
import pandas as pd
from joblib import load
from os.path import dirname

DIR = dirname(__file__)
MODELS_DIR = DIR + '/../models/'
DATA_DIR = DIR + '/../data/'

data_filename = DATA_DIR + 'NLP_songs_data.zip'
model_filename = MODELS_DIR + 'nlp_model.pkl'
dtm_filename = MODELS_DIR + 'nlp_dtm.pkl'
#dtm_filename = MODELS_DIR + 'encoded_dtm.pkl'

df = None
loaded_model = None
dtm = None

def load_files():
    global df, loaded_model, dtm
    print('Loading files')
    df = pd.read_csv(data_filename)
    loaded_model = load(model_filename)
    dtm = load(dtm_filename)
    print('Loaded files')

rec_cols = ['artist','song']

load_files()
print(loaded_model)
#print(dtm)
#Plotly Dash

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Label("Spotifinder: recommending you songs with similar lyrics", style={'fontSize':40, 'textAlign':'left'}),
    html.Label("Artist:", style={'fontSize':30, 'textAlign':'left'}),
    dcc.Dropdown(
        id='Artist',
        options=[{
            'label': c,
            'value': c}
                 for c in df['track_artist']],
        value = df['track_artist'][0]
    ),
    html.Label("Songs:", style={'fontSize':30, 'textAlign':'left'}),
    dcc.Dropdown(id='Songs',
                 multi=False),
    html.Label("Recommendations:", style={'fontSize':30, 'textAlign':'left'}),
    html.Div([
        html.Div(
            [html.Tr([html.Th(col) for col in rec_cols])], id='rec-table', style={'width': '100%', 'height': '100%'}),
        dcc.Interval(id='interval_component',
                     interval=1000,
                     n_intervals=0
        )
    ],id='Recommendations')
])

@app.callback(
    Output('Songs', 'options'),
    Output('Songs', 'value'),
    [Input('Artist', 'value')]
)
def set_options(artist):
    dff = df[df.track_artist == artist]
    dicosongs = [{'label': c, 'value': c} for c in sorted(dff.track_name.unique())]
    # values_selected = [x['value'] for x in dicosongs]
    return dicosongs, dicosongs[0]['value']

@app.callback(
    Output('rec-table', 'children'),
    [Input('Artist', 'value')],
    [Input('Songs', 'value')],
)
def predict(artist, song):
    #translate artist, song into doc dtm.iloc[x].values
    print(f'<{artist}>,<{song}>')
    artist_songs = df[df['track_artist'] == artist]
    print(artist_songs)
    selected_song = artist_songs.loc[artist_songs['track_name'] == song]
    print(selected_song)
    x = selected_song.index
    print(x)
    x = x[0]
    print(x)
    x = x.item()
    print(x)
    doc = dtm.loc[x].values
    print(doc)
    result = loaded_model.kneighbors([doc], n_neighbors=6)
    rec_songs = {"artist": [], "song": []};
    for i in range(5):
        song = result[1][0][1 + i]
        # translate the loc into an artist and song title
        artist = df.loc[song]['track_artist']
        song = df.loc[song]['track_name']
        rec_songs['artist'].append(artist)
        rec_songs['song'].append(song)
    print(rec_songs)
    return html.Table(
            [html.Tr([
                html.Td(rec_songs[col][i]) for col in rec_cols
            ]) for i in range(5)]
        ) 
if __name__ == '__main__':
    app.run_server(debug=True)