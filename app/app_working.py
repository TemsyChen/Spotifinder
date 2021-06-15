import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Input, Output
import pandas as pd
import uvicorn as uvicorn
from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware
# from os.path import dirname

# DIR = dirname(__file__)
# DATA_DIR = DIR + '/../data/'
# data_filename = DATA_DIR + 'NLP_songs_data.zip'

# df = pd.read_csv(data_filename)

data_filename = r'C:\Users\temsy\Documents\GitHub\Spotifinder\data\NLP_songs_data.zip'

df = pd.read_csv(data_filename)

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
    # html.Div(id='graph-container', children=[])
])

@app.callback(
    Output('Songs', 'options'),
    [Input('Artist', 'value')]
)
def set_options(artist):
    dff = df[df.track_artist == artist]
    dicosongs = [{'label': c, 'value': c} for c in sorted(dff.track_name.unique())]
    return dicosongs

if __name__ == '__main__':
    server = FastAPI()
    server.mount("/", WSGIMiddleware(app.server))
    uvicorn.run(server)
    # app.run_server(debug=True) 