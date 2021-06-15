# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Input, Output, State
import pandas as pd
from os.path import dirname
import logging
# import numpy as np

#load dataset
# log = logging.getLogger(__name__)

# DIR = dirname(__file__)
# DATA_DIR = DIR + '/../data/'

# data_filename = DATA_DIR + 'NLP_songs_data.zip'
data_filename = r'C:\Users\temsy\Documents\GitHub\Spotifinder\data\NLP_songs_data.zip'

df = pd.read_csv(data_filename)

# dic = dict([([artist, song]) for artist, song in zip(df.track_artist, df.track_name)])

# def create_dic(x):
#     artist_dic = {}
#     for value in x['track_artist']:
#         dic[value] = value
#     return artist_dic

# artist_dic = [create_dic(df)]

# data = list(dic.items())
# array = np.array(data)

# def load_files():
#     global df

#     df = pd.read_csv(data_filename)


#Plotly Dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Label("Artist:", style={'fontSize':30, 'textAlign':'center'}),
    dcc.Dropdown(
        id='Artist',
        options=[{
            'label': c,
            'value': c}
            for c in df['track_artist']]
    ),
    # html.Label("Songs:", style={'fontSize':30, 'textAlign':'center'}),
    # dcc.Dropdown(
    #     id='Songs',
    #     options=[{
    #         'label': c,
    #         'value': c}
    #         for c in df['track_artist']],
    #     multi=False),
    # html.Div(id='graph-container', children=[])
    html.Label("Songs:", style={'fontSize':30, 'textAlign':'center'}),
    dcc.Dropdown(id='Songs',
                 options = [],
                 value=[],
                 multi=False),
    # html.Div(id='graph-container', children=[])
])

# @app.callback(
#     Output('Songs', 'options'),
#     Output('Songs', 'value'),
#     Input('Artist', 'value'),
# )

# @app.callback(
#     Output('graph-container', 'children'),
#     Input('Songs', 'value'),
#     Input('Artist', 'value'),
#     prevent_initial_call=True
# )

# def set_options(artist):
#     dff = df[df.artist_name == artist]
#     dicosongs = [{'label': c, 'value': c} for c in sorted(dff.track_name.unique())]
#     values_selected = [x['value'] for x in dicosongs]
#     return dicosongs, values_selected

@app.callback(
    Output('Songs', 'children'),
    [Input('Artist', 'value')],
    [State('Artist','value')]
    )

def set_options(artist):
    dff = df[df.artist_name == artist]
    dicosongs = [{'label': c, 'value': c} for c in sorted(dff.track_name.unique())]
    values_selected = [x['value'] for x in dicosongs]
    songs_avail = [{'label': c,'value': c} for c in values_selected]
    return songs_avail

if __name__ == '__main__':
    app.run_server(debug=True)