# -*- coding: utf-8 -*-
"""
Created on Sun May 16 22:38:12 2021

@author: ramil.guluzada
"""


#from main import model
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Load the movies dataset and also pass header=None since files don't contain any headers
movies_df = pd.read_csv('C:/Users/RAMIL/Desktop/GNNCF/data/ml-100k/uitem.csv', sep='|', header=None, engine='python')
#print(movies_df.head())
# Change Path here
igembb = pd.read_csv('C:/Users/RAMIL/Desktop/GNNCF/data/ml-100k/igemb.csv', engine='python')
igembb = igembb.iloc[:,1:]
item_g_embeddings =  igembb.values
#item_g_embeddings = model.i_g_embeddings.detach().numpy()

knn_model_g = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
knn_model_g.fit(item_g_embeddings)

def get_movie_titles(input_id, nn_model, embeddings, n=20):
	dist, nnidx = nn_model.kneighbors(
		embeddings[input_id].reshape(1, -1),
		n_neighbors = n)
	titles = []
	for idx in nnidx[0]:
		try:
			titles.append(movies_df.iloc[idx,1])
		except:
			continue
	return titles

#similar_movies = get_movie_titles(58, knn_model_g, item_g_embeddings)
#similar_movies

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#ffffff',
    'text': '#111111'
}

def generate_table(dataframe, max_rows=20):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )
# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options


app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Movie Recommender System',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.Div(children='', style={
        'textAlign': 'center',
        'color': colors['text']
    }),

    html.Label(
            [
                "Select Movie",
                dcc.Dropdown(id="Movie", options = [{'label': i, 'value': i} for i in movies_df.iloc[:,1].unique()]),
            ],
            style={
            'textAlign': 'left',
            'color': colors['text']
        }
        ),
    html.Div(id='table-container')
])
#'Toy Story (1995)'
@app.callback(
    dash.dependencies.Output('table-container', 'children'),
    [dash.dependencies.Input('Movie', 'value')])
def display_table(dropdown_value):
    if dropdown_value is None:
        return "Select Movie"
    Fv = movies_df[movies_df[1]==dropdown_value][0]
    similar_movies = get_movie_titles(Fv-1, knn_model_g, item_g_embeddings)
    Movies = pd.DataFrame(similar_movies)
    Movies.columns=['Similar Movies']
    Movies = Movies.iloc[1:,:]
    return generate_table(Movies)

if __name__ == '__main__':
    app.run_server(debug=True)