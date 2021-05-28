import datetime
import sys
import base64
import io
import requests

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from PIL import Image
import numpy as np

from config import CONFIG
sys.path.insert(0, str(CONFIG.src))

external_stylesheets = [
    dbc.themes.FLATLY,
    "https://fonts.gstatic.com",
    "https://fonts.googleapis.com/css?family=Sofia",
    {
    "href":"https://fonts.gstatic.com",
    "rel":"preconnect",
    "href":"https://fonts.googleapis.com/css2?family=Roboto+Slab:wght@300&display=swap",
    "rel": "stylesheet"}]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dbc.NavbarSimple([
                dbc.NavItem(dbc.NavLink("Page 1", href="#")),
                dbc.NavItem(dbc.NavLink("About", href="#"))
            ], 
            className="upload-title",
            brand="Cat vs. Dog Classification with Tensorflow",
            color="primary",
            dark=True,
            ),
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H4(
                    "Upload or Drag an image for prediction",
                    ),
                ],
                className="upload-call"
            ),
            html.Div([
                dcc.Upload(
                    id='upload-image',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Files')
                    ]),
                    className="photo-upload",
                    # Allow multiple files to be uploaded
                    multiple=True
                ),
                html.Div(id='output-image-upload'),
            ]),
            ], 
            width=8,
            md=8,
            className="upload-col"),
        dbc.Col([
                html.Div([
                    dbc.Button(
                        "Predict", 
                        id="make-prediction", 
                        color="primary",
                        className="predict-button"
                        ),
                    html.Hr(),
                    html.Div(id="predicted-label"),
                    ],
                className="predict-col-button"),
            ], 
            width=4,
            md=4,
            className="predict-col")

        ])
    ], className="title")

def parse_contents(contents, filename, date):
    return html.Div([
        # html.H5(filename),
        # html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.H2("Predicting:", className="uploaded-img-header"),
        html.Img(src=contents,
                className="uploaded-image"),
        html.Hr(),
            ], 
        className="uploaded-col"
        )


@app.callback(Output('output-image-upload', 'children'),
            Input('upload-image', 'contents'),
            State('upload-image', 'filename'),
            State('upload-image', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

#========= Prediction Helper ===============
def read_image_string(contents):
    encoded_data = contents[0].split(",")[1]
    img_decode = base64.b64decode(encoded_data)
    np_arr = np.array(Image.open(io.BytesIO(img_decode)))
    return np_arr

@app.callback(
    Output(component_id="predicted-label", component_property="children"),
    Input(component_id="make-prediction", component_property="n_clicks"),
    State("upload-image", "contents")
)
def update_output(n_clicks, contents):
    if n_clicks is None:
        raise PreventUpdate
    else:
        url = "http://127.0.0.1:8000/predict"
        split_content = contents[0].split(",")
        encoded_data = split_content[1]
        r = requests.post(url, json={"contents": encoded_data})
        pred = r.json()[0]
        return html.H5(f"It's a {pred['class']}!")
        # if n_clicks % 2 == 1:
        #     return html.H4("Checkmate!")

if __name__ == '__main__':
    app.run_server(debug=True)