import datetime
import sys
import base64
import io
import requests

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate
from PIL import Image
import numpy as np

from config import CONFIG
sys.path.insert(0, str(CONFIG.src))
# from inference import make_prediction
# import tensorflow as tf

#Loading model
# model = tf.keras.models.load_model(str(CONFIG.models / "content" / "cat_vs_dog"))

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1("Cat vs. Dog Classification with Tensorflow"),
    html.H4("Upload or Drag an image for prediction"),
    html.Div([
        dcc.Upload(
            id='upload-image',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '50%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=True
        ),
        html.Div(id='output-image-upload'),
    ]),
    html.Button("Predict", id="make-prediction"),
    html.Div(id="predicted-label")
])

def parse_contents(contents, filename, date):
    return html.Div([
        # html.H5(filename),
        # html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.H2("Predicting:"),
        html.Img(src=contents,
        style={
            "width": "50%",
            "height": "50%"
        }),
        html.Hr(),
    ])


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
        return html.H5(f"Prediction {pred['class']}")

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)