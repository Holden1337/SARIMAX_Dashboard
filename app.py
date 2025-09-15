import time
import importlib

import dash
from dash import html
from dash import dcc
import numpy as np
from dash.dependencies import Input, Output, State
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt

import dash_reusable_components as drc

from utils import create_figure, train_test_data

app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
app.title = "SARIMAX Demo"
server = app.server



app.layout = html.Div(
    children=[
        # .container class is fixed, .container.scalable is scalable
        html.Div(
            className="banner",
            children=[
                # Change App Name here
                html.Div(
                    className="container scalable",
                    children=[
                        # Change App Name here
                        html.H2(
                            id="banner-title",
                            children=[
                                html.A(
                                    "Seasonal Autoregressive Integrated Moving Average (SARIMAX) Explorer",
                                    href="https://github.com/Holden1337/SARIMAX_Dashboard",
                                    style={
                                        "text-decoration": "none",
                                        "color": "inherit",
                                    },
                                )
                            ],
                        ),
                        html.A(
                            id="banner-logo",
                            children=[
                                html.Img(src=app.get_asset_url("dash-logo-new.png"))
                            ],
                            href="https://plot.ly/products/dash/",
                        ),
                    ],
                )
            ],
        ),
        html.Div(
            id="body",
            className="container scalable",
            children=[
                html.Div(
                    id="app-container",
                    # className="row",
                    children=[
                        html.Div(
                            # className="three columns",
                            id="left-column",
                            children=[
                                drc.Card(
                                    id="first-card",
                                    children=[
                                        dcc.Markdown("#####  Select a dataset and modify SARIMAX parameters to find the best model"),
                                        drc.NamedDropdown(
                                            name="Select Dataset",
                                            id="dropdown-select-dataset",
                                            options=[
                                                {
                                                 "label": "Airline passengers",
                                                 "value": "airpassenger"
                                                 },
                                                {
                                                  "label": "Austrailian Beer Consumption",
                                                  "value": "ausbeer",
                                                },
                                                {
                                                  "label": "Austrailian Population",
                                                  "value": "austres"
                                                },
                                            ],
                                            clearable=False,
                                            searchable=False,
                                            value="airpassenger",
                                        ),
                                    ],
                                ),
                                drc.Card(
                                    id="button-card",
                                    children=[
                                        dcc.Markdown("### Select order parameters"),
                                        drc.NamedSlider(
                                            name="p value",
                                            id="p-value",
                                            min=0,
                                            max=3,
                                            value=1,
                                            step=1,
                                        ),
                                        drc.NamedSlider(
                                            name="d value",
                                            id="d-value",
                                            min=0,
                                            max=3,
                                            value=1,
                                            step=1,
                                        ),
                                        drc.NamedSlider(
                                            name="q value",
                                            id="q-value",
                                            min=0,
                                            max=3,
                                            value=1,
                                            step=1,
                                        ),
                                    ],
                                ),
                                drc.Card(
                                    id="last-card",
                                    children=[
                                        dcc.Markdown("### Select Seasonal parameters"),
                                        drc.NamedSlider(
                                            name="P value",
                                            id="P-value",
                                            min=0,
                                            max=3,
                                            value=1,
                                            step=1,
                                        ),
                                        drc.NamedSlider(
                                            name="D value",
                                            id="D-value",
                                            min=0,
                                            max=3,
                                            value=1,
                                            step=1,
                                        ),
                                        drc.NamedSlider(
                                            name="Q value",
                                            id="Q-value",
                                            min=0,
                                            max=3,
                                            value=1,
                                            step=1,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            id="ARIMA-graph",
                            children=dcc.Graph(
                               id="ARIMA-figure",
                                figure=dict(
                                    layout=dict(
                                        plot_bgcolor="#282b38", paper_bgcolor="#282b38"
                                   )
                                ),
                            ),
                           
                        ),
                    ],
                )
            ],
        ),
    ]
)

@app.callback(
    Output("ARIMA-graph", "children"),
    [
        Input("dropdown-select-dataset", "value"),
        Input("p-value", "value"),
        Input("d-value", "value"),
        Input("q-value", "value"),
        Input("P-value", "value"),
        Input("D-value", "value"),
        Input("Q-value", "value"),
    ],
)
def update_ARIMA_graph(
    dataset,
    p_value, 
    d_value,
    q_value,
    P_value,
    D_value,
    Q_value,
    
):
    train, test, m = train_test_data(dataset)
    order = (p_value, d_value, q_value)
    seasonal_order = (P_value, D_value, Q_value, m)
    model = sm.tsa.statespace.SARIMAX(train, trend='n', order=order, seasonal_order=seasonal_order)
    results = model.fit()
    preds = results.predict(start=len(train)-1, end=len(train)+len(test)-2, dynamic=True)
    test.fillna(method='ffill', inplace=True)
    rmse = sqrt(mean_squared_error(test, preds))
    figure = create_figure(train, test, preds)
    figure.layout.paper_bgcolor = "#282b38"
    figure.layout.width = 1000
    figure.layout.height = 600
    figure.update_layout(
        plot_bgcolor='#282b38',
        yaxis=dict(color="#FFFFFF"),
        xaxis=dict(color="#FFFFFF"),
        title={"text": f"Time Series with SARIMAX Predictions, RMSE: {round(rmse,3)}"},
        font=dict(family="Source Code Pro", size=12, color="white")
        )
    return [
        html.Div(
            id='ARIMA-ghaph-container',
            children=dcc.Loading(
                className='graph-wrapper',
                children=dcc.Graph(id='ARIMA-figure', figure=figure),
                style={"display": "none"},
                type="circle"
            ),
        ),
    ]


# Running the server
if __name__ == "__main__":
    app.run(debug=False, port=8050, host='0.0.0.0')
