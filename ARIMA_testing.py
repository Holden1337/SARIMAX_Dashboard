import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import pandas as pd

from create_ts import make_ts
import pmdarima as pm
import plotly.graph_objs as go


def create_figure(df, preds):

    x1 = list(df.index)
    y1 = list(df['values'].values)

    x2 = list(np.arange(len(df)-12, len(df)))
    y2 = preds

    trace1 = go.Scatter(
        x=x1,
        y=y1,
        mode='lines',
        name='data'
    )

    trace2 = go.Scatter(
        x=x2,
        y=y2,
        mode='lines',
        name='predicted'
    )

    layout = go.Layout(
        title=f'Time Series with SARIMAX Predictions',
        xaxis=dict(
            title='Time'
        ),
        yaxis=dict(
            title='Value'
        ),
        legend=dict(x=1.05, y=1.0, orientation="h"),
        margin=dict(l=50, r=10, t=55, b=40),
    )

    data = [trace1, trace2]

    figure = go.Figure(data=data, layout=layout)

    return figure

