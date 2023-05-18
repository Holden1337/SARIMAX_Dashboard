import pmdarima.datasets
import plotly.graph_objs as go
import statsmodels.api as sm
import numpy as np


def train_test_data(dataset):
    if dataset == "airpassenger":
        data = pmdarima.datasets.load_airpassengers(as_series=True)
        m = 12
        train = data[:120]
        test = data[119:]
    elif dataset == "ausbeer":
        data = pmdarima.datasets.load_ausbeer(as_series=True)
        m = 4
        train = data[:180]
        test = data[179:]
    elif dataset == "austres":
        data = pmdarima.datasets.load_austres(as_series=True)
        m = 4
        train = data[:74]
        test = data[73:]
    
    return train, test, m


def create_figure(train, test, preds):

    x1 = list(train.index)
    y1 = list(train.values)

    x2 = list(test.index)
    y2 = list(test.values)

    x3 = list(np.arange(len(train)-1, (len(train)+len(test)-1)))
    y3 = list(preds.values)


    trace1 = go.Scatter(
        x=x1,
        y=y1,
        mode='lines',
        name='train'
    )

    trace2 = go.Scatter(
        x=x2,
        y=y2,
        mode='lines',
        name='actual'
    )

    trace3 = go.Scatter(
        x=x3,
        y=y3,
        mode='lines',
        name='predicted'
    )

    layout = go.Layout(
        title=f'Time Series with SARIMAX Predictions',
        xaxis=dict(
            title='Time',
            gridcolor='white'
        ),
        yaxis=dict(
            title='Value'
        ),
        legend=dict(x=1.05, y=1.0, orientation="h"),
        margin=dict(l=50, r=10, t=55, b=40),
    )

    data = [trace1, trace2, trace3]

    figure = go.Figure(data=data, layout=layout)

    return figure


if __name__ == "__main__":
    pass

        