import dash

from dash import dcc
from dash import html
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

data=pd.read_excel(r'G:\Kundan\NAMRC 2023\ax_1.xlsx',header=None,names=["time","acc"])

df=pd.read_excel(r'G:\Kundan\NAMRC 2023\ax_max.xlsx',header=None,names=["fr","max"])

app = dash.Dash(__name__)
app.layout = html.Div(
    children=[
        html.H1(children="Stability Analysis "
                         "During Thin-wall Machining",),
        html.P(
            children="Analyze the stability of machining process "
                     "during micro-machining of Thin-walled Ti6Al4V"
        ),
        dcc.Graph(
            figure={
                "data": [
                    {
                        "x": data["time"],
                        "y": data["acc"],
                        "type": "lines",
                        'xaxis':{'title': 'Time in sec'}
                    },
                ],
                "layout": {"title": "Acceleration Spectrum"},
            },
        ),
        dcc.Graph(
            figure={
                "data": [
                    {
                        "x": df["fr"],
                        "y": df["max"],
                        "type": "bar,width=-10, tick_label=df['fr'], align='center'"
                    },
                ],
                "layout": {"title": "Acceleration spectrum dominant amplitude plot"},
            },
        ),
    ]
)

if __name__ == '__main__':
    app.run_server(debug=False)