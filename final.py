from dash import html
from dash import dcc
from dash import Dash
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
app = Dash(__name__)

df = pd.read_csv('data21.csv')
X = df[['velocity', 'feed']].values.astype(float)
y_ktc = df['ktc'].values.astype(float)
y_krc = df['krc'].values.astype(float)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y_ktc_scaled = scaler.fit_transform(y_ktc.reshape(-1, 1))
y_krc_scaled = scaler.fit_transform(y_krc.reshape(-1, 1))

X_tensor = torch.from_numpy(X_scaled).float()
y_ktc_tensor = torch.from_numpy(y_ktc_scaled).float()
y_krc_tensor = torch.from_numpy(y_krc_scaled).float()


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 64)  # Input size changed to 2 for two features
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model_ktc = SimpleNN()
model_krc = SimpleNN()

criterion = nn.MSELoss()

optimizer_ktc = optim.Adam(model_ktc.parameters(), lr=0.001)
optimizer_krc = optim.Adam(model_krc.parameters(), lr=0.001)

predictions_ktc = []
predictions_krc = []

# Initialize figures
fig_ktc_velocity = go.Figure()
fig_ktc_feed = go.Figure()
fig_krc_velocity = go.Figure()
fig_krc_feed = go.Figure()
fig_ktc_diff_velocity = go.Figure()
fig_ktc_diff_feed = go.Figure()
fig_krc_diff_velocity = go.Figure()
fig_krc_diff_feed = go.Figure()

# Plot actual data only once
fig_ktc_velocity.add_trace(go.Scatter(x=df['velocity'], y=y_ktc, mode='lines', name='Actual Data'))
fig_ktc_feed.add_trace(go.Scatter(x=df['feed'], y=y_ktc, mode='lines', name='Actual Data'))
fig_krc_velocity.add_trace(go.Scatter(x=df['velocity'], y=y_krc, mode='lines', name='Actual Data'))
fig_krc_feed.add_trace(go.Scatter(x=df['feed'], y=y_krc, mode='lines', name='Actual Data'))

for epoch in range(100):
    optimizer_ktc.zero_grad()
    optimizer_krc.zero_grad()

    output_ktc = model_ktc(X_tensor)
    output_krc = model_krc(X_tensor)

    loss_ktc = criterion(output_ktc, y_ktc_tensor)
    loss_krc = criterion(output_krc, y_krc_tensor)

    loss_ktc.backward()
    loss_krc.backward()

    optimizer_ktc.step()
    optimizer_krc.step()

    predictions_ktc_epoch = scaler.inverse_transform(output_ktc.detach().numpy()).flatten()
    predictions_krc_epoch = scaler.inverse_transform(output_krc.detach().numpy()).flatten()

    predictions_ktc.append(predictions_ktc_epoch)
    predictions_krc.append(predictions_krc_epoch)

    # Update ktc vs velocity graph
    fig_ktc_velocity.add_trace(go.Scatter(x=df['velocity'], y=predictions_ktc_epoch, mode='lines', name=f'Prediction (Epoch {epoch + 1})'))

    # Update ktc vs feed graph
    fig_ktc_feed.add_trace(go.Scatter(x=df['feed'], y=predictions_ktc_epoch, mode='lines', name=f'Prediction (Epoch {epoch + 1})'))

    # Update krc vs velocity graph
    fig_krc_velocity.add_trace(go.Scatter(x=df['velocity'], y=predictions_krc_epoch, mode='lines', name=f'Prediction (Epoch {epoch + 1})'))

    # Update krc vs feed graph
    fig_krc_feed.add_trace(go.Scatter(x=df['feed'], y=predictions_krc_epoch, mode='lines', name=f'Prediction (Epoch {epoch + 1})'))

# Update ktc difference graph for velocity
avg_diff_ktc_velocity = [sum(abs(y_ktc - pred)) / len(y_ktc) for pred in predictions_ktc]
fig_ktc_diff_velocity.add_trace(go.Scatter(x=list(range(1, len(predictions_ktc) + 1)), y=avg_diff_ktc_velocity, mode='lines', name='Average Difference'))

# Update ktc difference graph for feed
avg_diff_ktc_feed = [sum(abs(y_ktc - pred)) / len(y_ktc) for pred in predictions_ktc]
fig_ktc_diff_feed.add_trace(go.Scatter(x=list(range(1, len(predictions_ktc) + 1)), y=avg_diff_ktc_feed, mode='lines', name='Average Difference'))

# Update krc difference graph for velocity
avg_diff_krc_velocity = [sum(abs(y_krc - pred)) / len(y_krc) for pred in predictions_krc]
fig_krc_diff_velocity.add_trace(go.Scatter(x=list(range(1, len(predictions_krc) + 1)), y=avg_diff_krc_velocity, mode='lines', name='Average Difference'))

# Update krc difference graph for feed
avg_diff_krc_feed = [sum(abs(y_krc - pred)) / len(y_krc) for pred in predictions_krc]
fig_krc_diff_feed.add_trace(go.Scatter(x=list(range(1, len(predictions_krc) + 1)), y=avg_diff_krc_feed, mode='lines', name='Average Difference'))

# Update layout
fig_ktc_velocity.update_layout(xaxis={'title': 'Velocity'}, yaxis={'title': 'KTC VS VELOCITY'})
fig_ktc_feed.update_layout(xaxis={'title': 'Feed'}, yaxis={'title': 'KTC VS FEED'})
fig_krc_velocity.update_layout(xaxis={'title': 'Velocity'}, yaxis={'title': 'KRC VS VELOCITY'})
fig_krc_feed.update_layout(xaxis={'title': 'Feed'}, yaxis={'title': 'KRC VS FEED'})
fig_ktc_diff_velocity.update_layout(xaxis={'title': 'Epochs'}, yaxis={'title': 'KTC Average Difference Over Epochs with Velocity'})
fig_ktc_diff_feed.update_layout(xaxis={'title': 'Epochs'}, yaxis={'title': 'KTC Average Difference Over Epochs with Feed'})
fig_krc_diff_velocity.update_layout(xaxis={'title': 'Epochs'}, yaxis={'title': 'KRC Average Difference Over Epochs with Velocity'})
fig_krc_diff_feed.update_layout(xaxis={'title': 'Epochs'}, yaxis={'title': 'KRC Average Difference Over Epochs with Feed'})

# Set up app layout
app.layout = html.Div([
    html.Div([
        dcc.Graph(id='ktc_velocity_graph', figure=fig_ktc_velocity),
        dcc.Graph(id='ktc_feed_graph', figure=fig_ktc_feed),
    ], style={'display': 'inline-block', 'width': '50%'}),
    html.Div([
        dcc.Graph(id='krc_velocity_graph', figure=fig_krc_velocity),
        dcc.Graph(id='krc_feed_graph', figure=fig_krc_feed),
    ], style={'display': 'inline-block', 'width': '50%'}),

    html.Div([
        dcc.Graph(id='ktc_diff_velocity_graph', figure=fig_ktc_diff_velocity),
        dcc.Graph(id='ktc_diff_feed_graph', figure=fig_ktc_diff_feed),
    ], style={'display': 'inline-block', 'width': '50%'}),
    html.Div([
        dcc.Graph(id='krc_diff_velocity_graph', figure=fig_krc_diff_velocity),
        dcc.Graph(id='krc_diff_feed_graph', figure=fig_krc_diff_feed),
    ], style={'display': 'inline-block', 'width': '50%'}),

    html.Div([
        html.H4('Enter Iteration Number, Velocity, and Feed:'),
        dcc.Input(id='iteration-input', type='number', placeholder='Enter Iteration Number', value=0),
        dcc.Input(id='velocity-input', type='number', placeholder='Enter Velocity', value=0),
        dcc.Input(id='feed-input', type='number', placeholder='Enter Feed', value=0),
        html.Button('Submit', id='submit-button', n_clicks=0),
        html.Div(id='output-container-button')
    ])
])

@app.callback(
    Output('output-container-button', 'children'),
    [Input('submit-button', 'n_clicks')],
    [Input('iteration-input', 'value'),
     Input('velocity-input', 'value'),
     Input('feed-input', 'value')]
)
def update_output(n_clicks, iteration, velocity, feed):
        if n_clicks > 0:
            if iteration >= 1 and iteration <= 100 and velocity >= min(df['velocity']) and velocity <= max(df['velocity']) and feed >= min(df['feed']) and feed <= max(df['feed']):
                speed_index = (df['velocity'] == velocity) & (df['feed'] == feed)
                ktc_actual = y_ktc[speed_index]
                krc_actual = y_krc[speed_index]

                ktc_predicted = predictions_ktc[int(iteration) - 1][df.index[speed_index][0]]
                krc_predicted = predictions_krc[int(iteration) - 1][df.index[speed_index][0]]
                return f'KTC Actual: {ktc_actual}, KTC Predicted: {ktc_predicted}, KRC Actual: {krc_actual}, KRC Predicted: {krc_predicted}'
            else:
                return 'Invalid input. Please enter valid iteration number (1-100), velocity, and feed within the dataset range.'

if __name__ == '__main__':
    app.run_server(debug=False, port=3009)
