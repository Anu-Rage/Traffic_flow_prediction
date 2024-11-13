import streamlit as st
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import pydeck as pdk
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Assume TimeBlock, STGCNBlock, and STGCN classes are already defined

class TimeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        return out.permute(0, 2, 3, 1)


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, spatial_channels, out_channels, num_nodes):
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels, out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels, spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels, out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)


class STGCN(nn.Module):
    def __init__(self, num_nodes, num_features, num_timesteps_input, num_timesteps_output):
        super(STGCN, self).__init__()
        self.block1 = STGCNBlock(in_channels=num_features, out_channels=64, spatial_channels=16, num_nodes=num_nodes)
        self.block2 = STGCNBlock(in_channels=64, out_channels=64, spatial_channels=16, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        self.fully = nn.Linear((num_timesteps_input - 2 * 5) * 64, num_timesteps_output)

    def forward(self, A_hat, X):
        out1 = self.block1(X, A_hat)
        out2 = self.block2(out1, A_hat)
        out3 = self.last_temporal(out2)
        return self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))


# Load required files
def load_data():
    sensor_data = pd.read_csv("graph_sensor_locations.csv")
    sensor_lats = sensor_data["latitude"].values
    sensor_longs = sensor_data["longitude"].values
    A_wave = torch.load("A_wave.pt", map_location='cpu')
    test_input = torch.load("test_input.pt", map_location='cpu')
    means = torch.load("means.pt")
    stds = torch.load("stds.pt")
    net = STGCN(A_wave.shape[0], test_input.shape[3], num_timesteps_input=12, num_timesteps_output=3).to('cpu')
    state_dict = torch.load("stgcn_model.pt", map_location='cpu')
    net.load_state_dict(state_dict)
    net.eval()
    return sensor_lats, sensor_longs, A_wave, test_input, means, stds, net


def denormalize_output(predicted_output, mean, std):
    return predicted_output


def predict_traffic_flow(A_wave, test_input, net, means, stds):
    # Limit to first 100 samples for demonstration
    test_input = test_input[:100]
    with torch.no_grad():
        predicted_output = net(A_wave, test_input)
    denormalized_predictions = denormalize_output(predicted_output.cpu().numpy(), means, stds)
    return denormalized_predictions


def get_color_for_prediction(prediction):
    # If the prediction is greater than 60, return red
    if prediction > 60:
        return [255, 0, 0]  # RGB for red
    
    # Otherwise, use colormap for values <= 60
    if isinstance(prediction, (int, float)):  # If it's a single value, make it a list
        prediction = [prediction]
    
    norm = mcolors.Normalize(vmin=min(prediction), vmax=max(prediction))  # Normalize based on prediction range
    cmap = plt.get_cmap("YlOrRd")  # Yellow to Red colormap
    color = cmap(norm(prediction))  # Get RGBA color for each prediction value
    return [int(c * 255) for c in color[0][:3]]  # Convert to RGB values (0-255 range)


# Streamlit GUI
st.title("Traffic Flow Prediction using GNN")
st.write("This application uses an ST-GCN model to predict traffic flow in real-time on a map.")

sensor_lats, sensor_longs, A_wave, test_input, means, stds, net = load_data()

# Add slider to select batch
batch_slider = st.slider("Select Batch", min_value=0, max_value=99, value=0)

# Display traffic predictions for the selected batch
if st.button("Show Traffic Predictions"):
    predictions = predict_traffic_flow(A_wave, test_input, net, means, stds)

    # Denormalize the predictions
    denormalized_predictions = predictions * stds[0] + means[0]

    # Select the batch chosen from the slider (update predictions for the selected batch)
    batch_predictions = denormalized_predictions[batch_slider, :, 0]  # First timestep for the selected batch

    # Prepare map data with the selected batch's predictions
    map_data = pd.DataFrame({
        "node_id": np.arange(len(sensor_lats)),  # Adding node number
        "latitude": sensor_lats,
        "longitude": sensor_longs,
        "prediction": batch_predictions
    })
    
    # Apply color coding based on prediction values
    map_data["color"] = map_data["prediction"].apply(lambda x: get_color_for_prediction(x))

    # Pydeck ColumnLayer to visualize each node as a tower with congestion color
    layer = pdk.Layer(
        "ColumnLayer",
        data=map_data,
        get_position=["longitude", "latitude"],
        get_elevation="prediction",
        elevation_scale=100,  # Adjust based on prediction values for visibility
        radius=50,
        get_fill_color="color",  # Use the color mapped to traffic flow
        pickable=True,
        auto_highlight=True,
    )

    # Define tooltip for interactive display, including node id and prediction
    tooltip = {
        "html": "<b>Node ID:</b> {node_id} <br><b>Prediction:</b> {prediction}",
        "style": {"color": "white", "background-color": "#333", "padding": "10px"}
    }

    # Set view based on data's center
    view_state = pdk.ViewState(
        latitude=map_data["latitude"].mean(),
        longitude=map_data["longitude"].mean(),
        zoom=11,
        pitch=50,
    )

    # Render map with pydeck in Streamlit
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))
