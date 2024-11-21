import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Set environment variables to suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress info and warnings (0=all, 1=info, 2=warning, 3=error)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv("ghg-emissions-by-sector-stacked.csv")
    return data

# Load the pre-trained LSTM model
@st.cache_resource
def load_trained_model():
    model = load_model('lstm_model.keras')
    return model

# Streamlit app
st.title("Global Greenhouse Gas Emissions Prediction")
st.write(
    """
    This application predicts greenhouse gas emissions by sector using LSTM-RNN.
    Explore actual emissions, predictions, and future trends.
    """
)

# Load and preprocess the data
data = load_data()

# Ensure the dataset has a 'Country' column (this is assuming your dataset has a 'Country' column)
country_column = 'Country'

# Get the list of unique countries from the dataset
countries = data[country_column].unique()

sector_columns = [
    'Greenhouse gas emissions from other fuel combustion',
    'Greenhouse gas emissions from bunker fuels',
    'Greenhouse gas emissions from waste',
    'Greenhouse gas emissions from buildings',
    'Greenhouse gas emissions from industry',
    'Fugitive emissions of greenhouse gases from energy production',
    'Greenhouse gas emissions from agriculture',
    'Greenhouse gas emissions from manufacturing and construction',
    'Greenhouse gas emissions from transport',
    'Greenhouse gas emissions from electricity and heat'
]

# Sidebar for navigation without radio buttons
st.sidebar.header("Select a Feature")
feature = st.sidebar.selectbox(
    "Choose the feature to visualize:",
    ["Interactive Emissions by Sector", "Predicted Future Emissions", "Metrics"]
)

# Add country selection in the sidebar
selected_country = st.sidebar.selectbox("Select a Country", countries)

# Filter data based on selected country
country_data = data[data[country_column] == selected_country]

# Load pre-trained model
model = load_trained_model()

# Function to create datasets
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        dataX.append(dataset[i:(i + time_step), 0])
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# Function to predict future emissions
def predict_future_emissions(model, last_sequence, num_years, scaler, time_step):
    predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(num_years):
        next_emission = model.predict(current_sequence.reshape(1, time_step, 1))
        predictions.append(next_emission[0, 0])
        current_sequence = np.append(current_sequence[1:], next_emission, axis=0)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Main code logic based on sidebar selection

if feature == "Interactive Emissions by Sector":
    st.header(f"Interactive Emissions by Sector for {selected_country}")

    # Filter data by selected country
    country_emissions_by_sector = country_data.groupby('Year')[sector_columns].sum() / 1e6

    # Dropdown to select a sector
    selected_sector = st.selectbox('Select a Sector to Visualize', sector_columns)

    # Get the selected sector's data
    selected_sector_data = country_emissions_by_sector[selected_sector]

    # Plot the selected sector emissions with Plotly
    fig = go.Figure()

    # Actual emissions
    fig.add_trace(go.Scatter(
        x=country_emissions_by_sector.index,
        y=selected_sector_data,
        mode='lines+markers',
        name=f'Actual {selected_sector}',
        line=dict(color='blue'),
    ))

    # Future predictions
    years = np.arange(country_emissions_by_sector.index[-1] + 1, country_emissions_by_sector.index[-1] + 10 + 1)
    fig.add_trace(go.Scatter(
        x=years,
        y=future_predictions[selected_sector].flatten(),
        mode='lines+markers',
        name=f'Predicted Future {selected_sector}',
        line=dict(color='red', dash='dash'),
    ))

    # Update layout
    fig.update_layout(
        title=f'{selected_sector} Emissions Over Time for {selected_country}',
        xaxis_title='Year',
        yaxis_title='Emissions (Million Tonnes)',
        template='plotly_dark'
    )

    # Display the Plotly chart
    st.plotly_chart(fig)

elif feature == "Predicted Future Emissions":
    st.header(f"Predicted Future Emissions for {selected_country}")

    # Filter data by selected country
    country_emissions_by_sector = country_data.groupby('Year')[sector_columns].sum() / 1e6

    # Allow the user to select a sector to predict future emissions
    selected_sector = st.selectbox("Select a Sector", sector_columns)

    # Get the last sequence of emissions data for prediction
    sector_data = country_emissions_by_sector[selected_sector]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(sector_data.values.reshape(-1, 1))
    last_sequence = scaled_data[-3:]  # Adjust time step as needed

    # Predict future emissions
    future_emissions = predict_future_emissions(model, last_sequence, 10, scaler, time_step=3)

    # Plot future emissions
    st.write(f"Predicted future emissions for {selected_sector} in {selected_country} over the next 10 years:")
    st.line_chart(future_emissions.flatten())

elif feature == "Metrics":
    st.header(f"Model Performance Metrics for {selected_country}")

    # Filter data by selected country
    country_emissions_by_sector = country_data.groupby('Year')[sector_columns].sum() / 1e6

    # Show metrics for each sector
    mse_values, rmse_values, mae_values = {}, {}, {}
    for sector in sector_columns:
        mse_values[sector] = 0.2  # Example value, replace with actual computation
        rmse_values[sector] = 0.4  # Example value, replace with actual computation
        mae_values[sector] = 0.3  # Example value, replace with actual computation

    for sector in sector_columns:
        st.subheader(sector)
        st.write(f"Mean Squared Error (MSE):  {mse_values[sector]:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse_values[sector]:.2f}")
        st.write(f"Mean Absolute Error (MAE):  {mae_values[sector]:.2f}")
