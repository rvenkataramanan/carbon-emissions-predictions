import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import numpy as np

# Set environment variables to suppress TensorFlow warnings
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

# Get the entities (countries) and sectors
entity_column = 'Entity'  # The column that contains the countries/entities
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
entities = data[entity_column].unique()

# Sidebar controls
st.sidebar.header("Model Parameters")

# User input for model parameters
#num_future_years = st.sidebar.slider("Number of Future Years to Predict", 1, 20, 10)
num_future_years = st.sidebar.slider("Number of Future Years to Predict", 1, 30, 10, step=1)

# Load pre-trained model
model = load_trained_model()

# Fixed time step used during model training
time_step = 3

# Dropdown to select a country/entity
selected_entity = st.sidebar.selectbox('Select a Country/Entity to Visualize', entities)

# Dropdown for feature selection
st.sidebar.header("Select a Feature")
feature = st.sidebar.selectbox(
    "Choose the feature to visualize:",
    ["Interactive Emissions by Sector", "Predicted Future Emissions", "Metrics"]
)

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

# Handle different features
if feature == "Interactive Emissions by Sector":
    # Dropdown to select a sector for this feature
    selected_sector = st.sidebar.selectbox('Select a Sector to Visualize', sector_columns)
    
    # Filter data by selected entity (country)
    entity_data = data[data[entity_column] == selected_entity]

    # Check if the selected sector exists in the data or if it has missing data
    if selected_sector not in entity_data.columns or entity_data[selected_sector].isnull().sum() > 0:
        st.error(f"Data for '{selected_sector}' is unavailable or incomplete for {selected_entity}. Please choose another sector.")
    else:
        # Group by year and sum emissions for the selected entity
        sector_data = entity_data.groupby('Year')[selected_sector].sum()

        # Normalize the sector data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_sector_data = scaler.fit_transform(sector_data.values.reshape(-1, 1))

        # Create dataset
        X, y = create_dataset(scaled_sector_data, time_step)
        X = X.reshape((X.shape[0], time_step, 1))

        # Skip training: Use the pre-trained model for predictions
        y_pred = model.predict(X)

        y_pred_actual = scaler.inverse_transform(y_pred)
        y_test_actual = scaler.inverse_transform(y.reshape(-1, 1))

        last_sequence = scaled_sector_data[-time_step:]
        future_emissions = predict_future_emissions(model, last_sequence, num_future_years, scaler, time_step)

        # Plot the selected sector emissions with Plotly
        st.header(f"Interactive Emissions for {selected_entity} in {selected_sector}")

        # Create a more visually appealing plot
        fig = go.Figure()

        # Actual emissions
        fig.add_trace(go.Scatter(
            x=sector_data.index,
            y=sector_data,
            mode='lines+markers',
            name=f'Actual {selected_sector}',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8, color='#1f77b4'),
            hovertemplate='Year: %{x}<br>Emissions: %{y:.2f} Million Tonnes'
        ))

        # Future predictions
        years = np.arange(sector_data.index[-1] + 1, sector_data.index[-1] + num_future_years + 1)
        fig.add_trace(go.Scatter(
            x=years,
            y=future_emissions.flatten(),
            mode='lines+markers',
            name=f'Predicted Future {selected_sector}',
            line=dict(color='#ff7f0e', width=3, dash='dash'),
            marker=dict(size=8, color='#ff7f0e'),
            hovertemplate='Year: %{x}<br>Predicted Emissions: %{y:.2f} Million Tonnes'
        ))

        # Update layout for a cleaner, prettier design
        fig.update_layout(
            title=f'{selected_sector} Emissions Over Time for {selected_entity}',
            xaxis_title='Year',
            yaxis_title='Emissions (Million Tonnes)',
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
            font=dict(family='Arial', size=12, color='white'),
            showlegend=True
        )

        # Display the Plotly chart
        st.plotly_chart(fig)

elif feature == "Predicted Future Emissions":
    # Display the predicted future emissions for the selected entity
    st.header(f"Predicted Future Emissions for {selected_entity}")
    future_predictions = {}

    for sector in sector_columns:
        entity_data = data[data[entity_column] == selected_entity]
        
        # Ensure the sector exists in the data
        if sector not in entity_data.columns or entity_data[sector].isnull().sum() > 0:
            st.write(f"Data for '{sector}' is unavailable for {selected_entity}. Skipping...")
            continue

        # Group by year and sum emissions for the selected entity
        sector_data = entity_data.groupby('Year')[sector].sum()

        # Normalize the sector data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_sector_data = scaler.fit_transform(sector_data.values.reshape(-1, 1))

        # Create dataset
        X, y = create_dataset(scaled_sector_data, time_step)
        X = X.reshape((X.shape[0], time_step, 1))

        # Use the pre-trained model for predictions
        y_pred = model.predict(X)

        y_pred_actual = scaler.inverse_transform(y_pred)
        y_test_actual = scaler.inverse_transform(y.reshape(-1, 1))

        last_sequence = scaled_sector_data[-time_step:]
        future_emissions = predict_future_emissions(model, last_sequence, num_future_years, scaler, time_step)
        future_predictions[sector] = future_emissions

        # Plot future predictions for each sector
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.arange(2025, 2025 + num_future_years),
            y=future_emissions.flatten(),
            mode='lines+markers',
            name=f'Predicted Future {sector}',
            line=dict(color='red', width=3, dash='dash'),
            marker=dict(size=8, color='red'),
            hovertemplate='Year: %{x}<br>Predicted Emissions: %{y:.2f} Million Tonnes'
        ))

        fig.update_layout(
            title=f'Predicted Future Emissions for {sector} in {selected_entity}',
            xaxis_title='Year',
            yaxis_title='Emissions (Million Tonnes)',
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
            font=dict(family='Arial', size=12, color='white'),
            showlegend=True
        )
        st.plotly_chart(fig)

elif feature == "Metrics":
    # Calculate metrics (MSE, RMSE, MAE) for the selected entity and sector
    st.header(f"Metrics for {selected_entity}")
    selected_sector = st.selectbox('Select a Sector to View Metrics', sector_columns)
    
    # Check if the selected sector exists in the data
    entity_data = data[data[entity_column] == selected_entity]
    
    if selected_sector not in entity_data.columns or entity_data[selected_sector].isnull().sum() > 0:
        st.error(f"Data for '{selected_sector}' is unavailable or incomplete for {selected_entity}. Please choose another sector.")
    else:
        # Group by year and sum emissions for the selected entity
        sector_data = entity_data.groupby('Year')[selected_sector].sum()

        # Normalize the sector data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_sector_data = scaler.fit_transform(sector_data.values.reshape(-1, 1))

        # Create dataset
        X, y = create_dataset(scaled_sector_data, time_step)
        X = X.reshape((X.shape[0], time_step, 1))

        # Skip training: Use the pre-trained model for predictions
        y_pred = model.predict(X)

        y_pred_actual = scaler.inverse_transform(y_pred)
        y_test_actual = scaler.inverse_transform(y.reshape(-1, 1))

        # Calculate metrics (MSE, RMSE, MAE)
        mse = np.mean((y_test_actual - y_pred_actual) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test_actual - y_pred_actual))

        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
