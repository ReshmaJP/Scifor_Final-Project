# Import Libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from statsmodels.tsa.stattools import adfuller
#from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
import os
import plotly.express as px

# Set the background color of the app
st.markdown(
    """
    <style>
    body {
        background-color: #000000;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main Heading with styling
st.markdown(
    """
    <h1 style='color: orange;'>Traffic Forecasting Application</h1>
    """,
    unsafe_allow_html=True
)

# Introduction
st.markdown(
    """

    Urban traffic poses significant challenges, necessitating advanced tools for prediction and analysis. Traffic congestion has always been a crucial aspect of urban planning but has become a serious issue that must be addressed due to the rapid increase in the number of vehicles and transportation demand.
    
    Traditional time series forecasting methods are often inadequate for capturing the complexities of traffic patterns. Hence, this forecasting application employs GRU, a type of recurrent neural network (RNN), renowned for its effectiveness in modeling sequential data.
   
    """,
    unsafe_allow_html=True
)

# Load the data
data = pd.read_csv("traffic.csv")
data["DateTime"] = pd.to_datetime(data["DateTime"])
data = data.drop(["ID"], axis=1)

# Display Title
st.subheader("Time series - Traffic On Junctions Over Years")

# Plotly Express Line Plot
fig4 = px.line(data, x='DateTime', y='Vehicles', color='Junction',
              title='Traffic On Junctions Over Years',
              labels={'Vehicles': 'Number of Vehicles', 'DateTime': 'Date'},
              line_shape='linear', template='plotly_dark')

# Show the plot using st.plotly_chart
st.plotly_chart(fig4)

# df to be used for EDA
df = data.copy()

# Exploring more features
df["Year"] = df['DateTime'].dt.year
df["Month"] = df['DateTime'].dt.month
df["Date_no"] = df['DateTime'].dt.day
df["Hour"] = df['DateTime'].dt.hour
df["Day"] = df.DateTime.dt.strftime("%A")

# Display the initial time series plot
st.subheader("Time Series - Analysis on Trend and Seasonality")
st.subheader("Choose a Plot to Display:")
plot_options = ["Traffic vs Year", "Traffic vs Month", "Traffic vs Date_no", "Traffic vs Hour", "Traffic vs Day"]
selected_plot = st.selectbox("Select Plot", plot_options)

# Plot the selected time series
st.subheader(f"{selected_plot}")
#colors = ["#FFD4DB", "#BBE7FE", "#D3B5E5", "#dfe2b6"]
colors = ["#A52A2A", "#FFFF00", "#3357FF", "#00FF00"]
plt.figure(figsize=(20, 4), facecolor="orange")

if "Year" in selected_plot:
    feature_column = "Year"
elif "Month" in selected_plot:
    feature_column = "Month"
elif "Date_no" in selected_plot:
    feature_column = "Date_no"
elif "Hour" in selected_plot:
    feature_column = "Hour"
elif "Day" in selected_plot:
    feature_column = "Day"
    
if selected_plot in ["Traffic vs Year", "Traffic vs Month", "Traffic vs Date_no", "Traffic vs Hour", "Traffic vs Day"]:
    feature_column = selected_plot.split("vs ")[1]
    feature_df = df[[feature_column, 'DateTime', 'Vehicles', 'Junction']]
    Time_series = sns.lineplot(x=feature_df[feature_column], y="Vehicles", data=feature_df, hue="Junction", palette=colors)
    Time_series.set_title(f"Traffic vs {feature_column}")
    Time_series.set_ylabel("Number of Vehicles")
    Time_series.set_xlabel(f"{feature_column}")
    st.pyplot(plt)  # Display the selected time series plot
    
#@st.cache_data
# Function to inverse difference and normalize
def inverse_transform(last_ob, value, average, std):
    inverted = value + last_ob
    inverted = (inverted * std) + average
    return inverted

# Define or load RMSE values
RMSE_J1 = np.load("junction_1_rmse.npy")
RMSE_J2 = np.load("junction_2_rmse.npy")
RMSE_J3 = np.load("junction_3_rmse.npy")
RMSE_J4 = np.load("junction_4_rmse.npy")

# Load the saved predictions (assuming they are saved as .npy)
PredJ1 = np.load("junction_1_predictions.npy")
PredJ2 = np.load("junction_2_predictions.npy")
PredJ3 = np.load("junction_3_predictions.npy")
PredJ4 = np.load("junction_4_predictions.npy")

# Load the test datasets
X_testJ1 = np.load("junction_1_test_features.npy")
X_testJ2 = np.load("junction_2_test_features.npy")
X_testJ3 = np.load("junction_3_test_features.npy")
X_testJ4 = np.load("junction_4_test_features.npy")

# Load the test datasets
y_testJ1 = np.load("junction_1_test_y.npy")
y_testJ2 = np.load("junction_2_test_y.npy")
y_testJ3 = np.load("junction_3_test_y.npy")
y_testJ4 = np.load("junction_4_test_y.npy")

# Display RMSE values
st.subheader("Root Mean Squared Error (RMSE) for Each Junction:")
st.write(f"Junction 1: {RMSE_J1:.4f}")
st.write(f"Junction 2: {RMSE_J2:.4f}")
st.write(f"Junction 3: {RMSE_J3:.4f}")
st.write(f"Junction 4: {RMSE_J4:.4f}")

# Display predictions and original values
st.subheader("Select any Junction")
# You can choose a specific junction
junction_choice = st.selectbox("Select Junction", ["Junction 1", "Junction 2", "Junction 3", "Junction 4"])

# Choose the predictions and test data accordingly
if junction_choice == "Junction 1":
    selected_predictions = PredJ1
    selected_test_data = X_testJ1
    true_values = y_testJ1
    selected_index = 0
elif junction_choice == "Junction 2":
    selected_predictions = PredJ2
    selected_test_data = X_testJ2
    true_values = y_testJ2
    selected_index = 1
elif junction_choice == "Junction 3":
    selected_predictions = PredJ3
    selected_test_data = X_testJ3
    true_values = y_testJ3
    selected_index = 2
else:
    selected_predictions = PredJ4
    selected_test_data = X_testJ4
    true_values = y_testJ4
    selected_index = 3
    
# Main Heading with styling
st.markdown(
    """
    <h2 style='color: cyan;'>Traffic Forecast for the Next Time Step:</h2>
    """,
    unsafe_allow_html=True
)    

# You can choose a specific time step
time_step_choice = st.selectbox("Select Time Step", range(len(selected_test_data)))

# Check if a time step is selected
if st.button("Show Forecast"):
    # Invert the differences and normalization
    last_ob = selected_test_data[time_step_choice][-1]
    average = np.load(f"junction_{selected_index + 1}_average.npy")
    std = np.load(f"junction_{selected_index + 1}_std.npy")
    forecast_final = inverse_transform(last_ob, selected_predictions[time_step_choice], average, std)

    # Debugging information
    #st.write("Debugging Info:")
    #st.write(f"last_ob: {last_ob}")
    #st.write(f"average: {average}")
    #st.write(f"std: {std}")
    #st.write(f"forecast_final: {forecast_final}")

    # Display the forecast using Plotly scatter plot
    st.write(f"Forecasted Traffic for {junction_choice} at Time Step {time_step_choice}:")
    fig1 = px.scatter(x=[time_step_choice], y=[forecast_final[0]], title=f"Forecasted Traffic for {junction_choice} at Time Step {time_step_choice}")
    fig1.update_layout(xaxis_title="Time Step", yaxis_title="Forecasted Traffic")
    st.plotly_chart(fig1, use_container_width=True)
    
    # Create a DataFrame with true and predicted values
    df_predictions = pd.DataFrame({"True Values": true_values.flatten(), "Predicted Values": selected_predictions.flatten()})

    # Display predictions vs true values
    st.subheader("Traffic Predictions vs True Values:")
    #st.write(df_predictions)

    # Create a scatter plot using Plotly with different colors for true and predicted values
    fig2 = px.scatter(df_predictions, x="True Values", y="Predicted Values", title="Predictions vs True Values", color=df_predictions.index, color_discrete_map={"True Values": "red", "Predicted Values": "blue"})
    fig2.update_layout(xaxis_title="True Values", yaxis_title="Predicted Values")
    st.plotly_chart(fig2)
 
    # Load Transform_reverssed_J1_J2_J3_J4 
    Transform_reverssed_J1_new = pd.read_csv('Transform_reverssed_J1_new.csv')
    Transform_reverssed_J2_new = pd.read_csv('Transform_reverssed_J2_new.csv')
    Transform_reverssed_J3_new = pd.read_csv('Transform_reverssed_J3_new.csv')
    Transform_reverssed_J4_new = pd.read_csv('Transform_reverssed_J4_new.csv')
    
    # Create a Plotly figure based on the selected junction
    if junction_choice == "Junction 1":
        #dates = Transform_reverssed_J1.index.tolist()
        dates = Transform_reverssed_J1_new['DateTime'].tolist()
        y_values = Transform_reverssed_J1_new['Pred_Final'].tolist()
        title = "Prediction for Junction 1"
    elif junction_choice == "Junction 2":
        #dates = Transform_reverssed_J2.index.tolist()
        dates = Transform_reverssed_J2_new['DateTime'].tolist()
        y_values = Transform_reverssed_J2_new['Pred_Final'].tolist()
        title = "Prediction for Junction 2"
    elif junction_choice == "Junction 3":
        #dates = Transform_reverssed_J3.index.tolist()
        dates = Transform_reverssed_J3_new['DateTime'].tolist()
        y_values = Transform_reverssed_J3_new['Pred_Final'].tolist()
        title = "Prediction for Junction 3"
    elif junction_choice == "Junction 4":
        #dates = Transform_reverssed_J4.index.tolist()
        dates = Transform_reverssed_J4_new['DateTime'].tolist()
        y_values = Transform_reverssed_J4_new['Pred_Final'].tolist()
        title = "Prediction for Junction 4"

    # Create a Plotly figure
    fig3 = px.line(x=dates, y=y_values, title=title)
    fig3.update_layout(xaxis_title="Date", yaxis_title="Prediction")

    # Display the figure using st.plotly_chart
    st.plotly_chart(fig3)


















