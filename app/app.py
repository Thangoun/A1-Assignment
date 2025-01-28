import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import pickle

# Load the pre-trained model
with open("model/A1_car_selling_price.model", "rb") as f:
    loaded_file = pickle.load(f)

# Extract components from the loaded file
model = loaded_file['model']
scaler = loaded_file['scaler']
max_power_default = loaded_file['max_power_default']
mileage_default = loaded_file['mileage_default']

# Initialize Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.Div([
        html.H1("Car Price Prediction", style={"textAlign": "center", "fontSize": "32px", "fontWeight": "bold"}),
        html.Div([
            html.H3([
                "Instruction:",  # First line
                html.Br(),  # Line break
                "Please fill out the fields below. If you don't know, leave them blank to use default values."  # Second line
            ], style={"textAlign": "left", "marginBottom": "20px", "fontSize": "18px", "color": "#555"}),
        ], style={
            "width": "50%",  # Matches the width of the input divs
            "margin": "auto",  # Centers the content horizontally
            "textAlign": "left"  # Aligns the text within the div
        }),
    ], style={"marginBottom": "30px"}),

    html.Div([
        html.Div([
            html.Label("year"),
            dcc.Input(id="input-year", type="number", placeholder="Enter year", required=True, style={"width": "100%", "marginTop": "10px"}),
        ], style={"marginBottom": "30px"}),

        html.Div([
            html.Label("mileage(kmpl):"),
            dcc.Input(id="input-mileage", type="number", placeholder=f"Enter mileage (default: {mileage_default})", required=False, style={"width": "100%", "marginTop": "10px"}),
        ], style={"marginBottom": "30px"}),

        html.Div([
            html.Label("max power(bhp):"),
            dcc.Input(id="input-maxpower", type="number", placeholder=f"Enter max power (default: {max_power_default})", required=False, style={"width": "100%", "marginTop": "10px"}),
        ], style={"marginBottom": "30px"}),

        html.Button("predict", id="predict-button", n_clicks=0, style={"marginTop": "20px"}),
    ], style={
        "width": "50%",
        "margin": "auto",
        "padding": "20px",
        "borderRadius": "10px",
        "textAlign": "left"  # Aligns all content to the left within the div
    }),

    html.Div(id="output-prediction", style={"textAlign": "center", "marginTop": "20px", "fontSize": "20px"})
])


# Callback to handle prediction
@app.callback(
    Output("output-prediction", "children"),
    Input("predict-button", "n_clicks"),
    State("input-year", "value"),
    State("input-mileage", "value"),
    State("input-maxpower", "value")
)
def predict_price(n_clicks, year, mileage, max_power):
    if n_clicks > 0:
        # Ensure the 'year' input is filled
        if year is None:
            return "Please provide the year of manufacture to calculate the price."

        # Use default values if optional inputs are not provided
        if mileage is None:
            mileage = mileage_default
        if max_power is None:
            max_power = max_power_default

        # Prepare the input features as a DataFrame
        input_features = pd.DataFrame({
            "year": [year],
            "mileage": [mileage],
            "max_power": [max_power]
        })

        # Apply scaling to the input features
        scaled_features = scaler.transform(input_features)

        # Predict the car's price
        predicted_price = model.predict(scaled_features)[0]
        predicted_price = np.exp(predicted_price)  # Reverse the log transformation if applied

        # Format and return the result
        return f"The predicted selling price of the car is: {predicted_price:,.0f} Baht"
    return ""


# Run the app
if __name__ == "__main__":
    app.run_server(host='0.0.0.0')
