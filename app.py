import json
import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, jsonify, Response
from model import download_data, format_data, train_model
from config import model_file_path

app = Flask(__name__)

currencies = ["ETH", "BTC", "SOL", "BNB", "ARB"]


def update_data():
    """Download price data, format data and train model."""
    for currency in currencies:
        download_data(currency)
        format_data(currency)
        train_model(currency)


def get_inference(currency_name):
    """Load model and predict current price."""
    currency_model_file_path = os.path.join(
        "inference-data", f"{currency_name}_model.pkl"
    )
    if not os.path.exists(currency_model_file_path):
        raise FileNotFoundError(
            f"Model file for {currency_name} not found at {currency_model_file_path}"
        )

    with open(currency_model_file_path, "rb") as f:
        model = pickle.load(f)

    # Assuming you have some input data to make predictions
    # For example, let's use the current timestamp as input
    current_timestamp = datetime.now().timestamp()
    prediction = model.predict([[current_timestamp]])

    return prediction[0][0]


@app.route("/inference/<string:token>")
def generate_inference(token):
    """Generate inference for given token."""
    try:
        price = get_inference(token)
        return jsonify({"currency": token, "predicted_price": price})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/update")
def update():
    """Update data and return status."""
    try:
        update_data()
        return "0"
    except Exception:
        return "1"


if __name__ == "__main__":
    update_data()
    app.run(host="0.0.0.0", port=8000)
