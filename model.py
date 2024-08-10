import os
import pickle
from zipfile import ZipFile
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from updater import download_binance_monthly_data, download_binance_daily_data
from config import data_base_path, model_file_path
import glob


binance_data_path = os.path.join(data_base_path, "binance/futures-klines")


def download_data(currency_name):
    cm_or_um = "um"
    symbols = [f"{currency_name}USDT"]
    intervals = ["1d"]
    years = ["2020", "2021", "2022", "2023", "2024"]
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    download_path = binance_data_path
    download_binance_monthly_data(
        cm_or_um, symbols, intervals, years, months, download_path
    )
    print(f"Downloaded monthly data to {download_path}.")
    current_datetime = datetime.now()
    current_year = current_datetime.year
    current_month = current_datetime.month
    download_binance_daily_data(
        cm_or_um, symbols, intervals, current_year, current_month, download_path
    )
    print(f"Downloaded daily data to {download_path}.")

    # Extract ZIP files
    zip_files = glob.glob(os.path.join(download_path, "*.zip"))
    for zip_file in zip_files:
        with ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(download_path)
        # print(f"Extracted {zip_file} to {download_path}")

    # Log the files in the download path
    all_files = glob.glob(os.path.join(download_path, "*.csv"))
    # if not all_files:
    #     print(f"No CSV files found in {download_path} after extraction.")
    # else:
    #     print(f"CSV files found in {download_path}: {all_files}")


def format_data(currency_name):
    training_price_data_path = os.path.join(
        data_base_path, f"data/{currency_name}_price_data.csv"
    )
    binance_data_path = os.path.join(data_base_path, "binance/futures-klines")

    # Combine all CSV files in the binance_data_path into one DataFrame
    all_files = glob.glob(os.path.join(binance_data_path, "*.csv"))

    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {binance_data_path}")

    df_list = []
    expected_columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "count",
        "taker_buy_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    for file in all_files:
        df = pd.read_csv(file)
        if df.columns.tolist() != expected_columns:
            # print(f"Skipping {file} due to unexpected columns: {df.columns.tolist()}")
            continue
        # print(f"Columns in {file}: {df.columns.tolist()}")
        df_list.append(df)

    if not df_list:
        raise ValueError("No valid CSV files found with the expected columns.")

    combined_df = pd.concat(df_list, ignore_index=True)

    # Add 'date' column derived from 'open_time'
    combined_df["date"] = pd.to_datetime(combined_df["open_time"], unit="ms")

    # Log the first few rows of the combined DataFrame
    # print("Combined DataFrame preview:")
    # print(combined_df.head())

    # Save the combined DataFrame to the training_price_data_path
    os.makedirs(os.path.dirname(training_price_data_path), exist_ok=True)
    combined_df.to_csv(training_price_data_path, index=False)
    print(f"Formatted data saved to {training_price_data_path}")


def train_model(currency_name):
    training_price_data_path = os.path.join(
        data_base_path, f"data/{currency_name}_price_data.csv"
    )
    # Load the coin price data
    price_data = pd.read_csv(training_price_data_path)
    df = pd.DataFrame()

    # Convert 'date' to a numerical value (timestamp) we can use for regression
    df["date"] = pd.to_datetime(price_data["date"])
    df["date"] = df["date"].map(pd.Timestamp.timestamp)

    df["price"] = price_data[["open", "close", "high", "low"]].mean(axis=1)

    # Reshape the data to the shape expected by sklearn
    x = df["date"].values.reshape(-1, 1)
    y = df["price"].values.reshape(-1, 1)

    # Split the data into training set and test set
    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=0)

    # Train the model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # create the model's parent directory if it doesn't exist
    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)

    # Save the trained model to a file with a unique name
    currency_model_file_path = os.path.join(
        os.path.dirname(model_file_path), f"{currency_name}_model.pkl"
    )
    with open(currency_model_file_path, "wb") as f:
        pickle.dump(model, f)

    # print(f"Trained model for {currency_name} saved to {currency_model_file_path}")
