import yfinance as yf
import pandas as pd
import os

TICKERS = [
    "CPA1T.TL", "ARC1T.TL", "EFT1T.TL", "EEG1T.TL", "MRK1T.TL",
    "NCN1T.TL", "HAE1T.TL", "LHV1T.TL", "PKG1T.TL", "APG1L.VS",
    "AKO1L.VS", "GRG1L.VS", "IGN1L.VS", "NTU1L.VS", "DGR1R.RG"
]

START_DATE = "2021-01-01"
END_DATE = pd.Timestamp.now().date()

def download_data():
    print("Downloading Baltic stock prices")
    data = yf.download(TICKERS, start=START_DATE, end=END_DATE)["Close"]
    
    os.makedirs("data", exist_ok=True)
    data.to_csv("data/baltic_prices.csv")
    print(f"Saved {data.shape[0]} rows x {data.shape[1]} stocks to data/baltic_prices.csv")
    return data

if __name__ == "__main__":
    download_data()