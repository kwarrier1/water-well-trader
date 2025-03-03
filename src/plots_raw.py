import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def get_merged_data() -> pd.DataFrame:
    # File path strings
    well_data_file_path = '/Users/keshavwarrier/water-well-trader/data/processed-well-data.csv'
    etp_data_file_path = '/Users/keshavwarrier/water-well-trader/data/etp-data.csv'

    # Read data into pd dataframes
    X = pd.read_csv(well_data_file_path, index_col=False)
    y = pd.read_csv(etp_data_file_path, index_col=False)

    # Deal with extra index col
    X = X.iloc[:, 1:]

    return pd.merge(X, y, on="Date", how="inner")

def scale_data (merged_data: pd.DataFrame, scaler) -> pd.DataFrame:
    # Don't scale the date! :)
    scaled_values = scaler.fit_transform(merged_data.iloc[:, 1:])
    scaled_data = pd.DataFrame(scaled_values, columns=merged_data.columns[1:])
    scaled_data.insert(0, "Date", merged_data["Date"])
    return scaled_data

def gwe_time(merged_data: pd.DataFrame) -> None:
    # Plot Mean GWE over Time
    fig_gwe = px.line(
        merged_data, x="Date", y="Mean GWE",
        title="Groundwater Elevation Over Time",
        labels={"Mean GWE": "Mean Groundwater Elevation (ft)", "Date": "Date"}
    )
    fig_gwe.show()

def etp_time(merged_data: pd.DataFrame) -> None:
    # Plot ETP prices over Time
    fig_stocks = px.line(
        merged_data, x="Date", y=merged_data.columns[2:],
        title="ETP Price Over Time",
        labels={"value": "Price", "variable": "Stock"}
    )
    fig_stocks.show()

def gwe_etp(merged_data: pd.DataFrame) -> None:
    # Mean GWE vs. ETP Prices
    for stock in merged_data.columns[2:]:
        fig_scatter = px.scatter(
            merged_data, x="Mean GWE", y=stock,
            title=f"{stock} ETP Price vs. Mean Groundwater Elevation",
            labels={"Mean GWE": "Mean Groundwater Elevation (ft)", stock: "Price"}
        )
        fig_scatter.show()

# Scatter Plot: Scaled Mean GWE vs. Scaled Stock Prices
def plot_gwe_vs_stock(etp: str, merged_data: pd.DataFrame) -> None:
    fig_scatter = px.scatter(
        merged_data, x="Mean GWE", y=etp,
        title=f"{etp} Price vs. Groundwater Elevation",
        labels={"Mean GWE": "Groundwater Elevation", etp: "Price"}
    )
    fig_scatter.show()

def plot_gwe_vs_AVO(merged_data: pd.DataFrame) -> None:
    plot_gwe_vs_stock("AVO", merged_data)

def plot_gwe_vs_CDZI(merged_data: pd.DataFrame) -> None:
    plot_gwe_vs_stock("CDZI", merged_data)

def plot_gwe_vs_CWT(merged_data: pd.DataFrame) -> None:
    plot_gwe_vs_stock("CWT", merged_data)

def plot_gwe_vs_FPI(merged_data: pd.DataFrame) -> None:
    plot_gwe_vs_stock("FPI", merged_data)

def plot_gwe_vs_ZCF(merged_data: pd.DataFrame) -> None:
    plot_gwe_vs_stock("ZC=F", merged_data)

merged_data = get_merged_data()
scaled_data = scale_data(merged_data, MinMaxScaler())
gwe_etp(merged_data)