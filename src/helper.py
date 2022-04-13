import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class TimeoutException(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)

def split_years(df: pd.DataFrame, country_code: str) -> pd.DataFrame:
    df['year'] = df.index.year
    for y in df['year'].unique():
        globals()[f"df_{y}_{country_code}"] = df[df['year'] == y]
        globals()[f"df_{y}_{country_code}"].drop(labels=['year'], axis=1, inplace=True)
    number_of_years = df['year'].unique()
    return [globals()[f"df_{y}_{country_code}"] for y in df['year'].unique()], number_of_years

# Function to calculate correlation coefficient between two arrays
def corr(x, y, **kwargs):
    
    # Calculate the value
    coef = np.corrcoef(x, y)[0][1]
    # Make the label
    label = r'$\rho$ = ' + str(round(coef, 2))
    
    # Add the label to the plot
    ax = plt.gca()
    ax.annotate(label, xy = (0.2, 0.95), xycoords = ax.transAxes)

def parse_day_ahead_prices_ireland():

    day_ahead_prices = pd.read_csv("./data/external/helper data/Day-ahead Prices_202101010000-202201010000.csv")

    day_ahead_prices["Date"] = day_ahead_prices["MTU (CET)"].apply(lambda x: x.split("-",1)[0].split(" ",1)[0])
    day_ahead_prices["Time"] = day_ahead_prices["MTU (CET)"].apply(lambda x: x.split("-",1)[0].split(" ",1)[1])

    day_ahead_prices.index = pd.to_datetime(day_ahead_prices["Date"] + " " + day_ahead_prices["Time"])

    day_ahead_prices = day_ahead_prices.drop(columns=["MTU (CET)", "Date", "Time", "BZN|IE(SEM)"])
    day_ahead_prices.rename(columns={day_ahead_prices.columns[0]: "day_ahead_prices_IE"}, inplace=True)

    day_ahead_prices.to_csv(f"./data/raw/day_ahead_prices/day_ahead_prices_IE.csv")

    return

parse_day_ahead_prices_ireland()