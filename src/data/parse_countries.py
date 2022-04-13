import pandas as pd
import numpy as np

import os
import glob

def parser(country_code: str):

    dir = f"./data/helper data/{country_code}/"

    files = glob.glob(dir + "/*.csv")

    for filename in files:

        df_total = pd.DataFrame()

        generation = pd.read_csv(filename)

        generation["Date"] = generation["MTU"].apply(lambda x: x.split("-",1)[0].split(" ",1)[0])
        generation["Time"] = generation["MTU"].apply(lambda x: x.split("-",1)[0].split(" ",1)[1])

        generation.index = pd.to_datetime(generation["Date"] + " " + generation["Time"])

        generation = generation.drop(columns=["Area", "MTU", "Date", "Time"])

        filter_col = [col for col in generation if col.endswith((' - Actual Aggregated [MW]'))]
        generation = generation[filter_col]

        generation = generation.rename(columns=lambda x: x.split("-",1)[0].strip())
        generation = generation.replace(to_replace=['n/e', '-'], value=0)
        generation = generation.astype(np.float64)

        df_total = pd.concat([df_total, generation])

        df_total.to_csv(f"./data/generation/generation_{country_code}.csv")
    
    return