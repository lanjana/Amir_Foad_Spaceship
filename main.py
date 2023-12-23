import pandas as pd
import numpy as np
# import tensorflow as tf


def preprocess(path):
    data = pd.read_csv(path)

    data["Group"] = data["PassengerId"].str.split("_").str[0].astype(int)
    data["N in Group"] = data["PassengerId"].str.split("_").str[1].astype(int)
    data["HomePlanetN"] = data["HomePlanet"].replace(
        "Europa", 1).replace("Earth", 2).replace("Mars", 3).fillna(0).astype(int)
    data["SleepN"] = data["CryoSleep"].fillna(0).astype(int)

    data["CabinD"] = data["Cabin"].str.split("/").str[0].replace("A", 1).replace("B", 2).replace("C", 3).replace(
        "D", 4).replace("E", 5).replace("F", 6).replace("G", 7).replace("T", "8").fillna(0).astype(int)
    data["CabinN"] = data["Cabin"].str.split(
        "/").str[1].fillna(method="ffill").astype(int)
    data["CabinS"] = data["Cabin"].str.split(
        "/").str[2].replace("P", 0).replace("S", 1).fillna()

    print(data["CabinN"].unique())

    return data


data = preprocess("./train.csv")


print(data)
