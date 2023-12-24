import pandas as pd
import numpy as np
import tensorflow as tf


class Model:
    def __init__(self):

        self.train = self.read_data("./train.csv", test=False)
        self.test = self.read_data("./test.csv", test=True)

        self.train_proccesed = self.pre_process(self.train)
        self.test_proccesed = self.pre_process(self.test)

        self.build_model()
        self.train_model()

        self.test_trained_model()

    def class_to_number(self, lst):
        for n, k in enumerate(lst.unique().tolist()):
            if pd.isna(k):
                continue
            lst = lst.replace(k, n)

        return lst

    def read_data(self, path, test=False):
        data = pd.read_csv(path)

        data["Group"] = data["PassengerId"].str.split("_").str[0].astype(int)
        data["N in Group"] = data["PassengerId"].str.split(
            "_").str[1].astype(int)
        data["HomePlanetN"] = data["HomePlanet"].replace("Europa", 1).replace(
            "Earth", 2).replace("Mars", 3).fillna(0).astype(int)
        data["SleepN"] = data["CryoSleep"].fillna(0).astype(int)

        data["CabinD"] = self.class_to_number(data["Cabin"].str.split(
            "/").str[0]).fillna(method="ffill").astype(int)
        data["CabinN"] = self.class_to_number(data["Cabin"].str.split(
            "/").str[1]).fillna(method="ffill").astype(int)
        data["CabinS"] = self.class_to_number(data["Cabin"].str.split(
            "/").str[2]).fillna(method="ffill").astype(int)

        data["DestitantionN"] = self.class_to_number(
            data["Destination"]).fillna(method="ffill").astype(int)

        data["AgeN"] = data["Age"].fillna(method="ffill").astype(int)

        data["VIPN"] = data["VIP"].fillna(method="ffill").astype(int)

        data.index = data["PassengerId"]

        if test:
            data = data[["Group", "N in Group", "HomePlanetN", "SleepN", "CabinD", "CabinN", "CabinS", "DestitantionN",
                         "VIPN", "AgeN", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]]
            data = data.drop(columns=["HomePlanetN", "VIPN", "AgeN",
                             "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"])
            return data

        else:
            data["TransportedN"] = data["Transported"].fillna(
                method="ffill").astype(int)

            data = data[["Group", "N in Group", "HomePlanetN", "SleepN", "CabinD", "CabinN", "CabinS", "DestitantionN",
                        "VIPN", "AgeN", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "TransportedN"]]

            return data

    def pre_process(self, data):
        data_min = data.min()
        data_max = data.max()
        data = (data - data_min) / (data_max - data_min)

        data_mean = data.mean()
        data_std = data.std()
        data = (data - data_mean) / data_std

        return data

    def build_model(self):
        self.model = tf.keras.models.Sequential()

        self.model.add(
            tf.keras.layers.Dense(420, input_shape=(
                self.train.shape[1]-1,), activation="relu")
        )
        self.model.add(tf.keras.layers.Dense(310, activation="relu"))
        self.model.add(tf.keras.layers.Dense(210, activation="relu"))
        self.model.add(tf.keras.layers.Dense(210, activation="relu"))
        self.model.add(tf.keras.layers.Dense(210, activation="relu"))
        self.model.add(tf.keras.layers.Dense(210, activation="relu"))
        self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        self.model.compile(optimizer="Adam", loss="binary_crossentropy")

    def train_model(self):
        x_train, y_train = self.train.drop(
            columns=["TransportedN"]), self.train["TransportedN"]

        earlystop = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=100, restore_best_weights=True)

        self.model.fit(x_train, y_train, validation_split=0.2,
                       epochs=100, batch_size=50, callbacks=[earlystop])

    def test_trained_model(self):
        x_train, y_train = self.train.drop(
            columns=["TransportedN"]), self.train["TransportedN"]

        y_pred = self.model.predict(x_train)
        y_pred = y_pred.round()

        correct, total, precent = 0, 0, 0
        for i in range(y_pred.shape[0]):
            if y_pred[i] == y_train[i]:
                correct += 1
            total += 1

        precent = correct/total * 100

        print(correct, total, precent)


model = Model()
