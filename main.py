import pandas as pd, numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import mlflow, mlflow.tensorflow

# Load data
data = pd.read_csv("vle_data_margules.csv")
X = data[["x1","T","P"]].values
y = data["y1"].values.reshape(-1,1)

scalerX, scalery = MinMaxScaler(), MinMaxScaler()
Xn, yn = scalerX.fit_transform(X), scalery.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(Xn, yn, test_size=0.2, random_state=42)

# ANN
mlflow.tensorflow.autolog()
with mlflow.start_run():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=(3,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid") 
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(X_train, y_train, epochs=40, validation_split=0.2, verbose=0)

    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    mlflow.log_metric("test_mae", float(test_mae))
    model.save("ann_vle_model.h5")

import matplotlib.pyplot as plt

y_pred = model.predict(X_test)
plt.scatter(scalery.inverse_transform(y_test), scalery.inverse_transform(y_pred))
plt.xlabel("y1 Experimental")
plt.ylabel("y1 ANN Predicted")
plt.title("Parity Plot")
plt.savefig("parity.png")