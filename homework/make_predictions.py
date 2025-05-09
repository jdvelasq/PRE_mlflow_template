import mlflow
import pandas as pd

# $ python3 homework/make_predictions.py

FILE_PATH = "data/winequality-red.csv"

df = pd.read_csv(FILE_PATH)
y = df["quality"]
x = df.drop(columns=["quality"])

logged_model = "runs:/7e545a1fb76042b5bee3ab26921028a6/model"
loaded_model = mlflow.pyfunc.load_model(logged_model)
y = loaded_model.predict(x)

print(y)
