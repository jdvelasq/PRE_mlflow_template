# Prepare data
import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_data(file_path, test_size, random_state):

    df = pd.read_csv(file_path)
    y = df["quality"]
    x = df.drop(columns=["quality"])

    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
    )
