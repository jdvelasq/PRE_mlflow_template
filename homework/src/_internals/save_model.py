import os
import pickle


def save_model(model, save_path="models/estimator.pkl"):
    """Save the model to the specified path."""
    if not os.path.exists("models"):
        os.makedirs("models")
    with open(save_path, "wb") as file:
        pickle.dump(model, file)
