import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from reward_model.rating_predictor import RatingPredictor


def _load_model(model_path, input_size):
    model = RatingPredictor(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Modell in den Evaluierungsmodus versetzen
    return model


def _predict(model, observation):
    with torch.no_grad():
        input_tensor = torch.tensor(observation, dtype=torch.float32)
        prediction = model(input_tensor.unsqueeze(0))  # Batch-Dimension hinzufügen
        return prediction.item()


def rate_observation(observation, puck_direction, puck_distance):
    observation = np.append(observation, [puck_distance, puck_direction])

    current_file_path = Path(__file__).resolve()
    current_dir = current_file_path.parent
    model_path = "trained_model.pth"
    model_path = os.path.join(current_dir, model_path)
    # input_size = 18
    input_size = 20
    model = _load_model(model_path, input_size)
    prediction = _predict(model, observation)
    return prediction
