from pathlib import Path

from src.model_service import load_model_bundle, predict_rul as _predict_rul


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


# Load and validate the final trained model (type, feature order, metrics fingerprint).
bundle = load_model_bundle(MODELS_DIR)


def predict_rul(sensor_data):
    """
    Predict Remaining Useful Life (RUL).

    Parameters
    ----------
    sensor_data : dict
        Dictionary containing the 15 raw (unscaled) sensor values.

    Returns
    -------
    float
        Predicted RUL in cycles.
    """

    return _predict_rul(bundle.model, sensor_data)


if __name__ == "__main__":

    # Example sensor input
    sample_data = {
        "sensor_2": 642.15,
        "sensor_3": 1580.20,
        "sensor_4": 1400.50,
        "sensor_6": 21.60,
        "sensor_7": 554.30,
        "sensor_8": 2388.10,
        "sensor_9": 9040.20,
        "sensor_11": 47.30,
        "sensor_12": 522.10,
        "sensor_13": 2388.00,
        "sensor_14": 8140.50,
        "sensor_15": 8.50,
        "sensor_17": 392.00,
        "sensor_20": 39.00,
        "sensor_21": 23.50
    }

    predicted_rul = predict_rul(sample_data)

    print(f"Predicted RUL: {predicted_rul:.2f} cycles")
