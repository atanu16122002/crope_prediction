
import pickle
import numpy as np

# Load the trained model
with open("dft.pkl", "rb") as file:
    dtr = pickle.load(file)

# Load the preprocessing object
with open("preprocesser.pkl", "rb") as file:
    preprocesser = pickle.load(file)


# Prediction function
def prediction(Year, average_rain_fall_mm_per_year,
               pesticides_tonnes, avg_temp, Area, Item):

    features = np.array([[
        Year,
        average_rain_fall_mm_per_year,
        pesticides_tonnes,
        avg_temp,
        Area,
        Item
    ]], dtype=object)

    # Transform input data
    transform_features = preprocesser.transform(features)

    # Predict crop yield
    predicted_yield = dtr.predict(transform_features)

    return predicted_yield[0]


# Test the model
result = prediction(
    2026,
    1485.0,
    121.0,
    16.37,
    "Albania",
    "Maize"
)

print("Predicted Crop Yield:", result)
print("Predicted Crop Yield (hg/ha):", round(result, 2))
