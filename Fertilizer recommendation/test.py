
import pickle
import numpy as np

# Load the trained model
with open('Fertilizer_Recommendation_Model.pkl', 'rb') as file:
    model = pickle.load(file)

# Take input from the user
nitrogen = float(input("Enter Nitrogen value: "))
phosphorus = float(input("Enter Phosphorus value: "))
potassium = float(input("Enter Potassium value: "))

# Prepare input data
data = np.array([[nitrogen, phosphorus, potassium]])

# Predict fertilizer
ans = model.predict(data)

# Display result
fertilizer = {
    0: "TEN-TWENTY SIX-TWENTY SIX",
    1: "Fourteen-Thirty Five-Fourteen",
    2: "Seventeen-Seventeen-Seventeen",
    3: "TWENTY-TWENTY",
    4: "TWENTY EIGHT-TWENTY EIGHT",
    5: "DAP",
    6: "UREA"
}

print("Recommended Fertilizer:", fertilizer.get(ans[0], "Unknown fertilizer"))
