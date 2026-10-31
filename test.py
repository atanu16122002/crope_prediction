
import joblib

app = joblib.load('crop_predict_model')
arr = [[71,54,16,22.61,63.69,5.74,87.7]]  
my_predict = app.predict(arr)
print("Predict: ",my_predict)