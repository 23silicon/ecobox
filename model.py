import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import warnings
from pandas.errors import PerformanceWarning

warnings.filterwarnings("ignore",category=PerformanceWarning)

df = pd.read_csv("data.csv")


#plt.plot(df["precip_mm"])
#plt.show()


#Getting and preprocessing the rainfall data from datacsv
hrs = 1  # e.g. 1-hour lag
periods_per_hour = 468 // 24  # if evenly spaced
for h in range(1, 24*7 + 1, hrs):
    df[f"lag_{h}h"] = df["precip_mm"].shift(h * periods_per_hour)

#for finding out the number of stations to divide them out of the final number
#stationset = set(df["station"])
#print(stationset)
#print(len(stationset))
del df["station"]
df.dropna(inplace=True)
X = df[[f"lag_{h}h" for h in range(1, 24*7 + 1)]]   
y = df["precip_mm"]


#Split the training into april 2025 and the testing data (what the model is predicting) to may 2025
split_index = int(len(df) * 0.5)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Model: XGBoosting algorithm
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

#some debug checks
#print(f"RMSE: {rmse:.3f} mm")
#print(df.shape)



plt.figure(figsize=(12, 4))
y_actual_full = pd.concat([y_train, y_test])
y_pred_full = np.concatenate([model.predict(X_train), model.predict(X_test)])


avs = []
i = 0

#chunks out each day of predictions, takes their averages, and sums them to get total rainfall for a month.
while i < len(y_pred):
    if i+468 < len(y_pred):
        avs.append(np.mean(y_pred[i:i+468]))
        i += 468
    elif i < len(y_pred)-1:
        avs.append(np.mean(y_pred[i:len(y_pred)-1]))
        i+= 468
    else:
        break
        
av = np.sum(avs)*4 #adjustment factor for limitations of station numbers that I added
print(f"-------\nPredicted Rainfall for May 2024 based on April 2024 data and rolling predictions: {av}mm\n--------")



#Plotting the results
plt.plot(y_actual_full.index, y_actual_full.values, label='Actual')
plt.plot(y_actual_full.index[split_index:], y_pred_full[split_index:], label='Predicted', alpha=0.7)
plt.title("Rainfall Forecasting (Test Set)")
plt.xlabel("Time Step")
plt.ylabel("Precipitation (mm)")
plt.legend()
plt.show()

#prints the testing range
print("Train range:", df.iloc[:split_index]["date"].min(), "-", df.iloc[:split_index]["date"].max())
print("Test range:", df.iloc[split_index:]["date"].min(), "-", df.iloc[split_index:]["date"].max())


