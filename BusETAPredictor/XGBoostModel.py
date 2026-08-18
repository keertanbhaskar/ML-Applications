import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('bus_eta_gps_logs.csv')
print(data.head(5))
print(data.shape)
print(data.isnull().sum())


features = [
    "distance_to_stop_km","speed_kmph","avg_speed_kmph","hour","traffic_level","day_of_week"
]

target  = 'eta_minutes'

X = data[features]
y = data[target]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = XGBRegressor(n_estimators = 80, learning_rate = 0.05,random_state=42)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print('mean absolute error:',mean_absolute_error(y_test,y_pred))

# New data
new_data = [[2.5, 27, 27, 8, 1, 1]]
prediction = model.predict(new_data)
print("Predicted ETA:", prediction[0], "minutes")