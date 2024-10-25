import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle


save_dir = r'D:\path_to_save'
model_path = os.path.join(save_dir, 'model.pkl')
scaler_path = os.path.join(save_dir, 'scaler.pkl')


if not os.path.exists(save_dir):
    os.makedirs(save_dir)

file_path = r'C:\Users\Nithishraj\Downloads\diabetes.csv' 
data = pd.read_csv(file_path)


X = data.drop('Outcome', axis=1)
y = data['Outcome']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression()
model.fit(X_train_scaled, y_train)


with open(model_path, 'wb') as model_file:
    pickle.dump(model, model_file)


with open(scaler_path, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler saved successfully.")


