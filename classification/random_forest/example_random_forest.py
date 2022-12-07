# %% requirements
# pandas
# openpyxl
# scikit learn

# %% import library
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# %% import data
data = pd.read_excel('example_data.xlsx', engine='openpyxl')
X = data[['invar01', 'invar02', 'invar03', 'invar04', 'invar05']]
y = data[['outvar01']]

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# %% output data
# TODO warning
# DataConversionWarning: A column-vector y was passed when a 1d array was expected. 
# Please change the shape of y to (n_samples, ), for example using ravel().
output_class = ['high', 'medium', 'low']

le = LabelEncoder()
le.fit(output_class)
y = le.transform(y)

# %% train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape : {X_test.shape}")
print()
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape : {y_test.shape}")

# %% input data
scaler_minmax = MinMaxScaler()
# scaler_standard = StandardScaler()

scaler_minmax.fit(X_train)
X_scaled_minmax_train = scaler_minmax.transform(X_train)
X_scaled_minmax_test = scaler_minmax.transform(X_test)

# %% mdoel training
model = RandomForestClassifier()
model.fit(X_scaled_minmax_train, y_train)

# %% performance
pred_train = model.predict(X_scaled_minmax_train)
print(f"train score: {model.score(X_scaled_minmax_train, y_train):.2f}")

pred_test = model.predict(X_scaled_minmax_test)
print(f"test score: {model.score(X_scaled_minmax_test, y_test):.2f}")

# %%
# train
cfreport_train = classification_report(y_train, pred_train)
print(f"classification report - train \n{cfreport_train}")

# test
cfreport_test = classification_report(y_test, pred_test)
print(f"classification report - test \n{cfreport_test}")

# %%
