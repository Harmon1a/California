import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as pyplot
from keras.models import Sequential
from keras.initializers import glorot_normal
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

housingData = pd.read_csv("cal_housing.csv")


X = housingData.iloc[:, :-1].values
y = housingData.iloc[:, [-1]].values


from sklearn.impute import SimpleImputer
missingValueImputer = SimpleImputer()
X[:, :-1] = missingValueImputer.fit_transform(X[:, :-1])
y = missingValueImputer.fit_transform(y)

from sklearn.preprocessing import LabelEncoder
X_labelencoder = LabelEncoder()
X[:, -1] = X_labelencoder.fit_transform(X[:, -1])
X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.2, random_state=2)

model = Sequential()
model.add(Dense(30, input_dim=9, activation="relu"))
model.add(Dense(15, activation="relu"))
model.add(Dense(1))
model.compile(optimizer='rmsprop', loss='mse')
history = model.fit(X_train, Y_train, epochs=1000, validation_data=(X_val, Y_val), verbose=2)
pyplot.title('Loss / Mean Squared Error')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
housingData2 = pd.read_csv("cal_housing2.csv")
predict = model.predict(X_val)
pre=pd.DataFrame(predict)
pre.to_excel("pre123.xlsx")
from sklearn.metrics import r2_score
#print(r2_score(housingData2["median_house_value"],pre))
print(r2_score(Y_val,pre))
print(mean_squared_error(Y_val, predict))
