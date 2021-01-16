# Import libraries
import keras
import numpy as np
import math
from keras.models import Sequential 
from keras.utils import to_categorical
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D
from keras.layers import Dropout, BatchNormalization
from keras.losses import categorical_crossentropy
from data import prepare
import matplotlib.pyplot as plt

# Define pre-proccessed data
covid_scans = prepare("covid")
non_covid_scans = prepare("non_covid")

# Assign labels to differentiate between covid(1) and non-covid(0) scans
covid_labels = np.array([1 for _ in range(len(covid_scans))])
non_covid_labels = np.array([0 for _ in range(len(non_covid_scans))])

# Split data: 80% train and 20% Test
x_scale = math.floor(len(covid_labels)*.8)
y_scale = math.floor(len(non_covid_labels)*.8)

# Join covid and non_covid data, split data
x_train = np.concatenate((covid_scans[:x_scale], non_covid_scans[:y_scale]))
y_train = np.concatenate((covid_labels[:x_scale], non_covid_labels[:y_scale]))
x_val = np.concatenate((covid_scans[x_scale:], non_covid_scans[y_scale:]))
y_val = np.concatenate((covid_labels[x_scale:], non_covid_labels[y_scale:]))

# visualizing one CT scan
data = x_train[0]
plt.imshow(np.squeeze(data[:, :, 15]), cmap='gray')
plt.show()

# Reshape for 3D CNN model
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],
                          x_train.shape[2], x_train.shape[3], 1)

x_val = x_val.reshape(x_val.shape[0], x_train.shape[1],
                      x_val.shape[2], x_val.shape[3], 1)

# Encode target labels 
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

print("Proccessing completed...")

# BUILDING 3D CNN
sample_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3], 1)
model = Sequential()
model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=sample_shape))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(BatchNormalization(center=True, scale=True))
model.add(Dropout(0.5))
model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(BatchNormalization(center=True, scale=True))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(2, activation='softmax'))
print("Model successfully defined...")


# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.0001),
              metrics=['accuracy'],
              )

# Model summary
model.summary()
print("Model compiled...")

# Fit data to model
model = model.fit(x_train, y_train, 
                  batch_size=2, 
                  epochs=10,
                  validation_data=(x_val, y_val),
                  verbose=1)















