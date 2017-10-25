import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator

number_of_classes = 7
dimension = 48
number_of_channels = 1

def load_dataset(file):
    dataset_features = []
    dataset_labels = []

    #file = '/input' + file

    with open(file) as csvfile:
        csv_reader_object = csv.reader(csvfile, delimiter=',')
        count = 0
        next(csv_reader_object, None)
        for row in csv_reader_object:
            if len(row) == 0 :
                _0 = 0  # ignore
            else:
                #print(row)
                dataset_features.append(row[1].split())
                # print(count)
                # count += 1
                temp = np.zeros(7, dtype=int)
                temp[int(row[0])] = int(1)
                dataset_labels.append(temp)

    return np.array(dataset_features), np.array(dataset_labels)

def get_next_batch(dataset_features, dataset_labels, batch_index, batch_size):
    return dataset_features[batch_index*batch_size:(batch_index+1)*batch_size, :], dataset_labels[batch_index*batch_size : (batch_index+1)*batch_size, :]

train_features,train_labels = load_dataset("training.csv")
#train_features = train_features.astype(int)
#train_features = train_features/255.0
test_features,test_labels = load_dataset("test.csv")
#test_features = test_features.astype(int)
#test_features = test_features/255.0
validation_features,validation_labels = load_dataset("testprivate.csv")
#validation_features = validation_features.astype(int)
#validation_features = validation_features/255.0


train_features = train_features.reshape((-1, 48, 48, 1))
validation_features = validation_features.reshape((-1, 48, 48, 1))
test_features = test_features.reshape((-1, 48, 48, 1))



# Three steps to create a CNN
# 1. Convolution
# 2. Activation
# 3. Polling
# Repeat Steps 1,2,3 for adding more hidden layers

# 4. After that make a fully connected network
# This fully connected network gives ability to the CNN
# to classify the samples

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(48, 48 ,1), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(number_of_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

epochs = 10
lrate = 0.01
decay = lrate/epochs
adam = Adam(decay=decay)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
print(model.summary())
model.fit(train_features, train_labels, validation_data=(validation_features, validation_labels), epochs=epochs, batch_size=50)
# Final evaluation of the model
scores = model.evaluate(dataset_test_features, dataset_test_labels, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
