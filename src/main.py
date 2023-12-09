import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout

from data_handler import get_data_frame

def main():

    df = get_data_frame('ysb_data.csv')
    df = df.dropna(subset=['bv_color', 'ri_code', 'ri_color', 'spect_code', 'ub_color'])

    X = df[['vmag', 'bv_color', 'ri_color', 'ub_color']]
    y = df['spect_code']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Encoding the target variable
    encoder = LabelEncoder()
    encoded_Y = encoder.fit_transform(y)
    y_one_hot = to_categorical(encoded_Y)

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_one_hot, test_size=0.2, random_state=220301)

    # Create model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=4))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(y_one_hot.shape[1], activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Fit the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=256)

    # Evaluate the model
    scores = model.evaluate(X_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

if __name__ == "__main__":
    main()

