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

from data_handler import combined_data_frame

def main():
    
    # Get data frame
    df = combined_data_frame()
    
    # Get rid of rows where bv_color or vmag are NaN
    df = df.dropna(subset=['bv_color', 'vmag'])

    # Create x y variables
    X = df[['vmag', 'bv_color']]
    y = df['spectral_type']

    # label encode spectral type
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=220301)

    # Scaling the features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)  
    X_test_scaled = scaler.transform(X_test)        

    # Model ~ 80% accurate
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=2))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(10, activation='softmax'))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Fit the model
    history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=200, batch_size=256)

    # Evaluate the model
    scores = model.evaluate(X_test_scaled, y_test)
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

