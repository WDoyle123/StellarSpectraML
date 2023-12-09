import pandas as pd
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
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=220301)

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

    # Model accuracy value 
    final_accuracy = scores[1]
    
    # Create a figure and an axes object
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot training & validation accuracy values
    ax.plot(history.history['accuracy'])
    ax.plot(history.history['val_accuracy'])
    ax.set_title('Model accuracy')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    ax.legend(['Train', 'Test'], loc='lower right')

    # Add a horizontal line for the final accuracy
    ax.axhline(y=final_accuracy, color='r', linestyle='--')

    # Add an annotation
    ax.annotate(f'Final Accuracy: {final_accuracy:.2f}%', 
                 xy=(len(history.history['accuracy'])-1, final_accuracy), 
                 xytext=(len(history.history['accuracy'])/2, final_accuracy+5),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 horizontalalignment='right', verticalalignment='top')

    # Save the plot
    plt.savefig('../model_accuracy_plot.png', dpi=300)

    plt.show()

if __name__ == "__main__":
    main()

