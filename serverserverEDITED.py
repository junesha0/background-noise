from typing import Dict, List, Optional
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pathlib
import uuid
import json
import pandas as pd
import os
import librosa
import librosa.display
import numpy as np
import IPython.display as ipd
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics   
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from tensorflow.keras.models import load_model
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
thisdir = pathlib.Path(
  __file__).parent.absolute()  # path to directory of this file

app.config['UPLOAD_FOLDER'] = './Uploads'


def get_prediction(audio, sample_rate):
    model = load_model('./model_weights.h5')
    
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)
    
    # create label encoder instance
    labelencoder = LabelEncoder()
    
    # load the labels
    labels = pd.read_csv('./archive/UrbanSound8K.csv')
    classes = np.array(labels['class'])
    
    # fit label encoder
    labelencoder.fit(classes)
    
    # predict the class
    predicted_label = model.predict(mfccs_scaled_features)
    
    classes_x = np.argmax(predicted_label, axis=1)
    
    # transform the predicted classes
    prediction_class = labelencoder.inverse_transform(classes_x)
    
    return prediction_class


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded'
        file = request.files['file']
        if file.filename == '':
            return 'No file selected'
        try:
            audio_data, sr = librosa.load(file, sr=None)
        except Exception as e:
            return f'Error processing file: {e}'
        result = get_prediction(audio_data, sr)   

        return render_template('result.html', RESULT=result)
    return render_template('upload.html')


if __name__ == '__main__':
  app.run(port=5000, debug=True)

