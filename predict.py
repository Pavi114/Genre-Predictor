from tensorflow.keras.models import model_from_json
import numpy as np
from utils import preprocess

'''
Input: Path of image file (eg: 'Sample/conjuring.jpg')
Output: Genres predicted with the poster
Function: Loads the trained model and predicts 3 most probable genre of the poster
'''

def predict_genre(file_path):

    x = []
    x.append(preprocess(file_path))
    x = np.asarray(x)
    x = x.astype('float32')

    PATH_MODEL = 'Model/model.json'
    PATH_WEIGHTS = 'Model/model.h5'
    classes = {
        0: 'Action',
        1: 'Adventure',
        2: 'Animation',
        3: 'Comedy',
        4: 'Documentary',
        5: 'Drama',
        6: 'Horror',
        7: 'Music',
        8: 'Romance',
        9: 'Sci-Fi',
        10: 'Thriller'
    }

    model_json_file = open(PATH_MODEL, 'r')
    load_model = model_json_file.read()
    model_json_file.close()
    model = model_from_json(load_model)
    model.load_weights(PATH_WEIGHTS)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    y = model.predict(x)
    indices = np.argpartition(y[0], -2)[-2:]
    genres = []
    for index in indices:
        genres.append(classes[index])
    return genres
