import os
import numpy as np
import requests
import pandas as pd
import cv2 as cv

'''
Function to download posters
'''

def download_posters(file_path='Data/data.csv', file_dest='Data/Posters/'):
    movie_data = pd.read_csv(file_path,usecols=['imdbId', 'Genre', 'Poster'])
    for index, row in movie_data.iterrows():
        path = str(file_dest) + str(row['imdbId']) + '.jpg'
        url = str(row['Poster'])
        r = requests.get(url)
        with open(path, 'wb') as f:
            f.write(r.content)

'''
Prints summary of dataset: Total genres and movies per genre
'''

def summmary(file_path='Data/data.csv'):
    movie_data = pd.read_csv(file_path, usecols=['imdbId', 'Genre', 'Poster'])
    length = len(movie_data)
    genrelist = []

    for row in movie_data.iterrows():
        genres = str(row["Genre"])
        genres = genres.split("|")
        genrelist.extend(genres)

    unique_genres = list(set(genrelist))

    print("Total Movies: " + str(length))
    print("Total Number of Genres is: " + str(len(unique_genres)))
    print("Movies Per Genre: ")
    movies_per_genre = []
    for genre in unique_genres:
        ct = genrelist.count(genre)
        movies_per_genre.append(ct)
        print(genre + " : " + str(ct))

    y_pos = np.arange(len(unique_genres))
    plt.bar(y_pos,movies_per_genre,align='center',alpha=0.5)
    plt.xticks(y_pos,unique_genres,rotation='vertical')
    plt.ylabel("Number of Movies")
    plt.title("Number of Movies by Genre")
    plt.show()

'''
Returns genres of a particular ImdbId as a tuple
'''

def get_genre(data, image_id):
    return tuple((data[data['imdbId'] == int(image_id)]['Genre'].values[0]).split('|'))

'''
Preprocessess the image: Resizes it to 150 x 150
'''
def preprocess(path, size=150):
    image = cv.imread(path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, (size, size))
    return image

'''
Get inputs/outputs for the model to train: Images and genres
'''

def get_dataset(file_path_data, file_path_image):
    image_paths = glob.glob(file_path_image + '*.jpg')
    data = pd.read_csv(file_path_data)
    image_ids = []
    x = []
    y = []
    classes = tuple()
    for path in image_paths:
        start = path.rfind('/') + 1
        end = len(path) - 4
        id = path[start:end]
        x.append(preprocess(path))
        genre = get_genre(data, id)
        y.append(np.asarray(genre))
        classes = classes + genre
    classes = set(classes)
    return np.asarray(x), np.asarray(y), classes

def cvt_seq_to_mlb(y):
    mlb = MultiLabelBinarizer()
    mlb.fit(y)
    print(mlb.classes_)
    return mlb.transform(y), mlb.classes_
