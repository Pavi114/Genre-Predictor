
from utils import *

'''
CNN Network:
    Input Shape: 150 x 150 x 3
    13 Layer Network: 8 Convolution Layers with Relu activation and 5 Fully Connected Layers
    Output Layer: 11 output units
    Final Layer with Sigmoid Activation
'''

def CNN():

        model = models.Sequential()

        model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=(150, 150, 3)))
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(16, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(layers.Dropout(0.2))

        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(32, (3, 3), activation='relu',padding='same'))
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D(2, 2))

        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Dropout(0.4))

        model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(11, activation='sigmoid'))
        return model

'''
Input:
    1. file_path_data: Path of dataset
    2. file_path_image: Path where images are stored
    3. epochs: number of iterations
Function: Trains the model with the given dataset and analyses loss and accuracy
'''

def init(file_path_data, file_path_image, epochs):
    x, y, classes = get_dataset(file_path_data, file_path_image)
    y, genre = cvt_seq_to_mlb(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
    model = CNN()
    train_time = time.time()
    model.compile(optimizer=adam,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=32, validation_data=(x_valid, y_valid))
    print("------ %s sec for training" % (time.time() - train_time))

    test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
    print("test_acc :", test_acc)

    analysis(history)
'''
Plots showing training and vaidation loss and accuracy
'''

def analysis(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title("Training and validation accuracy")
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss,'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()


start_time = time.time()
init('Data/data.csv', 'Data/Posters/', epochs=5)
print("--- %s seconds ---" % (time.time() - start_time))
