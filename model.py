from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, Sequential

input_img = Input(shape=(28, 28, 1))
model = Sequential(
    [   
        Conv2D(16, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),
        Conv2D(8, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),
        Conv2D(8, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),
        Conv2D(8, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(8, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(16, (3, 3), activation='relu'),
        UpSampling2D((2, 2)),
        Conv2D(1, (3, 3), activation='tanh', padding='same')
    ]
)(input_img)

autoencoder = Model(input_img, model)