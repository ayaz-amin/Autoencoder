import numpy as np
from keras.datasets import mnist
from model import autoencoder

(X, _), (_, _) = mnist.load_data()

X = X.astype('float32') / 255.
X = np.reshape(X, (len(X), 28, 28, 1))

noise = []

def generate_noise():
    for i in range(len(X)):
        Z = np.random.normal(0, 1, size=(28, 28, 1))
        noise.append(Z)

generate_noise()

noise = np.array(noise)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(noise, X, epochs=3, batch_size=128, shuffle=True)
autoencoder.save('AE.h5')