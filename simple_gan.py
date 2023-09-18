import numpy as np
import tensorflow as tf
from tensorflow import keras

# Generate real samples
def generate_real_samples(n):
    """Generate n 1D Gaussian numbers."""
    data = np.random.randn(n)
    labels = np.ones((n, 1))
    return data, labels

# Discriminator Model
def build_discriminator(input_shape=(1,)):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(25, activation='relu', input_shape=input_shape))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Generator Model
def build_generator(latent_dim):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(15, activation='relu', input_dim=latent_dim))
    model.add(keras.layers.Dense(1, activation='linear'))
    return model

# Combined GAN Model
def build_gan(generator, discriminator):
    discriminator.trainable = False  # Freeze weights of discriminator during GAN training
    model = keras.models.Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

# Training the GAN
def train_gan(generator, discriminator, gan, latent_dim, n_epochs=10000, n_batch=128):
    for i in range(n_epochs):
        # Train the discriminator
        real_data, real_labels = generate_real_samples(n_batch//2)
        fake_data = generator.predict(np.random.randn(n_batch//2, latent_dim))
        fake_labels = np.zeros((n_batch//2, 1))
        
        d_loss_real = discriminator.train_on_batch(real_data, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_data, fake_labels)

        # Train the generator
        x_gan = np.random.randn(n_batch, latent_dim)
        y_gan = np.ones((n_batch, 1))
        g_loss = gan.train_on_batch(x_gan, y_gan)

        print(f'Epoch {i+1}/{n_epochs} | Discriminator Loss Real: {d_loss_real[0]} | Discriminator Loss Fake: {d_loss_fake[0]} | Generator Loss: {g_loss}')

    return generator, discriminator

if __name__ == "__main__":
    # Parameters
    latent_dim = 5

    # Create the models
    discriminator = build_discriminator()
    generator = build_generator(latent_dim)
    gan = build_gan(generator, discriminator)

    # Train the GAN
    generator, discriminator = train_gan(generator, discriminator, gan, latent_dim)
