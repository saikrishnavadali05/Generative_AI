import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def generate_real_samples(n):
    """
    Generate real samples from a Gaussian distribution.
    
    Args:
    - n (int): Number of samples to generate.
    
    Returns:
    - Tuple[np.array, np.array]: Generated data and corresponding labels (ones).
    """
    data = np.random.randn(n)
    labels = np.ones((n, 1))
    return data, labels

def build_discriminator(input_shape=(1,)):
    """
    Build and compile the discriminator model.
    
    Args:
    - input_shape (tuple, optional): Shape of the input data. Default is (1,).
    
    Returns:
    - keras.models.Sequential: Compiled discriminator model.
    """
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(25, activation='relu', input_shape=input_shape))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_generator(latent_dim):
    """
    Build the generator model.
    
    Args:
    - latent_dim (int): Size of the latent space.
    
    Returns:
    - keras.models.Sequential: Generator model.
    """
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(15, activation='relu', input_dim=latent_dim))
    model.add(keras.layers.Dense(1, activation='linear'))
    return model

def build_gan(generator, discriminator):
    """
    Combine the generator and discriminator to build the GAN.
    
    Args:
    - generator (keras.models.Sequential): Generator model.
    - discriminator (keras.models.Sequential): Discriminator model.
    
    Returns:
    - keras.models.Sequential: Combined GAN model.
    """
    discriminator.trainable = False
    model = keras.models.Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def train_gan(generator, discriminator, gan, latent_dim, n_epochs=10000, n_batch=128):
    """
    Train the GAN.
    
    Args:
    - generator (keras.models.Sequential): Generator model.
    - discriminator (keras.models.Sequential): Discriminator model.
    - gan (keras.models.Sequential): Combined GAN model.
    - latent_dim (int): Size of the latent space.
    - n_epochs (int, optional): Number of training epochs. Default is 10,000.
    - n_batch (int, optional): Batch size for training. Default is 128.
    """
    for i in range(n_epochs):
        real_data, real_labels = generate_real_samples(n_batch//2)
        fake_data = generator.predict(np.random.randn(n_batch//2, latent_dim))
        fake_labels = np.zeros((n_batch//2, 1))
        
        d_loss_real = discriminator.train_on_batch(real_data, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_data, fake_labels)

        x_gan = np.random.randn(n_batch, latent_dim)
        y_gan = np.ones((n_batch, 1))
        g_loss = gan.train_on_batch(x_gan, y_gan)

        print(f'Epoch {i+1}/{n_epochs} | Discriminator Loss Real: {d_loss_real[0]} | Discriminator Loss Fake: {d_loss_fake[0]} | Generator Loss: {g_loss}')

def generate_fake_samples(generator, latent_dim, n):
    """
    Generate fake samples using the trained generator.
    
    Args:
    - generator (keras.models.Sequential): Generator model.
    - latent_dim (int): Size of the latent space.
    - n (int): Number of samples to generate.
    
    Returns:
    - np.array: Array of generated samples.
    """
    x_input = np.random.randn(latent_dim * n)
    x_input = x_input.reshape(n, latent_dim)
    generated_samples = generator.predict(x_input)
    return generated_samples

def plot_samples(real_samples, fake_samples):
    """
    Plot histograms of real and generated samples.
    
    Args:
    - real_samples (np.array): Array of real samples.
    - fake_samples (np.array): Array of generated samples.
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(real_samples, bins=20, color='blue', alpha=0.7, label='Real Samples')
    plt.title('Real Samples')
    plt.subplot(1, 2, 2)
    plt.hist(fake_samples, bins=20, color='red', alpha=0.7, label='Generated Samples')
    plt.title('Generated Samples')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    latent_dim = 5
    discriminator = build_discriminator()
    generator = build_generator(latent_dim)
    gan = build_gan(generator, discriminator)

    train_gan(generator, discriminator, gan, latent_dim)

    n_samples = 1000
    real_samples, _ = generate_real_samples(n_samples)
    generated_samples = generate_fake_samples(generator, latent_dim, n_samples)

    plot_samples(real_samples, generated_samples)
