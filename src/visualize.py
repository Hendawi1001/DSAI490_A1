import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
import os

def save_plot(filename):
    """Utility to save plot to reports/figures directory."""
    os.makedirs('reports/figures', exist_ok=True)
    plt.savefig(os.path.join('reports/figures', filename))
    plt.close()

def plot_losses(metadata, region_name):
    """Plots AE and VAE losses from metadata dictionary."""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    if "ae_loss_history" in metadata:
        plt.plot(metadata["ae_loss_history"], label='AE Loss', marker='o')
    plt.title(f'Autoencoder Loss - {region_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    if "vae_recon_loss_history" in metadata:
        plt.plot(metadata["vae_recon_loss_history"], label='VAE Recon Loss', marker='o', color='green')
    if "vae_kl_loss_history" in metadata:
        plt.plot(metadata["vae_kl_loss_history"], label='VAE KL Loss', marker='x', color='red')
    plt.title(f'VAE Losses - {region_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    save_plot(f'losses_{region_name}.png')

def visualize_reconstructions(model, dataset, title, filename, num_images=6):
    images, _ = next(iter(dataset.take(1)))
    images = images[:num_images]
    reconstructed = model.predict(images, verbose=0)

    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    fig.suptitle(title, fontsize=16)
    for i in range(num_images):
        axes[0, i].imshow(images[i].numpy().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title("Original")

        axes[1, i].imshow(reconstructed[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title("Reconstructed")
    
    save_plot(filename)

def visualize_latent_space(encoder, dataset, title, filename, is_vae=False):
    latent_vectors = []
    for batch_x, _ in dataset.take(50):
        if is_vae:
            z_mean, _, _ = encoder.predict(batch_x, verbose=0)
            latent_vectors.append(z_mean)
        else:
            z = encoder.predict(batch_x, verbose=0)
            latent_vectors.append(z)

    latent_vectors = np.concatenate(latent_vectors, axis=0)
    latent_2d = PCA(n_components=2).fit_transform(latent_vectors)

    plt.figure(figsize=(6, 5))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.5, s=15, c='blue')
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    
    save_plot(filename)

def generate_samples_vae(decoder, latent_dim, title, filename, num_samples=6):
    random_noise = tf.random.normal(shape=(num_samples, latent_dim))
    generated = decoder.predict(random_noise, verbose=0)

    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2, 3))
    fig.suptitle(title, fontsize=16)
    for i in range(num_samples):
        axes[i].imshow(generated[i].squeeze(), cmap='gray')
        axes[i].axis('off')
    
    save_plot(filename)

def visualize_denoising(model, dataset, title, filename, noise_factor=0.3, num_images=6):
    images, _ = next(iter(dataset.take(1)))
    images = images[:num_images]

    noise = tf.random.normal(shape=tf.shape(images), mean=0.0, stddev=noise_factor)
    noisy_images = tf.clip_by_value(images + noise, clip_value_min=0., clip_value_max=1.)
    reconstructed = model.predict(noisy_images, verbose=0)

    fig, axes = plt.subplots(3, num_images, figsize=(num_images * 2, 6))
    fig.suptitle(title, fontsize=16)
    for i in range(num_images):
        axes[0, i].imshow(images[i].numpy().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title("Original")

        axes[1, i].imshow(noisy_images[i].numpy().squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title(f"Noisy (+{noise_factor})")

        axes[2, i].imshow(reconstructed[i].squeeze(), cmap='gray')
        axes[2, i].axis('off')
        if i == 0: axes[2, i].set_title("Denoised")
    plt.tight_layout()
    save_plot(filename)
