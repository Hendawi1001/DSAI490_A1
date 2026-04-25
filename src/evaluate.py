import os
import glob
import json
import argparse
import tensorflow as tf
from data_processing import create_dataset
from model import VAE
from visualize import (
    plot_losses, visualize_reconstructions, 
    visualize_latent_space, generate_samples_vae, 
    visualize_denoising
)

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained AE and VAE models")
    parser.add_argument('--models_dir', type=str, default='models', help='Directory containing trained models')
    parser.add_argument('--extract_path', type=str, default='data/processed/medical-mnist', help='Data path')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    args = parser.parse_args()

    # Find all metadata files
    meta_files = glob.glob(os.path.join(args.models_dir, 'metadata_*_v1.json'))
    
    if not meta_files:
        print("No metadata files found. Please ensure training has completed.")
        return

    for meta_file in meta_files:
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
        
        region = metadata['region']
        latent_dim = metadata['latent_dim']
        print(f"\nEvaluating models for region: {region}")

        # Plot loss history
        plot_losses(metadata, region)

        # Load data
        region_path = os.path.join(args.extract_path, region)
        ds = create_dataset(region_path, batch_size=args.batch_size)

        # Load models
        ae_path = os.path.join(args.models_dir, f'ae_model_{region}_v1.keras')
        vae_enc_path = os.path.join(args.models_dir, f'vae_encoder_{region}_v1.keras')
        vae_dec_path = os.path.join(args.models_dir, f'vae_decoder_{region}_v1.keras')

        if os.path.exists(ae_path):
            try:
                ae_model = tf.keras.models.load_model(ae_path)
                ae_encoder = ae_model.get_layer('encoder')
                
                print("Generating AE visualizations...")
                visualize_reconstructions(ae_model, ds, f"AE Reconstructions - {region}", f"ae_reconstructions_{region}.png")
                visualize_latent_space(ae_encoder, ds, f"AE Latent Space (2D PCA) - {region}", f"ae_latent_{region}.png", is_vae=False)
                visualize_denoising(ae_model, ds, f"AE Denoising - {region}", f"ae_denoising_{region}.png", noise_factor=0.3)
            except Exception as e:
                print(f"Failed to evaluate AE for {region}: {e}")

        if os.path.exists(vae_enc_path) and os.path.exists(vae_dec_path):
            try:
                vae_encoder = tf.keras.models.load_model(vae_enc_path, compile=False)
                vae_decoder = tf.keras.models.load_model(vae_dec_path, compile=False)
                vae_model = VAE(vae_encoder, vae_decoder, beta=metadata['beta'])
                
                # Build the model by calling it on dummy data
                dummy_input = tf.zeros((1, 64, 64, 1))
                vae_model(dummy_input)

                print("Generating VAE visualizations...")
                visualize_reconstructions(vae_model, ds, f"VAE Reconstructions - {region}", f"vae_reconstructions_{region}.png")
                visualize_latent_space(vae_encoder, ds, f"VAE Latent Space (2D PCA) - {region}", f"vae_latent_{region}.png", is_vae=True)
                generate_samples_vae(vae_decoder, latent_dim, f"VAE Generated Samples - {region}", f"vae_samples_{region}.png")
                visualize_denoising(vae_model, ds, f"VAE Denoising - {region}", f"vae_denoising_{region}.png", noise_factor=0.3)
            except Exception as e:
                print(f"Failed to evaluate VAE for {region}: {e}")
                
        print(f"Visualizations saved to reports/figures/ for {region}")

if __name__ == "__main__":
    main()
