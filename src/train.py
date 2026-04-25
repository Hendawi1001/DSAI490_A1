import os
import argparse
import json
import tensorflow as tf
from data_processing import extract_data, get_regions, create_dataset
from model import build_autoencoder, build_vae_components, VAE

def main():
    parser = argparse.ArgumentParser(description="Train AE and VAE models on medical-mnist")
    parser.add_argument('--epochs', type=int, default=7, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--latent_dim', type=int, default=64, help='Latent dimension size')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta value for VAE')
    parser.add_argument('--zip_path', type=str, default='data/raw/medical-mnist.zip', help='Path to dataset zip file')
    parser.add_argument('--extract_path', type=str, default='data/processed/medical-mnist', help='Extraction path')
    parser.add_argument('--models_dir', type=str, default='models', help='Directory to save trained models')

    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)

    print("Step 1: Preparing Data...")
    extract_data(args.zip_path, args.extract_path)

    regions = get_regions(args.extract_path)
    if not regions:
        print(f"No data regions found in {args.extract_path}. Ensure the dataset is present.")
        return

    print(f"Found regions: {regions}")

    datasets = {}
    for region in regions:
        region_path = os.path.join(args.extract_path, region)
        datasets[region] = create_dataset(region_path, batch_size=args.batch_size)

    for region, ds in datasets.items():
        print(f"\n{'='*60}")
        print(f"Training for region: {region}")

        print("\n--- Training Autoencoder ---")
        ae_model, ae_encoder, ae_decoder = build_autoencoder(latent_dim=args.latent_dim)
        ae_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')
        
        ae_history = ae_model.fit(ds, epochs=args.epochs)
        
        ae_model_path = os.path.join(args.models_dir, f'ae_model_{region}_v1.keras')
        ae_model.save(ae_model_path)
        print(f"Saved AE model to {ae_model_path}")

        print("\n--- Training Variational Autoencoder ---")
        vae_encoder, vae_decoder = build_vae_components(latent_dim=args.latent_dim)
        vae_model = VAE(vae_encoder, vae_decoder, beta=args.beta)
        vae_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
        
        vae_history = vae_model.fit(ds, epochs=args.epochs)
        
        vae_encoder_path = os.path.join(args.models_dir, f'vae_encoder_{region}_v1.keras')
        vae_decoder_path = os.path.join(args.models_dir, f'vae_decoder_{region}_v1.keras')
        vae_encoder.save(vae_encoder_path)
        vae_decoder.save(vae_decoder_path)
        print(f"Saved VAE components to {args.models_dir}")

        metadata = {
            "region": region,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "latent_dim": args.latent_dim,
            "beta": args.beta,
            "ae_final_loss": float(ae_history.history['loss'][-1]),
            "vae_final_loss": float(vae_history.history['loss'][-1]),
            "ae_loss_history": [float(x) for x in ae_history.history['loss']],
            "vae_recon_loss_history": [float(x) for x in vae_history.history.get('recon_loss', [])],
            "vae_kl_loss_history": [float(x) for x in vae_history.history.get('kl_loss', [])],
            "vae_total_loss_history": [float(x) for x in vae_history.history.get('loss', [])]
        }
        meta_path = os.path.join(args.models_dir, f'metadata_{region}_v1.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Saved metadata to {meta_path}")

if __name__ == "__main__":
    main()
