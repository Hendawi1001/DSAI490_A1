import unittest
import os
import tensorflow as tf
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from model import build_autoencoder, build_vae_components, VAE

class TestModel(unittest.TestCase):
    def setUp(self):
        self.latent_dim = 64
        self.input_shape = (64, 64, 1)
        self.batch_size = 2
        self.dummy_input = tf.random.normal((self.batch_size, *self.input_shape))

    def test_autoencoder_shapes(self):
        ae_model, encoder, decoder = build_autoencoder(latent_dim=self.latent_dim, input_shape=self.input_shape)
        
        latent_output = encoder(self.dummy_input)
        self.assertEqual(latent_output.shape, (self.batch_size, self.latent_dim))
        
        reconstruction = decoder(latent_output)
        self.assertEqual(reconstruction.shape, (self.batch_size, *self.input_shape))
        
        ae_output = ae_model(self.dummy_input)
        self.assertEqual(ae_output.shape, (self.batch_size, *self.input_shape))

    def test_vae_shapes(self):
        encoder, decoder = build_vae_components(latent_dim=self.latent_dim, input_shape=self.input_shape)
        vae_model = VAE(encoder, decoder)
        
        z_mean, z_log_var, z = encoder(self.dummy_input)
        self.assertEqual(z_mean.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(z_log_var.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(z.shape, (self.batch_size, self.latent_dim))
        
        reconstruction = decoder(z)
        self.assertEqual(reconstruction.shape, (self.batch_size, *self.input_shape))
        
        vae_output = vae_model(self.dummy_input)
        self.assertEqual(vae_output.shape, (self.batch_size, *self.input_shape))

if __name__ == '__main__':
    unittest.main()
