import tensorflow as tf

def build_autoencoder(latent_dim=64, input_shape=(64, 64, 1)):
    encoder_inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(16, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
    x = tf.keras.layers.Flatten()(x)
    latent_outputs = tf.keras.layers.Dense(latent_dim, activation="linear")(x)
    encoder = tf.keras.Model(encoder_inputs, latent_outputs, name="encoder")

    decoder_inputs = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(16 * 16 * 32, activation="relu")(decoder_inputs)
    x = tf.keras.layers.Reshape((16, 16, 32))(x)
    x = tf.keras.layers.Conv2DTranspose(16, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = tf.keras.layers.Conv2DTranspose(1, 3, activation="sigmoid", strides=2, padding="same")(x)
    decoder = tf.keras.Model(decoder_inputs, decoder_outputs, name="decoder")

    ae_outputs = decoder(encoder(encoder_inputs))
    autoencoder = tf.keras.Model(encoder_inputs, ae_outputs, name="autoencoder")
    return autoencoder, encoder, decoder

class VAE_SM(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_vae_components(latent_dim=64, input_shape=(64, 64, 1)):
    encoder_inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(16, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
    x = tf.keras.layers.Flatten()(x)

    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
    z = VAE_SM()([z_mean, z_log_var])
    encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="vae_encoder")
    
    decoder_inputs = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(16 * 16 * 32, activation="relu")(decoder_inputs)
    x = tf.keras.layers.Reshape((16, 16, 32))(x)
    x = tf.keras.layers.Conv2DTranspose(16, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = tf.keras.layers.Conv2DTranspose(1, 3, activation="sigmoid", strides=2, padding="same")(x)
    decoder = tf.keras.Model(decoder_inputs, decoder_outputs, name="vae_decoder")

    return encoder, decoder

class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, beta=1.0, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(data - reconstruction), axis=(1, 2))
            )

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = recon_loss + self.beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, inputs):
        z_mean, _, z = self.encoder(inputs)
        return self.decoder(z)

    def get_config(self):
        return {"beta": self.beta}
