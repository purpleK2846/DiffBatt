import numpy as np
import tensorflow as tf
from tensorflow import keras

class DiffusionModel(keras.Model):
    def __init__(self, network, ema_network, timesteps, gdf_util, ema=0.999, p_uncond=0.0, first_channels=8):
        super().__init__()
        self.network = network
        self.ema_network = ema_network
        self.timesteps = timesteps
        self.gdf_util = gdf_util
        self.ema = ema
        self.p_uncond = p_uncond
        self.first_channels = first_channels

    @tf.function    
    def bernoulli(self, shape):
        c = tf.random.uniform(shape, minval=0, maxval=1, dtype=tf.float32)
        c = tf.where(c < self.p_uncond, 0.0, 1.0)
        return c

    @tf.function
    def train_step(self, images):
        images, _, protocol = images
        # 1. Get the batch size
        batch_size = tf.shape(images)[0]

        # 2. Sample timesteps uniformly
        t = tf.random.uniform(
            minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64
        )

        c_mask = self.bernoulli(shape=(batch_size,))
        c_mask = tf.tile(c_mask[...,None], [1, self.first_channels*4])
        
        with tf.GradientTape() as tape:
            # 3. Sample random noise to be added to the images in the batch
            noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)

            # 4. Diffuse the images with noise
            images_t = self.gdf_util.q_sample(images, t, noise)

            # 5. Pass the diffused images and time steps to the network
            pred_noise = self.network([images_t, t, protocol, c_mask], training=True)

            # 6. Calculate the loss
            loss = self.loss(noise, pred_noise)

        # 7. Get the gradients
        gradients = tape.gradient(loss, self.network.trainable_weights)

        # 8. Update the weights of the network
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        # 9. Updates the weight values for the network with EMA weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        # 10. Return loss values
        return {"loss": loss}
    
    @tf.function
    def test_step(self, images):
        images, _, protocol = images
        # 1. Get the batch size
        batch_size = tf.shape(images)[0]

        # 2. Sample timesteps uniformly
        t = tf.random.uniform(
            minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64
        )

        c_mask = tf.ones(shape=(batch_size, self.first_channels*4))

        # 3. Sample random noise to be added to the images in the batch
        noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)

        # 4. Diffuse the images with noise
        images_t = self.gdf_util.q_sample(images, t, noise)

        # 5. Pass the diffused images and time steps to the network
        pred_noise = self.network([images_t, t, protocol, c_mask], training=False)

        # 6. Calculate the loss
        loss = self.loss(noise, pred_noise)

        # 10. Return loss values
        return {"loss": loss}
    
    @tf.function
    def generate(self, samples, tt, capacity_matrices, guide_w):
        ones = tf.ones((len(samples), self.first_channels*4))
        zeros = tf.zeros((len(samples), self.first_channels*4))

        pred_noise1 = self.ema_network([samples, tt, capacity_matrices, ones], training=False)
        pred_noise2 = self.ema_network([samples, tt, capacity_matrices, zeros], training=False)
        pred_noise = (1+guide_w)*pred_noise1 - guide_w*pred_noise2
        samples = self.gdf_util.p_sample(
            pred_noise, samples, tt, clip_denoised=False
        )
        return samples
    
    def generate_samples(self, capacity_matrices, guide_w = 0.0, record_samples=False):
        # 1. Randomly sample noise (starting point for reverse process)
        num_images = len(capacity_matrices)
        samples = tf.random.normal(
            shape=(num_images, 256, 1), dtype=tf.float32
        )
        capacity_matrices = tf.cast(capacity_matrices, dtype=tf.float32)

        record = []
        record.append(samples)
        # 2. Sample from the model iteratively
        for t in reversed(range(0, self.timesteps)):
            tt = tf.cast(tf.fill(num_images, t), dtype=tf.int64)
            samples = self.generate(samples, tt, capacity_matrices, guide_w)
            if record_samples:
                record.append(samples)
        # 3. Return generated samples
        if record_samples:
            return samples, record
        else:
            return samples
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'network': self.network,
            'ema_network': self.ema_network,
            'timesteps': self.timesteps,
            'gdf_util': self.gdf_util,
            'ema': self.ema,
            'p_uncond': self.p_uncond,
            'first_channels': self.first_channels,
        })
        return config