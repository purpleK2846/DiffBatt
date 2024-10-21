import tensorflow as tf
from tensorflow import keras
from tensorflow.data import Dataset as tfds
from gaussian_diffusion import GaussianDiffusion
from diffusion_model import DiffusionModel
from postproc_utils import PostProcess

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

tf.config.set_visible_devices(gpus, 'GPU')

battery_dataset = 'mix'
total_timesteps = 1000
p_uncond = 0.2

mdir = './trained_models/'
network = keras.models.load_model(mdir+f'{battery_dataset}_network')
ema_network = keras.models.load_model(mdir+f'{battery_dataset}_ema_network')

# Get an instance of the Gaussian Diffusion utilities
gdf_util = GaussianDiffusion(timesteps=total_timesteps)

# Get the model
model = DiffusionModel(
    network=network,
    ema_network=ema_network,
    gdf_util=gdf_util,
    timesteps=total_timesteps,
    p_uncond=p_uncond,
)

train_ds = tfds.load(f'./data/{battery_dataset}_train_ds')
test_ds = tfds.load(f'./data/{battery_dataset}_test_ds')

post_process = PostProcess(test_ds, model)
refs, preds = post_process.pred(reps=1)
soh_rmse = post_process.eval_soh(refs, preds)
rul_rmse = post_process.eval_rul(refs, preds)

post_process.plot_sample(refs, preds)
print('RUL RMSE', rul_rmse, 'SOH RMSE', soh_rmse)
