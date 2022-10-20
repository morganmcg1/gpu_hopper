# Setting random seed to obtain reproducible results.
import tensorflow as tf

tf.random.set_seed(42)

import os
import glob
import imageio

import numpy as np
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras import layers
from functools import partial
import matplotlib.pyplot as plt

import wandb
from fastcore.script import *
from wandb.keras import WandbModelCheckpoint, WandbMetricsLogger

from nerf_utils import *

@call_parse
def main(
        wandb_run_id:Param("wandb run id from", str)=None,
        wandb_project:Param("wandb project to log to", str)=None,
        wandb_entity:Param("wandb entity to use", str)=None,
        ):

    # Resume the run
    wandb.init(
        id=wandb_run_id,
        resume="must",
        entity=wandb_entity,
        project=wandb_project
    )

    # Initialize global variables.
    AUTO = tf.data.AUTOTUNE
    BATCH_SIZE = wandb.config.BATCH_SIZE
    NUM_SAMPLES = wandb.config.NUM_SAMPLES
    EPOCHS = wandb.config.EPOCHS
    POS_ENCODE_DIMS = wandb.config.POS_ENCODE_DIMS
    SAVE_FREQ = wandb.config.SAVE_FREQ
    MODEL_NAME = wandb.config.MODEL_NAME


    # Download the data if it does not already exist.
    file_name = "tiny_nerf_data.npz"
    url = "https://people.eecs.berkeley.edu/~bmild/nerf/tiny_nerf_data.npz"
    if not os.path.exists(file_name):
        data = keras.utils.get_file(fname=file_name, origin=url)

    data = np.load(data)
    images = data["images"]
    im_shape = images.shape
    (num_images, H, W, _) = images.shape
    (poses, focal) = (data["poses"], data["focal"])

    # Create the training split.
    split_index = int(num_images * 0.8)

    # Split the images into training and validation.
    train_images = images[:split_index]
    val_images = images[split_index:]

    # Split the poses into training and validation.
    train_poses = poses[:split_index]
    val_poses = poses[split_index:]

    # Make the training pipeline.
    train_img_ds = tf.data.Dataset.from_tensor_slices(train_images)
    train_pose_ds = tf.data.Dataset.from_tensor_slices(train_poses)
    fn = partial(map_fn, H=H, W=2, focal=focal, NUM_SAMPLES=NUM_SAMPLES, POS_ENCODE_DIMS=POS_ENCODE_DIMS)
    train_ray_ds = train_pose_ds.map(fn,  num_parallel_calls=AUTO)
    training_ds = tf.data.Dataset.zip((train_img_ds, train_ray_ds))
    train_ds = (
        training_ds.shuffle(BATCH_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True, num_parallel_calls=AUTO)
        .prefetch(AUTO)
    )

    # Make the validation pipeline.
    val_img_ds = tf.data.Dataset.from_tensor_slices(val_images)
    val_pose_ds = tf.data.Dataset.from_tensor_slices(val_poses)
    val_ray_ds = val_pose_ds.map(fn, num_parallel_calls=AUTO)
    validation_ds = tf.data.Dataset.zip((val_img_ds, val_ray_ds))
    val_ds = (
        validation_ds.shuffle(BATCH_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True, num_parallel_calls=AUTO)
        .prefetch(AUTO)
    )

    test_imgs, test_rays = next(iter(train_ds))
    test_rays_flat, test_t_vals = test_rays

    loss_list = []

    num_pos = H * W * NUM_SAMPLES
    nerf_model = get_nerf_model(num_layers=8, num_pos=num_pos, POS_ENCODE_DIMS=POS_ENCODE_DIMS)

    model = NeRF(nerf_model, BATCH_SIZE=BATCH_SIZE, H=H, W=W, NUM_SAMPLES=NUM_SAMPLES)
    model.compile(
        optimizer=keras.optimizers.Adam(), loss_fn=keras.losses.MeanSquaredError()
    )

    # Create a directory to save the images during training.
    if not os.path.exists("images"):
        os.makedirs("images")

    model.fit(
        train_ds,
        validation_data=val_ds,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[TrainMonitor(), 
                   WandbMetricsLogger(log_freq="batch"),
                   WandbModelCheckpoint(filepath=MODEL_NAME, save_freq=SAVE_FREQ)],
        steps_per_epoch=split_index // BATCH_SIZE,
    )

    create_gif("images/*.png", "training.gif")

    # Get the trained NeRF model and infer.
    nerf_model = model.nerf_model
    test_recons_images, depth_maps = render_rgb_depth(
        model=nerf_model,
        rays_flat=test_rays_flat,
        t_vals=test_t_vals,
        rand=True,
        train=False,
    )

    # Create subplots.
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(10, 20))

    for ax, ori_img, recons_img, depth_map in zip(
        axes, test_imgs, test_recons_images, depth_maps
    ):
        ax[0].imshow(keras.preprocessing.image.array_to_img(ori_img))
        ax[0].set_title("Original")

        ax[1].imshow(keras.preprocessing.image.array_to_img(recons_img))
        ax[1].set_title("Reconstructed")

        ax[2].imshow(
            keras.preprocessing.image.array_to_img(depth_map[..., None]), cmap="inferno"
        )
        ax[2].set_title("Depth Map")

    rgb_frames = []
    batch_flat = []
    batch_t = []

    # Iterate over different theta value and generate scenes.
    for index, theta in tqdm(enumerate(np.linspace(0.0, 360.0, 120, endpoint=False))):
        # Get the camera to world matrix.
        c2w = pose_spherical(theta, -30.0, 4.0)

        #
        ray_oris, ray_dirs = get_rays(H, W, focal, c2w)
        rays_flat, t_vals = render_flat_rays(
            ray_oris, ray_dirs, near=2.0, far=6.0, num_samples=NUM_SAMPLES, rand=False
        )

        if index % BATCH_SIZE == 0 and index > 0:
            batched_flat = tf.stack(batch_flat, axis=0)
            batch_flat = [rays_flat]

            batched_t = tf.stack(batch_t, axis=0)
            batch_t = [t_vals]

            rgb, _ = render_rgb_depth(
                nerf_model, batched_flat, batched_t, rand=False, train=False
            )

            temp_rgb = [np.clip(255 * img, 0.0, 255.0).astype(np.uint8) for img in rgb]

            rgb_frames = rgb_frames + temp_rgb
        else:
            batch_flat.append(rays_flat)
            batch_t.append(t_vals)

    rgb_video = "rgb_video.mp4"
    imageio.mimwrite(rgb_video, rgb_frames, fps=30, quality=7, macro_block_size=None)

    wandb.finish()
    print('TRAINING FINISHED')