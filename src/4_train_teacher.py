"""Module to train the End to End Teacher model."""

import argparse
import json
import os
import shutil
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from math import ceil
from utils import get_loaders, get_model


PATH_CONFIG = "data/POPCORN_PAPER/models/config/teacher/seed_"


def main():
    """Script to train the End to end model."""
    if os.path.exists(config["path_dir_checkpoint"] + config["model_name"] + ".ckpt"):
        shutil.rmtree(config["path_dir_checkpoint"] + config["model_name"] + ".ckpt")
    train_loader, val_loader = get_loaders(config)
    nb_train_steps = int(ceil(train_loader.__len__() / train_loader.batch_size))
    model = get_model(config, nb_train_steps=nb_train_steps * config["max_epochs"])
    wandb_logger = WandbLogger(project="POPCORN", name=config["model_name"])

    trainer = Trainer(
        deterministic=True,
        gradient_clip_val=config["gradient_norm"],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=config["max_epochs"],
        accelerator="gpu",
        devices=1,
        precision=16,
        callbacks=[
            ModelCheckpoint(
                dirpath=config["path_dir_checkpoint"],
                verbose=1,
                filename=config["model_name"],
                monitor=config["monitored"],
                save_last=True,
                mode=config["monitored_mode"],
                auto_insert_metric_name=False,
                save_top_k=1,
                save_weights_only=False,
                every_n_epochs=1,
                save_on_train_epoch_end=False,
            ),
        ],
    )
    print("Model Loaded")
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    NB_SEED = 5
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default=PATH_CONFIG,
        help="Path to the config file for training",
    )
    args = parser.parse_args()
    for seed_idx in range(1, NB_SEED + 1):
        path_config = args.config_path + str(seed_idx) + ".json"
        with open(path_config, "r", encoding="utf-8") as f:
            config = json.load(f)
        seed_everything(config["seed"])
        os.environ["CUDA_VISIBLE_DEVICES"] = config["gpus"]
        main()
