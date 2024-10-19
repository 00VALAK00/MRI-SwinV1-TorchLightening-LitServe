import os
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from data.dataset import MriDataset
from model.build import Swin4MRI
import torch


def create_trainer():
    early_stopping = EarlyStopping(monitor='eval_loss', patience=3, verbose=True, mode='min')
    checkpoint_callback = ModelCheckpoint(monitor='eval_loss', verbose=True, save_top_k=1, every_n_epochs=1)
    trainer = Trainer(
        min_epochs=9,
        max_epochs=20,
        devices="auto",
        accelerator="cuda",
        log_every_n_steps=20,
        check_val_every_n_epoch=2,
        default_root_dir=os.getcwd(),
        enable_checkpointing=True,
        enable_model_summary=True,
        precision=32,
        inference_mode=False,
        callbacks=[early_stopping, checkpoint_callback],
        logger=TensorBoardLogger(save_dir='./logs', name='MRI_model')

    )
    return trainer


def main():
    torch.set_float32_matmul_precision('high')
    swin_model = Swin4MRI(num_classes=4, use_cls_token=True, freeze_percentage=60)
    mri_data = MriDataset(batch_size=64)
    trainer = create_trainer()
    trainer.test(model=swin_model,
                 datamodule=mri_data,
                 ckpt_path="logs/MRI_model/version_1/checkpoints/epoch=9-step=770.ckpt"
                 )


if __name__ == '__main__':
    main()
