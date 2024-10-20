import os
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from data.dataset import MriDataset
from model.build import Swin4MRI
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default=None, help="load model checkpoint")
    parser.add_argument('--max_epochs', type=int, default=10, help='maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--freeze_percentage', type=float, default=80, help='How many overall parameters to freeze')
    parser.add_argument("--train", type=bool, help="enable training",default=True)
    parser.add_argument('--test', type=bool, default=True, help='Enable testing after training')

    return parser.parse_args()


def create_trainer(args: argparse.Namespace):
    early_stopping = EarlyStopping(monitor='eval_loss', patience=3, verbose=True, mode='min')
    checkpoint_callback = ModelCheckpoint(monitor='eval_loss', verbose=True, save_top_k=1, every_n_epochs=1)
    trainer = Trainer(
        min_epochs=9,
        max_epochs=args.max_epochs,
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
    args = parse_args()
    torch.set_float32_matmul_precision('high')
    swin_model = Swin4MRI(num_classes=4, use_cls_token=True, freeze_percentage=args.freeze_percentage)
    mri_data = MriDataset(batch_size=args.batch_size)
    trainer = create_trainer(args)
    if args.train:
        trainer.fit(model=swin_model,
                    datamodule=mri_data,
                    ckpt_path=args.checkpoint_dir,
                    )

    if args.test:
        trainer.test(model=swin_model,
                     datamodule=mri_data,
                     ckpt_path=args.checkpoint_dir)




if __name__ == '__main__':
    main()
