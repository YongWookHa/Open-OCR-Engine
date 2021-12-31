import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

from utils.utils import load_setting
from torch.utils.data import DataLoader, random_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TRAIN OCR-ENGINE MODULE')
    parser.add_argument('-m', '--module', type=str,
                        choices=['detector', 'recognizer'],
                        help='module to train')
    parser.add_argument('-v', '--version', type=int, default=0,
                        help='version number')
    parser.add_argument('-s', '--setting', type=str, default='settings/default.yaml',
                        help='model setting yaml file directory')
    parser.add_argument('-d', '--data_path', type=str, required=True,
                        help='generated data (.pkl) path')
    parser.add_argument('-bs', '--batch_size', type=int, default=4,
                        help='batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1.0,
                        help='learning rate for training [detector: 5e-5 for Adam, recognizer: 1.0 for Adadelta]')
    parser.add_argument('-e', '--max_epoch', type=int, default=100,
                        help='max epoch')
    parser.add_argument('-nw', '--num_workers', type=int, default=4,
                        help='number of workers for calling data')
    parser.add_argument('-rt', '--resume_training', type=str, default=None,
                        help='resume from certain checkpoint file')

    args = parser.parse_args()
    
    cfg = load_setting(args.setting)
    cfg.lr = args.learning_rate
    cfg.data_path = args.data_path
    
    print('configs:', cfg)

    if args.module == 'detector':
        from models.craft import CRAFT
        from datasets.craft_dataset import DatasetSYNTH, CustomCollate
        dataset = DatasetSYNTH(cfg)
        model = CRAFT(cfg)
        collate = CustomCollate(cfg.craft.image_size)
    else:
        from models.deepTextRecog import DeepTextRecog
        from datasets.deepTextRecog_dataset import DatasetSYNTH, AlignCollate
        dataset = DatasetSYNTH(cfg)
        model = DeepTextRecog(cfg, dataset.tokens)
        collate = AlignCollate(cfg)

    trainSize = int(len(dataset)*0.9)
    trainDataset, validDataset = random_split(dataset, [trainSize, len(dataset)-trainSize])
    trainDataloader = DataLoader(trainDataset,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 collate_fn=collate)
    validDataloader = DataLoader(validDataset,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 collate_fn=collate)

    logger = TensorBoardLogger('tb_logs', name=args.module,
                               version=args.version, default_hp_metric=False)
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')
    
    if args.module == 'detector':
        filename='checkpoints-{epoch:02d}-{fscore:.2f}'
    else:
        filename='checkpoints-{epoch:02d}-{acc:.2f}'
    
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor='fscore' if args.module == 'detector' else 'acc',
        dirpath=f'checkpoints/version_{args.version}',
        filename=filename,
        save_top_k=3,
        mode='max',
    )

    n_gpu = torch.cuda.device_count()

    trainer = pl.Trainer(gpus=n_gpu, max_epochs=args.max_epoch, logger=logger,
                         num_sanity_val_steps=1, strategy='ddp2' if n_gpu else None,
                         callbacks=[lr_callback, ckpt_callback],
                         plugins=DDPPlugin(find_unused_parameters=False),
                         # gradient_clip_val=5.0,
                         resume_from_checkpoint=args.resume_training)

    trainer.fit(model, train_dataloaders=trainDataloader, val_dataloaders=validDataloader)
