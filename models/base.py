import torch
import pytorch_lightning as pl

from utils import base as utils

class ModelBase(pl.LightningModule):
    def __init__(self, cfg):
        super(ModelBase, self).__init__()
        self.cfg = cfg

        """ Define Layers """
        self.net = torch.nn.Identity()

    def forward(self, input):
        """
        input:
            [B, ]
        return:
            [B, ]
        """
        out = self.net(input)
        return out

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.cfg.optimizer)
        optimizer = optimizer(self.parameters(), lr=self.cfg.lr)

        if not hasattr(self.cfg, "scheduler") or  not self.cfg.scheduler:
            return optimizer
        elif hasattr(torch.optim.lr_scheduler, self.cfg.scheduler):
            scheduler = getattr(torch.optim.lr_scheduler, self.cfg.scheduler)
        elif hasattr(utils, self.cfg.scheduler):
            scheduler = getattr(utils, self.cfg.scheduler)
        else:
            raise ModuleNotFoundError

        scheduler = {
            'scheduler': scheduler(optimizer, **self.cfg.scheduler_param),
            'interval': self.cfg.scheduler_interval,
            'name': "learning rate"
            }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_nb):
        inp, labels = batch
        logits = self(inp)
        loss = self.cal_loss(logits, labels)
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        inp, labels = batch

        logits = self(inp)
        loss = self.cal_loss(logits, labels)

        pred = logits.argmax(1)

        acc = torch.all(labels==pred, dim=1).sum() / inp.size(0)
        self.log('val_acc', acc)
        self.log('val_loss', loss)
        return {'val_loss':loss, 'acc': acc}

    def validation_epoch_end(self, outputs):
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        print(f"\nEpoch {self.current_epoch} | avg_acc:{avg_acc}\n")

    def cal_loss(self, logits, targets):
        """
        Define how to calculate loss

        logits:
            [B, ]
        targets:
            [B, ]
        """
        loss = torch.nn.functional.cross_entropy(logits, targets)

        return loss
