import re
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.nn.functional import mse_loss, l1_loss, smooth_l1_loss
from torch import nn
from typing import Any, Dict, Tuple

import torch

from lightning import LightningDataModule, LightningModule, Trainer
from torchmetrics import MaxMetric, MeanMetric, MinMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.regression import PearsonCorrCoef, SpearmanCorrCoef, KendallRankCorrCoef, MeanAbsoluteError, R2Score

import rootutils

rootutils.setup_root(__file__,indicator='.project-root',pythonpath=True)

from models.components.model import MLPModel
from src.utils.evaluate_utils import Evaluator
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

class NetModule(LightningModule):
    """NetModule.
    """

    def __init__(self, 
                 model:nn.Module,
                 optimizer:torch.optim.Optimizer,
                 lr_scheduler:torch.optim.lr_scheduler,
                 pos_weight:int,  # positive class, not position
                 task_type:str,
                 metric: str,
                 num_tasks:int,
                 task_names:list,
                 criterion = None,
                 pos_criterion = None,
                 y_weight:float = 1.0,
                 denoising_weight:float = 0.0,
                 encoder_ckpt_path: str = '',
                 hidden_channels:int = -1,
                 mean: torch.tensor = torch.tensor([0.0]),  # shape(1,num_tasks)
                 std: torch.tensor = torch.tensor([1.0]),  # shape(1,num_tasks)
                 freeze_encoder: bool = False,
                ):
        """Initialize a `NetModule`.
        """
        super(NetModule, self).__init__()

        self.save_hyperparameters(logger=False)
        self.task_type = task_type
        self.metric = metric
        self.num_tasks = num_tasks
        self.criterion = criterion
        self.pos_criterion = pos_criterion
        self.y_weight = y_weight
        self.denoising_weight = denoising_weight
        self.val_preds, self.val_labels, self.val_valids = [], [], []
        self.test_preds, self.test_labels, self.test_valids = [], [], []
        
        # model
        self.model = model(task_type = task_type, num_tasks = num_tasks, hidden_channels = hidden_channels, mean = mean, std = std)
        
        # load weights
        if encoder_ckpt_path is not None and encoder_ckpt_path !='':
            ckpt = torch.load(encoder_ckpt_path,map_location='cpu')
            state_dict = {re.sub(r"^model\.", "", k): v for k, v in ckpt["state_dict"].items()}
            encoder_state_dict = {re.sub(r"^representation_model\.", "", k): v for k, v in state_dict.items() if 'representation_model' in k}
            msg = self.model.encoder.load_state_dict(encoder_state_dict)
            log.info(f'Encoder {msg}')
        
        # freeze model
        if freeze_encoder:
            self.freeze_model(self.model.encoder)
            
        # evaluator
        self.evaluator = Evaluator(self.metric)  # , mean, std)
        self.train_acc = None
        self.val_acc = None
        self.test_acc = None

        if criterion is None:
            # loss function
            if self.metric in ['rmse', 'mae']:  # regression
                self.criterion = nn.L1Loss()
                
            elif self.metric == 'rocauc':  # classification
                self.criterion = nn.BCELoss(reduction='none')  # self.criterion = nn.BCELoss(weight = torch.tensor(pos_weight), reduction='none')
                
                # metric objects for calculating and averaging accuracy across batches
                self.train_acc = Accuracy(task="binary", num_classes=2)
                self.val_acc = Accuracy(task="binary", num_classes=2)
                self.test_acc = Accuracy(task="binary", num_classes=2)
        else:
            self.criterion = criterion

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
        self.train_pos_loss = MeanMetric()
        self.val_pos_loss = MeanMetric()
        self.test_pos_loss = MeanMetric()
        
        self.train_y_loss = MeanMetric()
        self.val_y_loss = MeanMetric()
        self.test_y_loss = MeanMetric()
        
        # metric objects
        self.train_metric_value = MeanMetric()
        
        self.task_names = task_names
        if len(task_names)>0:
            # self.corrcoef_metrics = ['Pearson', 'Spearman', 'KendallRank']
            self.corrcoef_metrics = ['Pearson', 'MAE', 'R2']
            for task_name in task_names:
                setattr(self,f'train_{task_name}_Pearson',PearsonCorrCoef())
                setattr(self,f'val_{task_name}_Pearson',PearsonCorrCoef())
                setattr(self,f'test_{task_name}_Pearson',PearsonCorrCoef())
                
                setattr(self,f'train_{task_name}_MAE',MeanAbsoluteError())
                setattr(self,f'val_{task_name}_MAE',MeanAbsoluteError())
                setattr(self,f'test_{task_name}_MAE',MeanAbsoluteError())
                
                setattr(self,f'train_{task_name}_R2',R2Score())
                setattr(self,f'val_{task_name}_R2',R2Score())
                setattr(self,f'test_{task_name}_R2',R2Score())
                
                # setattr(self,f'train_{task_name}_Spearman',SpearmanCorrCoef())
                # setattr(self,f'val_{task_name}_Spearman',SpearmanCorrCoef())
                # setattr(self,f'test_{task_name}_Spearman',SpearmanCorrCoef())
                # setattr(self,f'train_{task_name}_KendallRank',KendallRankCorrCoef())
                # setattr(self,f'val_{task_name}_KendallRank',KendallRankCorrCoef())
                # setattr(self,f'test_{task_name}_KendallRank',KendallRankCorrCoef())
        else:
            self.corrcoef_metrics = ['Pearson', 'MAE', 'R2']
            setattr(self,f'train_Pearson',PearsonCorrCoef())
            setattr(self,f'val_Pearson',PearsonCorrCoef())
            setattr(self,f'test_Pearson',PearsonCorrCoef())
            
            setattr(self,f'train_MAE',MeanAbsoluteError())
            setattr(self,f'val_MAE',MeanAbsoluteError())
            setattr(self,f'test_MAE',MeanAbsoluteError())
            
            setattr(self,f'train_R2',R2Score())
            setattr(self,f'val_R2',R2Score())
            setattr(self,f'test_R2',R2Score())

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, pos, batch_id, charges = None) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.model(x, pos, batch_id, charges = charges)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

        if self.val_acc is not None:
            self.val_acc.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        losses = {}
        denoising_loss = 0.0
        x, pos, pos_target, y, valid, batch_id, charges = batch
        logits, noise_pred = self.forward(x, pos, batch_id,charges)
        
        # Denoising loss
        if self.denoising_weight>0.0:
            noise_pred = noise_pred # + logits.sum() * 0
            if self.model.pos_normalizer is not None:
                normalized_pos_target = self.model.pos_normalizer(pos_target)
                denoising_loss = self.pos_criterion(noise_pred, normalized_pos_target)
            else:
                denoising_loss = self.pos_criterion(noise_pred, pos_target)
            
        # Y loss
        logits = logits.view(y.shape)
        y_loss = self.criterion(logits, y.float())
        if self.task_type == 'classification':
            preds = (logits>0.5).long()
            y_loss = y_loss.view(logits.shape)
        else:
            preds = logits
        losses['y'] = y_loss
        losses['pos'] = denoising_loss
        return losses, preds, y, valid

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        losses, preds, labels, valids = self.model_step(batch)
        y_loss = losses['y']
        if self.task_type =='classification':
            y_loss = (torch.sum(y_loss * valids) / torch.sum(valids)).to(y_loss.device)
        metric_value = self.evaluator(preds, labels, valids)
    
        loss = 0.0
        loss = y_loss * self.y_weight + losses['pos'] * self.denoising_weight
        
        lr=self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        self.log("train/lr", lr, on_step=True, on_epoch=True, prog_bar=True)
        
        # Y Loss
        self.train_y_loss(y_loss)
        self.log("train/y_loss", self.train_y_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Denoising Loss
        if self.denoising_weight>0.0:
            self.train_pos_loss(losses['pos'])
            self.log("train/pos_loss", self.train_pos_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Metric
        if metric_value is not None:
            self.train_metric_value(metric_value)
        self.log(f"train/{self.metric}", self.train_metric_value, on_step=False, on_epoch=True, prog_bar=True)

        # Loss
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # ACC for classification
        if self.task_type == 'classification':
            self.train_acc(preds[valids], labels[valids])
            self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        if len(self.task_names)>0:
            type = 'train'
            for i,task_name in enumerate(self.task_names):
                for corrcoef_metric in self.corrcoef_metrics:
                    coef_recorder = getattr(self,f'{type}_{task_name}_{corrcoef_metric}')
                    coef_recorder(preds.view(-1,len(self.task_names))[:,i],labels.view(-1,len(self.task_names))[:,i])
                    self.log(f"{type}/{task_name}_{corrcoef_metric}", coef_recorder, on_step=False, on_epoch=True, prog_bar=False)
        else:
            type = 'train'
            for corrcoef_metric in self.corrcoef_metrics:
                coef_recorder = getattr(self,f'{type}_{corrcoef_metric}')
                coef_recorder(preds.view(-1),labels.view(-1))
                self.log(f"{type}/{corrcoef_metric}", coef_recorder, on_step=False, on_epoch=True, prog_bar=False)
        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        losses, preds, labels, valids = self.model_step(batch)
        self.val_preds.append(preds)
        self.val_labels.append(labels)
        self.val_valids.append(valids)
        y_loss = losses['y']
        
        # update and log metrics 
        if self.task_type =='classification':
            y_loss = (torch.sum(y_loss * valids) / torch.sum(valids)).to(y_loss.device)
            
        loss = 0.0
        loss = y_loss * self.y_weight + losses['pos'] * self.denoising_weight
        
        # Y Loss
        self.val_y_loss(y_loss)
        self.log("val/y_loss", self.val_y_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Denoising Loss
        if self.denoising_weight>0.0:
            self.val_pos_loss(losses['pos'])
            self.log("val/pos_loss", self.val_pos_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Loss
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # ACC for classification
        if self.task_type == 'classification':
            self.val_acc(preds[valids], labels[valids])
            self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        
    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        preds = torch.cat(self.val_preds,dim=0)
        labels = torch.cat(self.val_labels,dim=0)
        valids = torch.cat(self.val_valids,dim=0)
        
        metric_value = self.evaluator(preds, labels, valids, print_log=True)
        self.log(f"val/{self.metric}", metric_value, on_step=False, on_epoch=True, prog_bar=True)
        
        if len(self.task_names)>0:
            type = 'val'
            for i,task_name in enumerate(self.task_names):
                for corrcoef_metric in self.corrcoef_metrics:
                    coef_recorder = getattr(self,f'{type}_{task_name}_{corrcoef_metric}')
                    coef_recorder(preds.view(-1,len(self.task_names))[:,i],labels.view(-1,len(self.task_names))[:,i])
                    self.log(f"{type}/{task_name}_{corrcoef_metric}", coef_recorder, on_step=False, on_epoch=True, prog_bar=False)
        else:
            type = 'val'
            for corrcoef_metric in self.corrcoef_metrics:
                coef_recorder = getattr(self,f'{type}_{corrcoef_metric}')
                coef_recorder(preds.view(-1),labels.view(-1))
                self.log(f"{type}/{corrcoef_metric}", coef_recorder, on_step=False, on_epoch=True, prog_bar=False)
                    
        self.val_preds = []
        self.val_labels = []
        self.val_valids = []

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        losses, preds, labels, valids = self.model_step(batch)
        self.test_preds.append(preds)
        self.test_labels.append(labels)
        self.test_valids.append(valids)
        y_loss = losses['y']
        
        # update and log metrics 
        if self.task_type =='classification':
            y_loss = (torch.sum(y_loss * valids) / torch.sum(valids)).to(y_loss.device)
            
        loss = 0.0
        loss = y_loss * self.y_weight + losses['pos'] * self.denoising_weight
        
        # Y Loss
        self.test_y_loss(y_loss)
        self.log("test/y_loss", self.test_y_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Denoising Loss
        if self.denoising_weight>0.0:
            self.test_pos_loss(losses['pos'])
            self.log("test/pos_loss", self.test_pos_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Loss
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # ACC for classification      
        if self.task_type == 'classification':
            self.test_acc(preds[valids], labels[valids])
            self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        
    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # if len(self.test_preds)>0:  # TODO
        preds = torch.cat(self.test_preds,dim=0)
        labels = torch.cat(self.test_labels,dim=0)
        valids = torch.cat(self.test_valids,dim=0)
        
        metric_value = self.evaluator(preds, labels, valids, print_log=True)
        self.log(f"test/{self.metric}", metric_value, on_step=False, on_epoch=True, prog_bar=True)
        
        if len(self.task_names)>0:
            type = 'test'
            for i,task_name in enumerate(self.task_names):
                for corrcoef_metric in self.corrcoef_metrics:
                    coef_recorder = getattr(self,f'{type}_{task_name}_{corrcoef_metric}')
                    coef_recorder(preds.view(-1,len(self.task_names))[:,i],labels.view(-1,len(self.task_names))[:,i])
                    self.log(f"{type}/{task_name}_{corrcoef_metric}", coef_recorder, on_step=False, on_epoch=True, prog_bar=False)
        else: 
            type = 'test'
            for corrcoef_metric in self.corrcoef_metrics:
                coef_recorder = getattr(self,f'{type}_{corrcoef_metric}')
                coef_recorder(preds.view(-1),labels.view(-1))
                self.log(f"{type}/{corrcoef_metric}", coef_recorder, on_step=False, on_epoch=True, prog_bar=False)
                    
        self.test_preds = []
        self.test_labels = []
        self.test_valids = []

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        
        train_steps_per_epoch=int(len(self.trainer.datamodule.train_dataloader())/self.trainer.num_devices)
        num_training_steps = train_steps_per_epoch*self.trainer.max_epochs
        num_warmup_steps=int(num_training_steps*self.hparams.lr_scheduler.warmup)
        
        if self.hparams.lr_scheduler.scheduler is not None:
            # scheduler = self.hparams.lr_scheduler.scheduler(optimizer=optimizer)
            if 'lr_end' in self.hparams.lr_scheduler.keys() and self.hparams.lr_scheduler.lr_end>0:
                scheduler = self.hparams.lr_scheduler.scheduler(optimizer=optimizer,num_training_steps=num_training_steps,num_warmup_steps=num_warmup_steps, lr_end = self.hparams.lr_scheduler.lr_end)
            else:
                scheduler = self.hparams.lr_scheduler.scheduler(optimizer=optimizer,num_training_steps=num_training_steps,num_warmup_steps=num_warmup_steps)
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": self.hparams.lr_scheduler.interval,
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

if __name__ == "__main__":
    _ = NetModule(None, None, None)