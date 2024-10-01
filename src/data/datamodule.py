'''
Author: 
Date: 2024-03-08 10:42:37
LastEditors: 
LastEditTime: 2024-06-04 10:23:19
Description: 
'''
from typing import Any, Dict, Optional, Tuple
from tqdm import tqdm


import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.utils.dataset_utils import get_downstream_task_names, get_metric, get_task_type
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

class DataModule(LightningDataModule):
    """`LightningDataModule` for the dataset.
    """

    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        train,
        val,
        test,
        collate_fn,
        lmdb_folder: str = 'lmdb',
        load_lmdb: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        task_idx:int = -1,  # [-1] all task → one model
        debug:bool = False,
    ) -> None:
        """Initialize a `DataModule`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        self.task_idx = task_idx
        self.task_type = get_task_type(dataset_name)
        task_names = get_downstream_task_names(dataset_name)
        self.task_names = task_names if self.task_idx==-1 else [task_names[self.task_idx]]

        self.data_path = ''
        self.csv_path = ''
        self.metric = get_metric(dataset_name)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            
            # load dataset
            self.data_train = self.hparams.train.dataset(task_type = self.task_type, task_names = self.task_names)
            self.data_val = self.hparams.val.dataset(task_type = self.task_type, task_names = self.task_names)
            self.data_test = self.hparams.test.dataset(task_type = self.task_type, task_names = self.task_names)
                    
            self.data_train.set_dataset('train')
            self.data_val.set_dataset('valid')
            self.data_test.set_dataset('test')
                
            
            log.info(f'Training Dataset: {len(self.data_train)}')
            log.info(f'Validation Dataset: {len(self.data_val)}')
            log.info(f'Testing Dataset: {len(self.data_test)}')
                        
            # stat mean, std and N
            if self.task_type=='regression':
                # if self.hparams.dataset_name=='qm9' or self.hparams.dataset_name=='docking':
                if self.hparams.dataset_name in ['moleculecla']:
                    # z, pos, pos_target, label, valid, batch_id 
                    labels = torch.cat([x[3] for x in tqdm(self.train_dataloader())],dim=0).view(-1,len(self.task_names))
                    
                    dataset_mean = labels.mean(axis=0)  # -6.5363
                    dataset_std = labels.std(axis=0)  # 0.5989
                    N = labels.size(0)
                    if len(self.task_names)==1:
                        log.info(f'{self.hparams.dataset_name} {self.task_names} ({N}) mean: {dataset_mean} std: {dataset_std}')
                    else:
                        for i,task_name in enumerate(self.task_names):
                           log.info(f'{self.hparams.dataset_name} {task_name} ({N}) mean: {dataset_mean[i]} std: {dataset_std[i]}') 
            self.mean = dataset_mean
            self.std = dataset_std
            self.dataset_n = N
            

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn = self.hparams.collate_fn,
            shuffle= not self.hparams.debug,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn = self.hparams.collate_fn,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn = self.hparams.collate_fn,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = DataModule()