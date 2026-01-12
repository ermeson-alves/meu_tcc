import gc
import time
import torch
import numpy as np
from typing import List, Callable
from pathlib import Path
import albumentations as A
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import KFold
from src.data import LOCCADataset
from src.constants import NUM_FOLDS, BATCH_SIZE, NUM_EPOCHS, NUM_REPT


def volumes_to_pngs_paths(volumes: List[Path], png_dir: Path) -> List[Path]:
    """
    Dado uma lista de volumes do conjunto LOCCA, é retornada uma lista
    de todos os arquivos pngs relacionados (todas as slices correspondentes
    convertidas.)
    """
    png_paths = []

    for volume in volumes:
        base_path = volume.stem.split('.')[0]
        png_paths = png_paths + list(png_dir.glob(f'images/{base_path}*.png'))

    return png_paths


def cleanup_memory(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


class KFoldExecutor:
    def __init__(
            self,
            volumes_paths: np.array,
            png_dir: Path,
            model_factory: Callable[[int, int], pl.LightningModule],
            trainer_callbacks: List,
            trainer_logger,
            train_transform: A.Compose,
            test_transform: A.Compose,
    ):
        self.volumes_paths = volumes_paths
        self.png_dir = png_dir
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.model_factory = model_factory
        self.trainer_callbacks = trainer_callbacks
        self.trainer_logger = trainer_logger
        self.lightning_trainer = None

    
    def base_loop(self):
        # Shufle aqui apenas embaralha as folds e não as amostras em cada fold
        kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=96)

        # Ao realizar o split sobre os volumes ao invés de slices, garanto que a fold de teste
        # não contém slices de algum volume de treino.
        for repetition in range(1, NUM_REPT+1):
            for fold, (train_ids, test_ids) in enumerate(kfold.split(self.volumes_paths)):
                # Print
                print('_'*80)
                print(f'FOLD {fold}')

                volume_paths_train = self.volumes_paths[train_ids]
                volume_paths_test = self.volumes_paths[test_ids]
            
                pngs_paths_train = volumes_to_pngs_paths(volume_paths_train, self.png_dir)
                pngs_paths_test = volumes_to_pngs_paths(volume_paths_test, self.png_dir)

                
                # Definição dos conjuntos de treino e teste.
                train_dataset = LOCCADataset(
                    pngs_paths_train,
                    self.train_transform
                )

                test_dataset = LOCCADataset(
                    pngs_paths_test,
                    self.test_transform
                ) 
                
                # Definição do carregador do conjunto de Treino e Teste com base nas folds.
                # aqui o conjunto de treino pode ser embaralhado.
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    num_workers=4,
                )

                test_loader = DataLoader(
                    test_dataset,
                    batch_size=BATCH_SIZE,
                    num_workers=4
                )
                
                # Treinamento do modelo utilizando Lighting Module
                # aqui irei armazenar todas as métricas resultantes com a informação
                # sobre qual iteração do Kfold e qual repetição para análise posteriormente.
                model = self.model_factory(fold=fold, repetition=repetition)
                self._train(model, train_loader)


                # Teste do modelo
                self._test(model, test_loader)

                # Limpeza
                self.lightning_trainer = None
                cleanup_memory(model, train_loader, test_loader)

    

    def _train(self, model, train_dataloader):
        torch.set_float32_matmul_precision('high')

        self.lightning_trainer = pl.Trainer(
            max_epochs=NUM_EPOCHS,
            limit_train_batches=10,
            enable_checkpointing=False,
            limit_test_batches=5,
            callbacks=self.trainer_callbacks,
            logger=self.trainer_logger
        )

        self.lightning_trainer.fit(
            model,
            train_dataloaders=train_dataloader,
        )

    
    def _test(self, model, test_dataloader):
        self.lightning_trainer.test(
            model,
            test_dataloader,
        )