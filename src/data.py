from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import torch
import cv2
from .utils.load_image_and_annotation import load_nifti, load_nrrd


class LOCCADataset(Dataset):
    """
    @ARTICLE{11029450,
        author={Ribeiro, Jean A. and Carmo, Diedre S. Do and Reis, Fabiano and Magalhães, Ricardo S. and Dertkigil, Sergio S. J. and Appenzeller, Simone and Rittner, Leticia},
        journal={IEEE Data Descriptions}, 
        title={Descriptor: Manually Annotated CT Dataset of Lung Lobes in COVID-19 and Cancer Patients (LOCCA)}, 
        year={2025},
        volume={2},
        number={},
        pages={239-246},
        keywords={Lungs;Computed tomography;Annotations;Lung cancer;Biomedical imaging;Lesions;Image segmentation;Manuals;COVID-19;Three-dimensional displays;Cancer;computed tomography (CT) images;COVID-19;dataset;manual annotation for lung lobes},
        doi={10.1109/IEEEDATA.2025.3577999}}


    Para utilizar esta classe, realizei antes um pré-processamento nos dados que, 
    dentre outras coisas, converte cada slice para png.
    """
    
    def __init__(self, base_dir: Path, transform=None):
        self.base_dir = base_dir
        self.transform = transform
        self.pngs_paths = sorted(base_dir.glob('images/*.png'))

    def __len__(self):
        return len(self.pngs_paths)

    def __getitem__(self, idx):
        base_path = self.pngs_paths[idx]
        volume_file = str(base_path)
        mask_file = self.base_dir / f'masks/{base_path.name}'
        # rl_path = self.dataset_root / f'pngs_preprocessed/masks/rl_{base_path.name}'
        # ll_path = self.dataset_root / f'pngs_preprocessed/masks/ll_{base_path.name}'

        volume = cv2.imread(volume_file, 0)  # leitura em tons de cinza
        mask = cv2.imread(mask_file, 0)
        # right_lung = cv2.imread(rl_path, 0)  # leitura em tons de cinza
        # left_lung = cv2.imread(ll_path, 0)  # leitura em tons de cinza

        assert volume.shape == mask.shape


        new_mask = np.zeros_like(mask, dtype=np.uint8)
        new_mask[np.isin(mask, [1, 2])] = 1  # pulmão direito
        new_mask[np.isin(mask, [3, 4, 5])] = 2  # pulmão esquerdo

        # DataAugmentation (transformações geométricas, normalização...)
        if self.transform:
            augmented = self.transform(image=volume, mask=new_mask)
            volume = augmented["image"]
            new_mask = augmented["mask"]

        # Independente de transformações, o formato esperado pelo Module é
        # objetos torch com float16/float32 para o volume e int8/(...) para a mascara..
        volume = volume.astype(np.float32)
        new_mask = new_mask.astype(np.uint8)

        volume = torch.from_numpy(volume)
        new_mask = torch.from_numpy(new_mask)


        return {
            'volume': volume,
            'mask': new_mask,
            'path_nifti': str(base_path),
        }
