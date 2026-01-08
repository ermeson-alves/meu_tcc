import torch
import torchmetrics
from skimage.metrics import hausdorff_distance


class HausdorffDistance(torchmetrics.Metric):
    def __init__(self, dist_threshold: float = 10.0, compute_on_step: bool = True, dist_type: str = "max"):
        super().__init__(compute_on_step=compute_on_step)
        self.dist_threshold = dist_threshold
        self.dist_type = dist_type
        self.add_state("distances", torch.zeros(0), dist_init=torch.zeros)
        
    def update(self, preds, target):
        preds = preds.cpu().numpy()
        target = target.cpu().numpy()

        distances = [hausdorff_distance(pred, gt) for pred, gt in zip(preds, target)]
        
        self.distances = torch.cat([self.distances, torch.tensor(distances)])
        
    def compute(self):
        return self.distances.mean()