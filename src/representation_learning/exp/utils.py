# exp/utils.py
import random, numpy as np, torch
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=True
class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, classes, smoothing=0.0):
        super().__init__(); self.confidence=1.0-smoothing; self.smoothing=smoothing; self.cls=classes
    def forward(self, pred, target):
        logprobs = torch.nn.functional.log_softmax(pred, dim=-1)
        with torch.no_grad():
            true = torch.zeros_like(logprobs); true.fill_(self.smoothing/(self.cls-1)); true.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true*logprobs, dim=-1))
