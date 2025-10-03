"""Training engine: optimizer/scheduler factories and simple train/eval loops.

Lightweight utilities used by exp/train.py. Designed to be device-agnostic and
work on CUDA, MPS and CPU. AMP is handled by the caller.
"""
# exp/engine.py
import torch, torch.nn as nn
from dataclasses import dataclass
from torch.optim.lr_scheduler import CosineAnnealingLR
from contextlib import nullcontext
@dataclass
class TrainState:
    epoch:int=0; step:int=0; best_val_acc:float=0.0
def make_optimizer(model, lr, weight_decay, opt_name='adamw'):
    params=[p for p in model.parameters() if p.requires_grad]
    opt_name=opt_name.lower()
    if opt_name=='sgd': return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    if opt_name=='adam': return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
def make_scheduler(optimizer, sched_name, epochs):
    if str(sched_name).lower()=='cosine': return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    return None
@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval(); loss_sum=acc_sum=n=0.0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images); loss = criterion(logits, labels)
        loss_sum += loss.item()*labels.size(0); acc_sum += (logits.argmax(1)==labels).float().sum().item(); n += labels.size(0)
    return loss_sum/n, acc_sum/n
def train_one_epoch(model, loader, optimizer, device, scaler, criterion, grad_clip=None):
    model.train(); running_loss=running_acc=n=0.0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        # Use autocast only for CUDA; for MPS/CPU run in eager mode
        if scaler is not None and device.type == 'cuda':
            autocast_ctx = torch.amp.autocast(device_type='cuda', enabled=True)
        else:
            autocast_ctx = nullcontext()
        with autocast_ctx:
            logits = model(images); loss = criterion(logits, labels)
        if scaler is not None and device.type == 'cuda':
            scaler.scale(loss).backward()
            if grad_clip: 
                scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            if grad_clip: nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        with torch.no_grad():
            running_loss += loss.item()*labels.size(0); running_acc += (logits.argmax(1)==labels).float().sum().item(); n += labels.size(0)
    return running_loss/n, running_acc/n
