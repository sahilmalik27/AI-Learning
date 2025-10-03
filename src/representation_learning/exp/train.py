"""Experiment runner for representation learning on CIFAR-10.

Features:
- Model factory (LeNet/VGG/ResNet/MobileNet)
- Device pick (CUDA→MPS→CPU), AMP on CUDA
- Config-driven training with optional small subsets
- Configurable logging (TB, confusion, feature maps, Grad-CAM, CSV)
- Checkpointing to models/representations/exp/<exp_name>
"""
import os, argparse, yaml, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision

from .engine import TrainState, make_optimizer, make_scheduler, train_one_epoch, evaluate
from .data import make_cifar10_loaders
from .utils import set_seed, LabelSmoothingLoss
from ..cnn_suite import make_model, count_params

def compute_confusion_and_per_class(model, loader, device, num_classes=10):
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                cm[int(t.item()), int(p.item())] += 1
    per_class_acc = {}
    for c in range(num_classes):
        total = cm[c].sum()
        correct = cm[c, c]
        per_class_acc[c] = (correct / total) if total > 0 else 0.0
    return cm, per_class_acc

def plot_confusion_matrix(cm, class_names=None):
    import itertools
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix')
    tick_marks = np.arange(cm.shape[0])
    ax.set_xticks(tick_marks); ax.set_yticks(tick_marks)
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    fmt = 'd'
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8)
    ax.set_ylabel('True label'); ax.set_xlabel('Predicted label')
    fig.tight_layout()
    return fig

def _find_last_conv(model: nn.Module):
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last

class _GradCAMHook:
    def __init__(self, layer: nn.Module):
        self.a = None
        self.g = None
        self.h1 = layer.register_forward_hook(self._save_a)
        self.h2 = layer.register_full_backward_hook(self._save_g)
    def _save_a(self, module, inp, out):
        self.a = out.detach()
    def _save_g(self, module, gin, gout):
        self.g = gout[0].detach()
    def close(self):
        self.h1.remove(); self.h2.remove()

@torch.no_grad()
def _to_grid(tensor_images: torch.Tensor, nrow: int = 8):
    return torchvision.utils.make_grid(tensor_images.clamp(0,1), nrow=nrow)

def _overlay_heatmap_on_images(images: torch.Tensor, cams: torch.Tensor, alpha: float = 0.45) -> torch.Tensor:
    B, _, H, W = images.shape
    overlays = []
    for i in range(B):
        img = images[i].permute(1,2,0).cpu().numpy()  # H,W,3
        cam = cams[i,0].cpu().numpy()                 # H,W
        cmap = plt.get_cmap("jet")(cam)[..., :3]      # H,W,3
        overlay = (1-alpha)*img + alpha*cmap
        overlay = np.clip(overlay, 0, 1)
        overlays.append(torch.from_numpy(overlay).permute(2,0,1))
    return torch.stack(overlays, dim=0)

def log_feature_maps(writer, model, images, step, tag_prefix="FeatMaps"):
    # Try to find the first conv-like layer
    first_conv = None
    for m in model.modules():
        if isinstance(m, (nn.Conv2d,)):
            first_conv = m
            break
    if first_conv is None:
        return
    activation = {}
    def hook_fn(module, inp, out):
        activation['fm'] = out.detach().cpu()
    handle = first_conv.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            _ = model(images)
        fm = activation.get('fm')
        if fm is None: 
            return
        fm0 = fm[0]  # (C,H,W)
        C = min(32, fm0.shape[0]); fm0 = fm0[:C]
        fm0 = (fm0 - fm0.amin(dim=(1,2), keepdim=True)) / (fm0.amax(dim=(1,2), keepdim=True) - fm0.amin(dim=(1,2), keepdim=True) + 1e-6)
        grid = torchvision.utils.make_grid(fm0.unsqueeze(1), nrow=8, normalize=False, scale_each=False)
        writer.add_image(f"{tag_prefix}/first_conv", grid, step)
    finally:
        handle.remove()

def log_gradcam(writer, model, images, device, step, tag="GradCAM/val", max_items: int = 8):
    model.eval()
    images = images[:max_items].to(device)
    logits = model(images)
    preds = logits.argmax(dim=1)
    layer = _find_last_conv(model)
    if layer is None:
        return
    hook = _GradCAMHook(layer)
    loss = logits[torch.arange(images.size(0)), preds].sum()
    model.zero_grad(set_to_none=True)
    loss.backward(retain_graph=False)
    a, g = hook.a, hook.g  # (B,C,H,W)
    hook.close()
    if a is None or g is None:
        return
    B, C, H, W = a.shape
    weights = g.view(B, C, -1).mean(dim=2)           # (B,C)
    cams = torch.einsum("bc, bchw -> bhw", weights, a)  # (B,H,W)
    cams = torch.relu(cams)
    cams = cams - cams.view(B, -1).min(dim=1, keepdim=True)[0].unsqueeze(-1)
    cams = cams / (cams.view(B, -1).max(dim=1, keepdim=True)[0].unsqueeze(-1) + 1e-6)
    cams = cams.unsqueeze(1)
    if cams.shape[-2:] != images.shape[-2:]:
        cams = torch.nn.functional.interpolate(cams, size=images.shape[-2:], mode="bilinear", align_corners=False)
    overlays = _overlay_heatmap_on_images(images.cpu(), cams.cpu(), alpha=0.45)
    grid = _to_grid(overlays, nrow=min(4, overlays.size(0)))
    writer.add_image(tag, grid, step)

def parse_args():
    p=argparse.ArgumentParser(description="Modular CNN Trainer")
    p.add_argument('--config', type=str, required=True)
    return p.parse_args()

def main():
    args=parse_args(); cfg=yaml.safe_load(open(args.config)); set_seed(cfg['misc']['seed'])
    misc_cfg = cfg.get('misc', {})
    logging_mode = str(misc_cfg.get('logging','full')).lower()  # 'full' | 'none'
    log_every_n = int(misc_cfg.get('log_every_n', 1))
    enable_confusion = bool(misc_cfg.get('log_confusion', logging_mode != 'none'))
    enable_feature_maps = bool(misc_cfg.get('log_feature_maps', logging_mode != 'none'))
    enable_gradcam = bool(misc_cfg.get('log_gradcam', logging_mode != 'none'))
    csv_logging = bool(misc_cfg.get('csv_logging', logging_mode != 'none'))
    # Prefer CUDA, then Apple Metal (MPS), else CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    out_dir=Path(cfg['misc']['out_dir'])/cfg['experiment_name']; out_dir.mkdir(parents=True, exist_ok=True)
    log_path=out_dir/'log.csv'
    # Save checkpoints under models/representations/exp/<experiment_name>
    ckpt_dir = Path('models/representations/exp')/cfg['experiment_name']
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best=ckpt_dir/'best.pt'; ckpt_last=ckpt_dir/'last.pt'
    writer=None
    if logging_mode != 'none':
        tb_dir=out_dir/'tb'; writer=SummaryWriter(log_dir=str(tb_dir))

    model=make_model(cfg['model'], num_classes=10).to(device); total, trainable=count_params(model)
    print(f"Model: {cfg['model']} | params: {trainable:,} / {total:,}")

    # TB: graph
    if writer is not None:
        try:
            dummy=torch.randn(1,3,32,32, device=device); writer.add_graph(model, dummy)
        except Exception as e:
            print('[TB] graph skipped:', e)

    trainloader, testloader = make_cifar10_loaders(
        cfg['data']['root'],
        cfg['train']['batch_size'],
        cfg['misc']['num_workers'],
        max_train=cfg['data'].get('max_train'),
        max_test=cfg['data'].get('max_test'),
    )

    # TB: image previews
    if writer is not None:
        try:
            tr_imgs,_=next(iter(trainloader)); te_imgs,_=next(iter(testloader))
            writer.add_image('Preview/Train', torchvision.utils.make_grid(tr_imgs[:32], nrow=8, normalize=True, scale_each=True), 0)
            writer.add_image('Preview/Test',  torchvision.utils.make_grid(te_imgs[:32],  nrow=8, normalize=True, scale_each=True), 0)
        except Exception as e:
            print('[TB] images skipped:', e)

    criterion = LabelSmoothingLoss(10, cfg['train']['label_smoothing']).to(device) if cfg['train']['label_smoothing']>0 else nn.CrossEntropyLoss().to(device)
    optimizer = make_optimizer(model, cfg['train']['lr'], cfg['train']['weight_decay'], cfg['train']['optimizer'])
    scheduler = make_scheduler(optimizer, cfg['train']['scheduler'], cfg['train']['epochs'])
    # AMP only on CUDA. Use new torch.amp API and pass device_type
    use_amp = bool(cfg['misc']['amp']) and (device.type == 'cuda')
    try:
        from torch import amp as _amp  # PyTorch >=2.0
        scaler = _amp.GradScaler(device_type=device.type, enabled=use_amp)
    except Exception:
        scaler = None

    state=TrainState()
    with open(log_path,'a',newline="") as f:
        w=csv.writer(f)
        if csv_logging and f.tell()==0:
            w.writerow(['epoch','train_loss','train_acc','val_loss','val_acc','lr'])
        for epoch in range(cfg['train']['epochs']):
            state.epoch=epoch+1; lr_now=optimizer.param_groups[0]['lr']
            tr_loss,tr_acc = train_one_epoch(model, trainloader, optimizer, device, scaler, criterion, grad_clip=cfg['train']['grad_clip'] or None)
            val_loss,val_acc = evaluate(model, testloader, device, criterion)
            if scheduler: scheduler.step()

            # Scalars
            if writer is not None:
                writer.add_scalar('Loss/train', tr_loss, state.epoch)
                writer.add_scalar('Loss/val',   val_loss, state.epoch)
                writer.add_scalar('Acc/train',  tr_acc,   state.epoch)
                writer.add_scalar('Acc/val',    val_acc,  state.epoch)
                writer.add_scalar('LR',         lr_now,   state.epoch)

            # Histograms (heavy): only when logging enabled and at interval
            if writer is not None and (state.epoch % log_every_n == 0):
                for name, param in model.named_parameters():
                    try:
                        writer.add_histogram(f'Weights/{name}', param.data.detach().cpu(), state.epoch)
                        if param.grad is not None:
                            writer.add_histogram(f'Grads/{name}', param.grad.detach().cpu(), state.epoch)
                    except Exception:
                        pass

            # Per-class & confusion (heavy): only when enabled and at interval
            if writer is not None and enable_confusion and (state.epoch % log_every_n == 0):
                try:
                    cm, per_class = compute_confusion_and_per_class(model, testloader, device, num_classes=10)
                    for cls_idx, accv in per_class.items():
                        writer.add_scalar(f'PerClassAcc/class_{cls_idx}', accv, state.epoch)
                    fig = plot_confusion_matrix(cm, class_names=getattr(testloader.dataset, "classes", [str(i) for i in range(10)]))
                    writer.add_figure('ConfusionMatrix/val', fig, state.epoch); plt.close(fig)
                except Exception as e:
                    print('[TB] per-class/confusion skipped:', e)

            # Feature maps & Grad-CAM (heavy): only when enabled and at interval
            if writer is not None and (state.epoch % log_every_n == 0):
                try:
                    if enable_feature_maps:
                        batch_images, _ = next(iter(trainloader))
                        batch_images = batch_images.to(device)
                        log_feature_maps(writer, model, batch_images, state.epoch)
                except Exception as e:
                    print('[TB] feature maps skipped:', e)
                try:
                    if enable_gradcam:
                        val_batch = next(iter(testloader))
                        images, _ = val_batch
                        log_gradcam(writer, model, images, device, state.epoch, tag='GradCAM/val', max_items=8)
                except Exception as e:
                    print('[TB] grad-cam skipped:', e)

            print(f"[{state.epoch}/{cfg['train']['epochs']}] train_loss={tr_loss:.4f} acc={tr_acc*100:.2f}% | val_loss={val_loss:.4f} acc={val_acc*100:.2f}%")
            if csv_logging:
                w.writerow([state.epoch, tr_loss, tr_acc, val_loss, val_acc, lr_now]); f.flush()

            # Checkpointing
            try:
                # Always save last
                torch.save({'model': model.state_dict(), 'epoch': state.epoch, 'val_acc': val_acc}, ckpt_last)
                # Save best by val_acc
                if val_acc > state.best_val_acc:
                    state.best_val_acc = val_acc
                    torch.save({'model': model.state_dict(), 'epoch': state.epoch, 'val_acc': val_acc}, ckpt_best)
            except Exception as e:
                print('[warn] checkpoint save failed:', e)

    if writer is not None:
        writer.close()
    print('Done. Logs at', log_path)

if __name__=='__main__': main()
