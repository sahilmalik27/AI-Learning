# exp/gradcam_resnet_tiny.py
# Grad-CAM for ResNet18Tiny on CIFAR-10
# Usage:
#   pip install torch torchvision matplotlib opencv-python
#   python exp/gradcam_resnet_tiny.py --ckpt runs/cifar10_resnet/best.pt --out runs/cifar10_resnet/gradcam --num 12

import os
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T

from cnn_suite import ResNet18Tiny

def tensor_to_img(t):
    t = t.clamp(0,1).detach().cpu().numpy()
    return np.transpose(t, (1,2,0))

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook_a = target_layer.register_forward_hook(self._save_activation)
        self.hook_g = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self):
        B, C, H, W = self.activations.shape
        cams = []
        for b in range(B):
            grads = self.gradients[b]
            acts  = self.activations[b]
            weights = grads.view(C, -1).mean(dim=1)
            cam = torch.sum(weights[:, None, None] * acts, dim=0)
            cam = F.relu(cam)
            cam -= cam.min()
            cam = cam / (cam.max() + 1e-6)
            cams.append(cam.cpu().numpy())
        return np.stack(cams, axis=0)

    def close(self):
        self.hook_a.remove()
        self.hook_g.remove()

def find_last_conv(model):
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last

def overlay_cam_on_image(img, cam, alpha=0.45):
    try:
        import cv2
        heatmap = (cam * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)[:, :, ::-1]
        heatmap = heatmap.astype(np.float32) / 255.0
        overlay = (1 - alpha) * img + alpha * heatmap
        return np.clip(overlay, 0, 1)
    except Exception:
        cmap = plt.get_cmap("jet")(cam)[..., :3]
        overlay = 0.55 * img + 0.45 * cmap
        return np.clip(overlay, 0, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="", help="Path to checkpoint .pt (optional)")
    ap.add_argument("--out", type=str, default="runs/gradcam_demo")
    ap.add_argument("--num", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18Tiny(num_classes=10).to(device)

    if args.ckpt and os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)

    tf = T.Compose([T.ToTensor()])
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=tf)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.num, shuffle=True, num_workers=2)

    images, labels = next(iter(testloader))
    images, labels = images.to(device), labels.to(device)

    logits = model(images)
    preds = logits.argmax(dim=1)

    target_layer = find_last_conv(model)
    cam = GradCAM(model, target_layer)

    cams_list = []
    for i in range(images.size(0)):
        model.zero_grad(set_to_none=True)
        score = logits[i, preds[i]]
        score.backward(retain_graph=True)
        cams = cam.generate()
        cams_list.append(cams[i])
    cam.close()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    class_names = testset.classes

    for i in range(images.size(0)):
        img = tensor_to_img(images[i])
        heat = cams_list[i]
        overlay = overlay_cam_on_image(img, heat, alpha=0.45)

        pred_lbl = class_names[int(preds[i].item())]
        true_lbl = class_names[int(labels[i].item())]

        fig, ax = plt.subplots(1,3, figsize=(9,3))
        ax[0].imshow(img);   ax[0].set_title(f"Image\nTrue:{true_lbl}")
        ax[1].imshow(heat, cmap="jet"); ax[1].set_title("Grad-CAM")
        ax[2].imshow(overlay); ax[2].set_title(f"Overlay\nPred:{pred_lbl}")
        for a in ax: a.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"gradcam_{i:02d}_{true_lbl}_pred_{pred_lbl}.png", dpi=120)
        plt.close(fig)

    print(f"Saved {images.size(0)} Grad-CAM visualizations to {out_dir}")

if __name__ == "__main__":
    main()
