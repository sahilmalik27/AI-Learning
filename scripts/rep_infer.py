import os, sys, argparse, yaml
import torch
import torch.nn.functional as F
from PIL import Image

# Ensure 'src' is importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from representation_learning.cnn_suite import make_model
from representation_learning.exp.data import make_cifar10_loaders


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_model_from_ckpt(model_name: str, ckpt_path: str, device: torch.device):
    model = make_model(model_name, num_classes=10).to(device)
    state = torch.load(ckpt_path, map_location=device)
    state_dict = state.get('model', state)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_transform():
    # CIFAR-10 default transform used in training
    from torchvision import transforms as T
    return T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
    ])


def infer_image(model, device, image_path: str, class_names):
    tfm = build_transform()
    img = Image.open(image_path).convert('RGB')
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu()
        top_prob, top_idx = probs.max(dim=0)
    pred_label = class_names[top_idx.item()] if class_names else str(top_idx.item())
    print(f"Pred: {top_idx.item()} ({pred_label}) | prob={top_prob.item():.4f}")


def infer_sample(model, device, cfg, sample_index: int):
    # Use the test loader to fetch one sample with identical transforms
    _, test_loader = make_cifar10_loaders(
        cfg['data']['root'],
        batch_size=1,
        num_workers=0,
        max_test=cfg['data'].get('max_test'),
        max_train=cfg['data'].get('max_train'),
    )
    class_names = getattr(test_loader.dataset, 'classes', [str(i) for i in range(10)])
    # Iterate to the desired index (simple, reliable for Subset)
    from itertools import islice
    sample = next(islice(test_loader, sample_index, None))
    x, y = sample
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu()
        top_prob, top_idx = probs.max(dim=0)
    pred_label = class_names[top_idx.item()] if class_names else str(top_idx.item())
    true_label = class_names[y.item()] if class_names else str(y.item())
    print(f"Idx {sample_index} | True: {y.item()} ({true_label}) | Pred: {top_idx.item()} ({pred_label}) | prob={top_prob.item():.4f}")


def parse_args():
    p = argparse.ArgumentParser(description='Representation inference utility')
    p.add_argument('--config', required=True, help='Path to experiment YAML config')
    p.add_argument('--ckpt', required=True, help='Path to checkpoint (.pt)')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--image', help='Path to an RGB image to classify')
    g.add_argument('--sample-index', type=int, help='Index from the test subset to classify')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = pick_device()
    model = load_model_from_ckpt(cfg['model'], args.ckpt, device)

    # For class names convenience
    _, test_loader = make_cifar10_loaders(
        cfg['data']['root'], batch_size=1, num_workers=0,
        max_test=cfg['data'].get('max_test'), max_train=cfg['data'].get('max_train')
    )
    class_names = getattr(test_loader.dataset, 'classes', [str(i) for i in range(10)])

    if args.image:
        infer_image(model, device, args.image, class_names)
    else:
        infer_sample(model, device, cfg, args.sample_index)


if __name__ == '__main__':
    main()


