"""
Shared CLI utilities for building consistent command-line interfaces across use-cases.
"""

import argparse
from dataclasses import dataclass


def add_common_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Mini-batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay (L2)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--opt', type=str, default='sgd', choices=['sgd','momentum','nesterov','adam','adamw'], help='Optimizer')
    parser.add_argument('--beta1', type=float, default=0.9, help='Momentum/Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta2')
    parser.add_argument('--eps', type=float, default=1e-8, help='Adam eps')
    parser.add_argument('--init', type=str, default='he', choices=['he','xavier','normal'], help='Weight init for W1/W2')
    parser.add_argument('--clip-grad', type=float, default=0.0, help='Global grad-norm clip (0=off)')
    parser.add_argument('--lr-sched', type=str, default='none', choices=['none','step','cosine'], help='LR scheduler')
    parser.add_argument('--lr-step', type=int, default=5, help='StepLR step size (epochs)')
    parser.add_argument('--lr-gamma', type=float, default=0.5, help='StepLR decay factor')
    parser.add_argument('--warmup-epochs', type=int, default=0, help='Cosine warmup epochs')
    return parser


@dataclass
class OptimSchedConfig:
    opt: str
    lr: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float
    nesterov: bool
    lr_sched: str
    lr_step: int
    lr_gamma: float
    warmup_epochs: int


def build_optim_and_scheduler(model_params, args, OptimCls, SchedulerCls):
    optim = OptimCls(model_params, opt=args.opt, lr=args.lr, beta1=args.beta1, beta2=args.beta2, eps=args.eps,
                     weight_decay=args.weight_decay, nesterov=(args.opt=='nesterov'))
    sched = SchedulerCls(args.lr, kind=args.lr_sched, step=args.lr_step, gamma=args.lr_gamma,
                         total_epochs=args.epochs, warmup=args.warmup_epochs)
    return optim, sched
