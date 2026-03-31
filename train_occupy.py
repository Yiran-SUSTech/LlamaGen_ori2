import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import broadcast
from torch.optim.lr_scheduler import LambdaLR

import os
import sys
import time
import inspect
import argparse
import math

import wandb
import numpy as np

param = 7 * 24 * 60 * 60 * 1.5



            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-path", type=str, required=True)
    parser.add_argument("--cloud-save-path", type=str, required=True, help='please specify a cloud disk path, if not, local path')
    parser.add_argument("--no-local-save", action='store_true', help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--gpt-model", type=str, default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--vq-model", type=str, default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--vocab-size", type=int, default=16384, help="vocabulary size of visual tokenizer")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--ema", action='store_true', help="whether using ema training")
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path-rate", type=float, default=0.0, help="using stochastic depth decay")
    parser.add_argument("--no-compile", action='store_true')
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--dataset", type=str, default='imagenet_code')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=384)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--adam-block-size", type=int, choices=[1,2,4,8,16], default=8)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 parameter for the Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.95, help="beta2 parameter for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10) # log every log_every steps
    parser.add_argument("--ckpt-every", type=int, default=5000) # save checkpoint every ckpt_every epochs
    parser.add_argument("--is-wandb-log", action='store_true')
    parser.add_argument("--wandb_offline", action='store_true')
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--num-datapoints", type=int, default=None, help="number of data points to train on")
    parser.add_argument("--num-data", type=int, default=None, help="number of data points to train on") # sample only first num_data files
    parser.add_argument("--from_llamagen", action='store_true')
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--profiler_port", type=int, default=9012, help="the port of investigation") 
    parser.add_argument("--profile", action='store_true', default=True)
    parser.add_argument("--num-workers", type=int, default=24)  #############################################################
    parser.add_argument("--warmup_percent", type=float, default=0.1, help="the ratio of warm-up steps in total number of steps")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-k", type=int, default=0,help="top-k value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--cfg-scale",  type=float, default=1.0)
    parser.add_argument("--is-lr-scheduler", action='store_true')
    parser.add_argument("--subpass-len", type=int, default=None, help="the length of each subpass, None means no subpass")
    parser.add_argument("--subpass-num", type=int, default=None, help="the number of subpasses within each pass, None means no subpass")
    parser.add_argument("--pre_token_choose", type=str, choices=['close_min', 'close_max', 'close_unattach_min', 'close_unattach_max', 'close_left_up'], default="close_min")
    parser.add_argument("--freqs_cis_reorder_shceme", type=str, choices=['output_reorder', 'input_reorder', 'None'], default='None')

    args = parser.parse_args()
    time.sleep(param)