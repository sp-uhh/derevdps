import numpy as np
import glob

from tqdm import tqdm
from torchaudio import load, save
import torch
import os
from argparse import ArgumentParser

from sgmse.backbones.shared import BackboneRegistry
from sgmse.data_module import SpecsDataModule
from sgmse.sdes import SDERegistry
from sgmse.sampling import PredictorRegistry, CorrectorRegistry, OperatorRegistry, PosteriorRegistry, SchedulerRegistry, PosteriorRegistry
from sgmse.model import ScoreModel
from preprocessing import kernel as ker

from sgmse.util.other import *

import matplotlib.pyplot as plt

EPS_LOG = 1e-10

# Tags
base_parser = ArgumentParser(add_help=False)
parser = ArgumentParser()
for parser_ in (base_parser, parser):
    parser_.add_argument("--gt_dir", type=str, required=False, help="Directory containing the paired clean files.")
    parser_.add_argument("--generated_dir", type=str, required=True, help="Where to write your cleaned files.")
    parser_.add_argument("--ckpt", required=True)
    parser_.add_argument("--n", type=int, default=-1, help="Number of cropped files")

    parser_.add_argument("--N", type=int, default=50, help="Number of reverse steps")
    parser_.add_argument("--sampler_type", type=str, default="karras", choices=["song", "karras", "BABE"])
    parser_.add_argument("--predictor", type=str, default="euler-heun", choices=PredictorRegistry.get_all_names(), help="Predictor class for the PC sampler.")
    parser_.add_argument("--corrector", type=str, default="none", choices=CorrectorRegistry.get_all_names(), help="Corrector class for the PC sampler.")
    parser_.add_argument("--scheduler", type=str, default="karras_10_23")
    parser_.add_argument("--noise_std", type=float, default=1.)
    parser_.add_argument("--smin", type=float, default=0.)
    parser_.add_argument("--smax", type=float, default=0.)
    parser_.add_argument("--churn", type=float, default=0.)

args = parser.parse_args()

os.makedirs(args.generated_dir, exist_ok=True)

# Settings
model_sr = 16000

# Load score model
model_cls = ScoreModel

print("loading checkpoint", args.ckpt)

model = model_cls.load_from_checkpoint(
    args.ckpt, base_dir="",
    batch_size=1, num_workers=0, 
    gpu=False 
)
model.eval(no_ema=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

gt_files = sorted(glob.glob(os.path.join(args.gt_dir, "*.wav")))
gt_files = gt_files[: args.n] if args.n > 0 else gt_files
_len_generation = 5.

for i, f in tqdm.tqdm(enumerate(gt_files), total=len(gt_files)):

    reference_tensor = torch.zeros(1, int(_len_generation*model.data_module.sample_rate))

    model.t_eps = 0.0
    x_hat = model.unconditional_sampling(
        reference_tensor, 
        N=args.N,
        sampler_type=args.sampler_type,
        predictor=args.predictor,
        corrector=args.corrector,
        scheduler=args.scheduler,
        noise_std=args.noise_std,
        smin=args.smin,
        smax=args.smax,
        churn=args.churn,
        probability_flow=True,
        ).squeeze()

    save(f'{args.generated_dir}/{os.path.basename(f)}', x_hat.type(torch.float32).cpu().squeeze().unsqueeze(0), model_sr)
