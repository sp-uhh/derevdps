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

def get_posterior_sampling_args(model, file, i, args, kernel_kwargs):

    if args.operator != "none":

        operator = OperatorRegistry.get_by_name(args.operator)(kernel_size=kernel_kwargs["size"], stft=kernel_kwargs["stft"], **model.data_module.stft_kwargs)
        kernel = getattr(ker, kernel_kwargs["class"])()
        kernel_kwargs["i"] = i
        A = kernel.forward(**kernel_kwargs)
        operator.load_weights(A)
        zeta = args.zeta
        zeta_schedule = args.zeta_schedule

    else:
        operator, A, zeta, zeta_schedule = None, None, None, None

    y, sr = torchaudio.load(file)

    return y, A, zeta, operator, zeta_schedule
        




# Tags
base_parser = ArgumentParser(add_help=False)
parser = ArgumentParser()
for parser_ in (base_parser, parser):
    parser_.add_argument("--test_dir", type=str, required=True, help="Directory containing your corrupted files to enhance.")

    parser_.add_argument("--enhanced_dir", type=str, required=True, help="Where to write your cleaned files.")
    parser_.add_argument("--ckpt", required=True)
    parser_.add_argument("--n", type=int, default=-1, help="Number of cropped files")

    parser_.add_argument("--sampler_type", type=str, default="song", choices=["song", "karras"])
    parser_.add_argument("--no_probability_flow", action="store_true", help="Use SDE sampling instead of ODE probability flow sampling.")
    parser_.add_argument("--N", type=int, default=50, help="Number of reverse steps")
    parser_.add_argument("--scheduler", type=str, default="linear", choices=SchedulerRegistry.get_all_names())

    parser_.add_argument("--predictor", type=str, default="euler-maruyama", choices=PredictorRegistry.get_all_names(), help="Predictor class for the PC sampler.")

    parser_.add_argument("--corrector", type=str, default="ald", choices=CorrectorRegistry.get_all_names(), help="Corrector class for the PC sampler.")
    parser_.add_argument("--corrector_steps", type=int, default=1, help="Number of corrector steps")
    parser_.add_argument("--r", type=float, default=0.4, help="Corrector step size for ALD.")

    parser_.add_argument("--operator", type=str, default="reverberation", choices=["none"] + OperatorRegistry.get_all_names())
    parser_.add_argument("--posterior", type=str, default="dps", choices=["none"] + PosteriorRegistry.get_all_names())
    parser_.add_argument("--zeta", type=float, default=50, help="Step size for log-likelihood term.")
    parser_.add_argument("--zeta_schedule", type=str, default="saw-tooth-increase", help="Anneal the log-likelihood term with a zeta step size schedule.")
    parser_.add_argument("--sw", type=float, default=None, help="Switching time between posteriors if posterior==switching.")
    
    parser_.add_argument("--measurement_noise", type=float, default=None, help="Additive Gaussian measurement noise. Given as a SNR in dB.")

    parser_.add_argument("--kernel_kwargs", type=dict, default={
        "class": "RealRIRKernel",
        "rir_path": "/data/lemercier/databases/wsj0_derev_with_rir/rir/tt",
        "stft": False,
        "size": 16000
    })

args = parser.parse_args()

os.makedirs(args.enhanced_dir, exist_ok=True)

# Settings
model_sr = 16000

# Load score model
model_cls = ScoreModel
model = model_cls.load_from_checkpoint(
    args.ckpt, base_dir="",
    batch_size=1, num_workers=0, 
    gpu=False
)
model.eval(no_ema=False)

model.cuda()

files = sorted(glob.glob(os.path.join(args.test_dir, "*.wav")))
files = files[: args.n] if args.n > 0 else files

for i, f in tqdm.tqdm(enumerate(files), total=len(files)):

    y, A, zeta, operator, zeta_schedule = get_posterior_sampling_args(
        model,
        file=f, 
        i=i,
        args=args,
        kernel_kwargs=args.kernel_kwargs)

    other_kwargs = dict(path=args.enhanced_dir, unconditional_prior=True)

    x_hat = model.enhance(y, 
        sampler_type=args.sampler_type, probability_flow=not(args.no_probability_flow), N=args.N, scheduler=args.scheduler,
        predictor=args.predictor,
        corrector=args.corrector, corrector_steps=args.corrector_steps, r=args.r, 
        posterior=args.posterior, operator=operator, A=A,  zeta=zeta, zeta_schedule=zeta_schedule, sw=args.sw,
        **other_kwargs)

    save(f'{args.enhanced_dir}/{os.path.basename(f)}', x_hat.type(torch.float32).cpu().squeeze().unsqueeze(0), model_sr)