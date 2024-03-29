import numpy as np
import glob

from tqdm import tqdm
import torch
import torchaudio
import os
from argparse import ArgumentParser

from sgmse.backbones.shared import BackboneRegistry
from sgmse.data_module import SpecsDataModule
from sgmse.sdes import SDERegistry
from sgmse.sampling import PredictorRegistry, CorrectorRegistry, OperatorRegistry, PosteriorRegistry, SchedulerRegistry, PosteriorRegistry
from sgmse.model import ScoreModel
from preprocessing import kernel as ker
import gc

from sgmse.util.other import *

import matplotlib.pyplot as plt

EPS_LOG = 1e-10

torch.random.manual_seed(0)

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
    if sr != model.data_module.sample_rate:
        y = torchaudio.transforms.Resample(orig_freq=sr, new_freq=model.data_module.sample_rate)(y)
    y = y[..., : int(3.5*sr)]
    # y = y[..., : int(12*sr)]

    return y, A, zeta, operator, zeta_schedule
        




# Tags
base_parser = ArgumentParser(add_help=False)
parser = ArgumentParser()
for parser_ in (base_parser, parser):
    # parser_.add_argument("--test_dir", type=str, required=True, help="Directory containing your corrupted files to enhance.")
    # parser_.add_argument("--enhanced_dir", type=str, required=True, help="Where to write your cleaned files.")
    # parser_.add_argument("--ckpt", type=str, help="Which pretrained checkpoint to use", required=True) 
    # parser_.add_argument("--rir_dir", type=str, required=True, help="Directory containing your RIRs.")

    parser_.add_argument("--test_dir", type=str, help="Directory containing your corrupted files to enhance.", default="/data3/lemercier/databases/wsj0_derev_with_rir/audio/tt/noisy")
    parser_.add_argument("--enhanced_dir", type=str, help="Where to write your cleaned files.", default="./results")
    parser_.add_argument("--ckpt", type=str, help="Which pretrained checkpoint to use", default="/export/home/lemercier/code/score_derev/.logs/waspaa2023/mode=score-only_sde=VESDE_backbone=ncsnpp_data=reverb_ch=1/version_11_alpha=1.0_beta=0.1_sigma=0.5_pre=song/checkpoints/epoch=204.ckpt")
    parser_.add_argument("--rir_dir", type=str, default="/data3/lemercier/databases/wsj0_derev_with_rir/rir/tt", help="Directory containing your RIRs.")
    
    parser_.add_argument("--n", type=int, default=-1, help="Number of cropped files")
    parser_.add_argument("--gpu", type=int, default=0, help="Which GPU to perform inference on")

    parser_.add_argument("--sampler_type", type=str, default="song", choices=["song", "karras", "red-diff"])
    parser_.add_argument("--no_probability_flow", action="store_true", help="Use SDE sampling instead of ODE probability flow sampling.")
    parser_.add_argument("--N", type=int, default=50, help="Number of reverse steps")
    parser_.add_argument("--scheduler", type=str, default="linear", choices=SchedulerRegistry.get_all_names())

    parser_.add_argument("--predictor", type=str, default="euler-maruyama", choices=PredictorRegistry.get_all_names(), help="Predictor class for the PC sampler.")

    parser_.add_argument("--corrector", type=str, default="ald", choices=CorrectorRegistry.get_all_names(), help="Corrector class for the PC sampler.")
    parser_.add_argument("--corrector_steps", type=int, default=1, help="Number of corrector steps")
    parser_.add_argument("--r", type=float, default=0.4, help="Corrector step size for ALD.")

    parser_.add_argument("--operator", type=str, default="reverberation", choices=["none"] + OperatorRegistry.get_all_names())
    parser_.add_argument("--posterior", type=str, default="dps", choices=["none"] + PosteriorRegistry.get_all_names())
    parser_.add_argument("--zeta", type=float, default=2500, help="Step size for log-likelihood term." + 
                         "Attention: in the paper the value reported is 50. However when rescaling by" + 
                         "the usual number of steps N=50 (correction by dt in the code after submitting the paper), one needs to use the value zeta0*N = 2500")
    parser_.add_argument("--zeta_schedule", type=str, default="saw-tooth-increase", help="Anneal the log-likelihood term with a zeta step size schedule.")
    parser_.add_argument("--sw", type=float, default=None, help="Switching time between posteriors if posterior==switching.")
    parser_.add_argument("--churn", type=float, default=10, help="Karras sampler.") 
    
    parser_.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser_.add_argument("--lr", type=float, default=1e-1, help="Learning rate for optimizer used in RED-Diff.")

    parser_.add_argument("--measurement_noise", type=float, default=None, help="Additive Gaussian measurement noise. Given as a SNR in dB.")

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
torch.cuda.set_device(f'cuda:{args.gpu}')
model.cuda()

files = sorted(glob.glob(os.path.join(args.test_dir, "*.wav")))
files = files[: args.n] if args.n > 0 else files

for i, f in tqdm.tqdm(enumerate(files), total=len(files)):


    kernel_kwargs = {
        "class": "RealRIRKernel",
        "rir_path": args.rir_dir,
        "stft": False,
        "size": 16000
    }
    y, A, zeta, operator, zeta_schedule = get_posterior_sampling_args(
        model,
        file=f, 
        i=i,
        args=args,
        kernel_kwargs=kernel_kwargs)

    other_kwargs = dict(path=args.enhanced_dir, unconditional_prior=True)

    x_hat = model.enhance(y, 
        sampler_type=args.sampler_type, probability_flow=not(args.no_probability_flow), N=args.N, scheduler=args.scheduler,
        predictor=args.predictor,
        corrector=args.corrector, corrector_steps=args.corrector_steps, r=args.r,
        smin=model.sde.sigma_min, smax=model.sde.sigma_max, churn=args.churn,
        posterior=args.posterior, operator=operator, A=A,  zeta=zeta, zeta_schedule=zeta_schedule, sw=args.sw,
        optimizer=args.optimizer, lr=args.lr,
        **other_kwargs)
    
    torchaudio.save(f'{args.enhanced_dir}/{os.path.basename(f)}', x_hat.type(torch.float32).cpu().squeeze().unsqueeze(0), model_sr)

    y = None
    x_hat = None
