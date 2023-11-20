import argparse
from argparse import ArgumentParser
import os
import sys

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch

from sgmse.backbones.shared import BackboneRegistry
from sgmse.data_module import SpecsDataModule
from sgmse.sdes import SDERegistry
from sgmse.model import ScoreModel

from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint

def get_argparse_groups(parser):
	groups = {}
	for group in parser._action_groups:
		group_dict = { a.dest: getattr(args, a.dest, None) for a in group._group_actions }
		groups[group.title] = argparse.Namespace(**group_dict)
	return groups


if __name__ == '__main__':

	# throwaway parser for dynamic args - see https://stackoverflow.com/a/25320537/3090225
	base_parser = ArgumentParser(add_help=False)
	parser = ArgumentParser()
	for parser_ in (base_parser, parser):
		parser_.add_argument("--backbone", type=str, choices=["none"] + BackboneRegistry.get_all_names(), default="ncsnpp")
		parser_.add_argument("--sde", type=str, choices=SDERegistry.get_all_names(), default="ve")
		parser_.add_argument("--nolog", action='store_true', help="Turn off logging (for development purposes)")
	temp_args, _ = base_parser.parse_known_args()

	model_cls = ScoreModel

	# Add specific args for ScoreModel, pl.Trainer, the SDE class and backbone DNN class
	backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
	sde_class = SDERegistry.get_by_name(temp_args.sde)
	parser = pl.Trainer.add_argparse_args(parser)
	model_cls.add_argparse_args(
		parser.add_argument_group(model_cls.__name__, description=model_cls.__name__))
	sde_class.add_argparse_args(
		parser.add_argument_group("SDE", description=sde_class.__name__))
	backbone_cls.add_argparse_args(
		parser.add_argument_group("Backbone", description=backbone_cls.__name__))

	# Add data module args
	data_module_cls = SpecsDataModule
	data_module_cls.add_argparse_args(
		parser.add_argument_group("DataModule", description=data_module_cls.__name__))
	# Parse args and separate into groups
	args = parser.parse_args()
	arg_groups = get_argparse_groups(parser)

	model = model_cls(
		backbone=args.backbone, sde=args.sde, data_module_cls=data_module_cls,
		**{
			**vars(arg_groups['ScoreModel']),
			**vars(arg_groups['SDE']),
			**vars(arg_groups['Backbone']),
			**vars(arg_groups['DataModule'])
		},
		nolog=args.nolog
	)
	data_tag = model.data_module.base_dir.strip().split("/")[-3] if model.data_module.format == "whamr" else model.data_module.base_dir.strip().split("/")[-1] 
	logging_name = f"sde={sde_class.__name__}_backbone={args.backbone}_data={model.data_module.format}_ch={model.data_module.spatial_channels}"

	logger = TensorBoardLogger(save_dir=f"./.logs/", name=logging_name, flush_secs=30) if not args.nolog else None

	# Callbacks
	callbacks = []
	callbacks.append(EarlyStopping(monitor="valid_loss", mode="min", patience=50))
	callbacks.append(TQDMProgressBar(refresh_rate=50))
	if not args.nolog:
		callbacks.append(ModelCheckpoint(dirpath=os.path.join(logger.log_dir, "checkpoints"), 
			save_last=True, save_top_k=1, monitor="valid_loss", filename='{epoch}'))
		callbacks.append(ModelCheckpoint(dirpath=os.path.join(logger.log_dir, "checkpoints"), 
			save_top_k=1, monitor="ValidationPESQ", mode="max", filename='{epoch}-{pesq:.2f}'))

	# additional_kwargs = {"gradient_clip_val": 75., "gradient_clip_algorithm": "value"} if args.grad_clip else {}
	additional_kwargs = {}
	
	# Initialize the Trainer and the DataModule
	trainer = pl.Trainer.from_argparse_args(
		arg_groups['pl.Trainer'],
		strategy=DDPStrategy(find_unused_parameters=False), 
		logger=logger,
		log_every_n_steps=10, num_sanity_val_steps=0, 
		callbacks=callbacks,
		max_epochs=300,
		**additional_kwargs
	)

	# Train model
	trainer.fit(model)