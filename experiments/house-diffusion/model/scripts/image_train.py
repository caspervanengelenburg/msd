"""
Train a diffusion model on images.
"""

import argparse

from house_diffusion import dist_util, logger
from house_diffusion.rplanhg_datasets import load_rplanhg_structural_data
from house_diffusion.resample import create_named_schedule_sampler
from house_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    update_arg_parser,
)
from house_diffusion.train_util import TrainLoop

from house_diffusion.modified_swiss_dwellings_housediffusion_dataset import load_modified_swiss_dwellings, get_dataloader_modified_swiss_dwellings

def main():
    args = create_argparser().parse_args()
    update_arg_parser(args)

    print("Going to run: dist_util.setup_dist()")

    dist_util.setup_dist()

    print("Finished running: dist_util.setup_dist()")
    
    logger.configure()

    logger.log_config(vars(args))

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    if args.dataset=='rplan_structural':
        data = load_rplanhg_structural_data(
            batch_size=args.batch_size,
            analog_bit=args.analog_bit,
            target_set=args.target_set,
            set_name=args.set_name,
        )

        data_val={
            "eval": load_rplanhg_structural_data(
                batch_size=min(args.batch_size, 5),
                analog_bit=args.analog_bit,
                target_set=args.target_set,
                set_name="eval",
            ),
            
            # Takes too much RAM:
            # "train": load_rplanhg_structural_data(
            #     batch_size=min(args.batch_size, 5),
            #     analog_bit=args.analog_bit,
            #     target_set=args.target_set,
            #     set_name="train",
            # )
        }
    
    elif "modified_swiss_dwellings" in args.dataset:
    
        # Disable for modified_swiss_dwellings_without_structural
        use_structural = "without_structural" not in args.dataset

        data = load_modified_swiss_dwellings(
            batch_size=args.batch_size,
            set_name=args.set_name,
            dataset_name=args.dataset,
            use_structural_feats=use_structural
        )

        data_val_no_aug = get_dataloader_modified_swiss_dwellings(
            batch_size=min(args.batch_size, 5), 
            set_name="val",
            override_use_augmentation=None,
            dataset_name=args.dataset,
            use_structural_feats=use_structural
        )

        data_val_train_aug = get_dataloader_modified_swiss_dwellings(
            batch_size=min(args.batch_size, 5), 
            set_name="train",
            override_use_augmentation=True,
            dataset_name=args.dataset,
            use_structural_feats=use_structural
        )

        data_val_aug = get_dataloader_modified_swiss_dwellings(
            batch_size=min(args.batch_size, 5), 
            set_name="val",
            override_use_augmentation=True,
            dataset_name=args.dataset,
            use_structural_feats=use_structural
        )

        data_val = {
            "val_no_aug": data_val_no_aug,
            "val_with_augmentation": data_val_aug,
            "train_with_augmentation": data_val_train_aug,
        }

    else:
        print('dataset not exist!')
        assert False

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        analog_bit=args.analog_bit,
        timeout=args.timeout,
        data_val=data_val,
        test_interval=args.test_interval
    ).run_loop()


def create_argparser():
    defaults = dict(
        dataset = '',
        schedule_sampler= "uniform", #"loss-second-moment", "uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=1000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        struct_in_channels=0,
        timeout = None,
        test_interval=2000,
    )
    parser = argparse.ArgumentParser()
    defaults.update(model_and_diffusion_defaults())
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
