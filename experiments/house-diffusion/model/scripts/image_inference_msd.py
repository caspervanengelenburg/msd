"""
Generate predictions and save them as images.
"""

import argparse
import os
import numpy as np
import pandas as pd

import torch as th

import drawsvg as drawsvg
import matplotlib.pyplot as plt
# from pytorch_fid.fid_score import calculate_fid_given_paths

from PIL import Image


from house_diffusion.respace import SpacedDiffusion


from house_diffusion import dist_util, logger

from house_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    update_arg_parser,
)

from house_diffusion.modified_swiss_dwellings_housediffusion_dataset import get_dataloader_modified_swiss_dwellings, gather_ids

from house_diffusion import modified_swiss_dwellings_housediffusion_dataset

from house_diffusion.transformer import TransformerModel

import pickle

from house_diffusion.plotting.plot_from_feats import plot_from_batch, draw_from_batch


def create_argparser():
    defaults = dict(
        dataset='',
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        # draw_graph=False,
        # save_svg=False,
        # save_edges=False,
        # save_gif=True,
        override_use_augmentation=False,
        path_struct="/path/to/modified-swiss-dwellings-v1-train/struct_in/",
        dataset_name=modified_swiss_dwellings_housediffusion_dataset.DEFAULT_DATASET_PATH,
        gather_all_ids=True,
        save_prefix="inference_msd",
    )

    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def load_model(args):
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    
    model.to(dist_util.dev())
    model.eval()

    return model, diffusion

def load_dataset(args, todo_ids=None):
    return get_dataloader_modified_swiss_dwellings(
        batch_size=args.batch_size,
        set_name=args.set_name,
        override_use_augmentation=args.override_use_augmentation,
        override_shuffle=False,
        dataset_name=args.dataset_name,
        ids_list=todo_ids,
    )


def dict_to_device(dict: dict, device) -> dict:
    return {key: value.to(device) for key, value in dict.items()}

class HouseDiffusionInference:
    def __init__(self, model: TransformerModel, diffusion: SpacedDiffusion, use_ddim=False, clip_denoised=True, analog_bit=False) -> None:
        self.model = model
        self.diffusion = diffusion

        self.clip_denoised = clip_denoised
        self.analog_bit = analog_bit

        # self.sample_fn = (
        #     diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        # )

        assert not use_ddim, "use_ddim is not supported"
        self.sample_fn = diffusion.p_sample_loop
    
    def _sample(self, shape, cond_kwargs) -> dict:
        with th.no_grad():
            cond_kwargs = dict_to_device(cond_kwargs, dist_util.dev())
            
            for key in cond_kwargs:
                cond_kwargs[key] = cond_kwargs[key].cuda()

            sample_dict = self.sample_fn(
                self.model,
                shape,
                clip_denoised=self.clip_denoised,
                model_kwargs=cond_kwargs,
                analog_bit=self.analog_bit,
                return_every_nth=10,
                return_dict=True
            )

            return sample_dict
    
    def sample_with_gt(self, data_sample_gt, model_kwargs):
        sample_dict = self._sample(data_sample_gt.shape, model_kwargs)

        sample = sample_dict["samples"]
        timesteps = sample_dict["timesteps"]

        sample_gt = data_sample_gt.unsqueeze(0)
        
        # Timestep x batch index x num points x 2
        sample = sample.permute([0, 1, 3, 2]).cpu()
        sample_gt = sample_gt.permute([0, 1, 3, 2]).cpu()

        model_kwargs = dict_to_device(model_kwargs, "cpu")

        sample_and_gt = {
            "sample": sample.cpu(),
            "timesteps": timesteps,

            "sample_gt": sample_gt.cpu(),

            "model_kwargs": dict_to_device(model_kwargs, "cpu"),

            "id": model_kwargs["id"],
        }

        return sample_and_gt




def plot_predictions(sample_and_gt, file="inference_sample.png", dpi=100):
    batch_size = sample_and_gt["sample"].shape[1]

    fig, axs = plt.subplots(2, batch_size, figsize=(20, 10), dpi=dpi)

    sample = sample_and_gt["sample"]
    sample_gt = sample_and_gt["sample_gt"]

    model_kwargs = sample_and_gt["model_kwargs"]

    for i in range(batch_size):
        plot_from_batch(sample, model_kwargs, i, ax=axs[0][i])
        plot_from_batch(sample_gt, model_kwargs, i, ax=axs[1][i])

        axs[0][i].set_title(f"prediction {model_kwargs['id'][i]}")
        axs[1][i].set_title(f"ground truth {model_kwargs['id'][i]}")
    
    fig.savefig(file)
    
    plt.close(fig)


def load_structural_img(id, path_struct):

    return np.load(os.path.join(path_struct, f"{id}.npy"))[:, :, 0].astype(np.uint8).T


def save_prediction_images(sample_and_gt, path_struct, submission_folder="submission", submission_gt_folder="submission_gt", submission_with_structure="submission_with_structure"):

    os.makedirs(submission_folder, exist_ok=True)
    os.makedirs(submission_gt_folder, exist_ok=True)
    os.makedirs(submission_with_structure, exist_ok=True)
    
    batch_size = sample_and_gt["sample"].shape[1]

    sample = sample_and_gt["sample"]
    sample_gt = sample_and_gt["sample_gt"]

    model_kwargs = sample_and_gt["model_kwargs"]

    for i in range(batch_size):
        id = model_kwargs["id"][i]

        try:
            pred_i = draw_from_batch(sample, model_kwargs, i, time_step=-1, draw_outline=False, structural_img=None)
            pred_gt_i = draw_from_batch(sample_gt, model_kwargs, i, time_step=-1, draw_outline=False, structural_img=None)

            structural_img = load_structural_img(id, path_struct)

            pred_structure_i = draw_from_batch(sample, model_kwargs, i, time_step=-1, draw_outline=False, structural_img=structural_img)

            # For some reason the images are transposed?
            pred_i = pred_i.T
            pred_gt_i = pred_gt_i.T

            pred_structure_i = pred_structure_i.T

            img = Image.fromarray(pred_i)
            img.save(os.path.join(submission_folder, f'{id}.png'))

            img_pred_gt = Image.fromarray(pred_gt_i)
            img_pred_gt.save(os.path.join(submission_gt_folder, f'{id}.png'))

            img_pred_structure = Image.fromarray(pred_structure_i)
            img_pred_structure.save(os.path.join(submission_with_structure, f'{id}.png'))
        except Exception as e:
            print(f"failed to save {id}: {e}")

    

def main():
    args = create_argparser().parse_args()
    update_arg_parser(args)

    if args.analog_bit:
        raise NotImplementedError("Analog bit should be false")


    dist_util.setup_dist()
    logger.configure(log_suffix="_inference")

    logger.log_config(vars(args))

    logger.log("creating model and diffusion...")
    
    model, diffusion = load_model(args)

    inference = HouseDiffusionInference(model, diffusion, use_ddim=args.use_ddim, clip_denoised=args.clip_denoised, analog_bit=args.analog_bit)

    

    save_prefix = f"{args.save_prefix}/{args.dataset_name}_{args.set_name}"
    os.makedirs(save_prefix, exist_ok=True)

    submission_folder = f"{save_prefix}/submission"
    submission_gt_folder = f"{save_prefix}/submission_gt"
    submission_with_structure = f"{save_prefix}/submission_with_structure"

    if args.gather_all_ids:
        all_ids = gather_ids(f"../datasets/{args.dataset_name}/house_dicts")
    else:
        ids_csv = f"../datasets/{args.dataset_name}/{args.set_name}_ids.csv"

        all_ids = pd.read_csv(ids_csv, header=None).values.flatten().tolist()    

    if os.path.exists(submission_with_structure):
        done_ids = gather_ids(submission_with_structure)

        todo_ids = sorted(set(all_ids) - set(done_ids))
    else:
        todo_ids = all_ids

    data = load_dataset(args, todo_ids)


    for i, (data_sample_gt, model_kwargs) in enumerate(data):
        #  = next(data)

        assert "id" in model_kwargs

        # sample_and_gt contains the predicted corner locations of a batch.
        sample_and_gt = inference.sample_with_gt(data_sample_gt, model_kwargs)


        # Save the predictions from the batch to a pickle file. Useful to experiment with post-processing later without having to re-run the inference.
        with open(f"{save_prefix}/sample_and_gt_{i}.pkl", "wb") as f:
            pickle.dump(sample_and_gt, f)
        
        # batch_plot_image = f"{save_prefix}/inference_sample_{i}.png"
        # plot_predictions(sample_and_gt, file=batch_plot_image, dpi=300)
        # logger.log_image("inference_sample", batch_plot_image)


        save_prediction_images(sample_and_gt, path_struct=args.path_struct, submission_folder=submission_folder, submission_gt_folder=submission_gt_folder, submission_with_structure=submission_with_structure)



if __name__ == "__main__":
    main()
