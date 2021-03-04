import argparse
import math
import random

import torch
import numpy as np
from torch.serialization import save
from torchvision import utils
from model import Generator
from tqdm import tqdm, trange

from pathlib import Path


def calculate_statistics(args, g_ema, n_mean_latent=10000, device='cuda'):
    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        # latent_out = g_ema.style(noise_sample)
        latent_out = g_ema.style_forward(noise_sample, depth=8 -
                                         args.f1_d)  # used imtermidiate layer

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() /
                      n_mean_latent)**0.5
        torch.save(
            latent_mean,
            Path(args.ckpt).parent /
            'latent_mean_layer{}.pt'.format(8 - args.f1_d))
        torch.save(
            latent_std,
            Path(args.ckpt).parent /
            'latent_std_layer{}.pt'.format(8 - args.f1_d))
        return latent_mean, latent_std


def generate(args,
             g_ema,
             device,
             mean_latent,
             init_latent=None,
             random_sample=True,
             latent_code=None,
             noise=None,
             name=None,
             inject_index=4):

    if not random_sample:
        # generate given code
        sample, _ = g_ema(
            latent_code,
            inject_index=inject_index,
            input_is_latent=True,
            truncation=args.truncation,
            truncation_latent=mean_latent,
            noise=noise,
        )
        utils.save_image(
            sample,
            f"sample/{name}.png",
            normalize=True,
            range=(-1, 1),
        )
    else:
        stds = torch.linspace(0, 5, args.sample).view(args.sample,
                                                      1).to('cuda')
        if init_latent is not None:
            if isinstance(init_latent, list):
                init_latent = [
                    latent.repeat(args.sample, 1) for latent in init_latent
                ]
            else:
                init_latent = init_latent.repeat(args.sample, 1)

        with torch.no_grad():
            g_ema.eval()
            for i in tqdm(range(args.pics)):
                if init_latent == None:
                    sample_z = torch.randn(args.sample,
                                           args.latent,
                                           device=device)
                else:
                    # add jitter noise into H space and style_forward into W space
                    if not isinstance(init_latent, (list)):
                        z_rand = torch.zeros_like(init_latent,
                                                  device=device).normal_()
                        sample_z = [init_latent + z_rand * stds]
                    else:
                        # cross-over inject
                        z_rand = torch.zeros_like(init_latent[0],
                                                  device=device).normal_()
                        sample_z = [
                            init_latent[0] + z_rand * stds, init_latent[1]
                        ]
                        # import ipdb
                        # ipdb.set_trace()

                    # sample_z = g_ema.style_forward(sample_z,
                    #                                skip=8 - args.f1_d)

                sample, _ = g_ema(sample_z,
                                  truncation=args.truncation,
                                  truncation_latent=mean_latent,
                                  inject_index=args.inject_index,
                                  input_is_latent=True)

                utils.save_image(
                    sample,
                    f"sample/{str(i).zfill(6)}.png",
                    nrow=math.floor(args.sample**0.5),
                    normalize=True,
                    range=(-1, 1),
                )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(
        description="Generate samples from the generator")
    parser.add_argument(
        "--f1_d",
        type=int,
        default=2,
        help="layers of F1() regularizer",
    )
    parser.add_argument("--size",
                        type=int,
                        default=1024,
                        help="output image size of the generator")
    parser.add_argument(
        "--inject_index",
        type=int,
        default=4,
        help="which layer to cross-over",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=16,
        help="number of samples to be generated for each image",
    )
    parser.add_argument("--pics",
                        type=int,
                        default=8,
                        help="number of images to be generated")
    parser.add_argument("--truncation",
                        type=float,
                        default=1,
                        help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--proj_latent",
        type=str,
        default=None,
        help="path to the projected latent code",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--id_aware",
        action="store_true",
        help="shared id latent code for all input files",
    )

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(args.size,
                      args.latent,
                      args.n_mlp,
                      channel_multiplier=args.channel_multiplier).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    # load statistics
    try:
        latent_mean = torch.load(
            Path(args.ckpt).parent /
            'latent_mean_layer{}.pt'.format(8 - args.f1_d))
    except FileNotFoundError:
        latent_mean, _ = calculate_statistics(args, g_ema)

    saved_result = torch.load(args.proj_latent)
    # saved_result = torch.load('inversion/regularized/009990.pt')
    if args.id_aware:
        latent_code = [
            saved_result[i]['latent'].unsqueeze(0)
            for i in saved_result.keys()
        ]
    else:
        img_keys = list(saved_result.keys())[:-1]

        # import ipdb
        # ipdb.set_trace()

        pose_codes = [
            saved_result[img_name]['latent_pose'] for img_name in img_keys
        ]
        pose_code = (pose_codes[0] + pose_codes[4]) / 2
        id_code = saved_result['latent_id'].squeeze()

        latent_code = [latent.unsqueeze(0) for latent in [pose_code, id_code]]

    # latent_code.reverse()

    for i in trange(1):
        generate(
            args,
            g_ema,
            device,
            mean_latent,
            init_latent=latent_code,
            # latent_code=latent_code,
            random_sample=True,
            #  noise=saved_result['noise']
            name=Path(args.proj_latent).name +
            f'_inject{args.inject_index}_{i}',
            inject_index=args.inject_index)
        # latent_code.reverse()
