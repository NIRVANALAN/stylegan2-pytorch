import argparse
import math
import os
from pathlib import Path

import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import lpips
from model import Generator


def prepare_parser():
    usage = 'Parser for all scripts.'
    parser = argparse.ArgumentParser(
        description="Image projector to the generator latent spaces")
    parser.add_argument("--ckpt",
                        type=str,
                        required=True,
                        help="path to the model checkpoint")
    parser.add_argument("--size",
                        type=int,
                        default=256,
                        help="output image sizes of the generator")
    parser.add_argument(
        "--lr_rampup",
        type=float,
        default=0.05,
        help="duration of the learning rate warmup",
    )
    parser.add_argument(
        "--lr_rampdown",
        type=float,
        default=0.25,
        help="duration of the learning rate decay",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--noise",
                        type=float,
                        default=0.05,
                        help="strength of the noise level")
    parser.add_argument(
        "--noise_ramp",
        type=float,
        default=0.75,
        help="duration of the noise level decay",
    )
    parser.add_argument("--step",
                        type=int,
                        default=1000,
                        help="optimize iterations")
    parser.add_argument(
        "--noise_regularize",
        type=float,
        default=1e5,
        help="weight of the noise regularization",
    )
    parser.add_argument("--mse",
                        type=float,
                        default=0,
                        help="weight of the mse loss")
    parser.add_argument(
        "--no_noise_explore",
        action="store_true",
        help="don't add stochastic noise in first three quater",
    )
    parser.add_argument(
        "--normalize_vgg_loss",
        action="store_true",
        help="normalize lpips by input image numbers",
    )
    parser.add_argument(
        "--w_plus",
        action="store_true",
        help="allow to use distinct latent codes to each layers",
    )
    parser.add_argument("files",
                        metavar="FILES",
                        nargs="+",
                        help="path to image files to be projected")

    return parser


def add_parser(parser):
    parser.add_argument(
        "--f1_d",
        type=int,
        default=0,
        help="layers of F1() regularizer",
    )
    parser.add_argument(
        "--grid_search",
        action="store_true",
        help="debug mode",
    )
    parser.add_argument(
        "--id_aware",
        action="store_true",
        help="shared id latent code for all input files",
    )
    parser.add_argument(
        "--inject_index",
        type=int,
        default=None,
        help="injection starts from which layer in W",
    )
    return parser


def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss +
                (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2) +
                (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2))

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latents: list, strength):
    latents_n = []

    for latent in latents:
        noise = torch.randn_like(latent) * strength
        latents_n.append(latent + noise)

    # TODO
    if latents_n[1].shape != latents_n[0].shape:
        latents_n[1] = latents_n[1].expand_as(latents_n[0])  # expand

    return latents_n


def make_image(tensor):
    return (tensor.detach().clamp_(min=-1, max=1).add(1).div_(2).mul(255).type(
        torch.uint8).permute(0, 2, 3, 1).to("cpu").numpy())


if __name__ == "__main__":
    device = "cuda"

    parser = prepare_parser()
    parser = add_parser(parser)
    args = parser.parse_args()

    n_mean_latent = 10000

    resize = min(args.size, 256)

    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(resize),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    imgs = []

    for imgfile in args.files:
        img = transform(Image.open(imgfile).convert("RGB"))
        imgs.append(img)

    imgs = torch.stack(imgs, 0).to(device)

    g_ema = Generator(args.size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)

    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        # latent_out = g_ema.style(noise_sample)
        latent_out = g_ema.style_forward(noise_sample, depth=9 -
                                         args.f1_d)  # used imtermidiate layer

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() /
                      n_mean_latent)**0.5

    percept = lpips.PerceptualLoss(model="net-lin",
                                   net="vgg",
                                   use_gpu=device.startswith("cuda"))

    noises_single = g_ema.make_noise()
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())

    # prepare for latent code(s)
    if args.id_aware:
        # shared id code + independent pose code
        latent_in = latent_mean.detach().clone().unsqueeze(0)
        latent_pose = latent_in.repeat(imgs.shape[0],
                                       1)  # independent pose code
        # latent_id = latent_in.expand(imgs.shape[0], 1)  # shared identity code
        latent_id = latent_in.repeat(1, 1)  # shared identity code

        latent_pose.requires_grad = True
        latent_id.requires_grad = True

        latents = [latent_pose, latent_id]
    else:
        latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(
            imgs.shape[0], 1)
        if args.w_plus:
            latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)

        latent_in.requires_grad = True
        latents = [latent_in]

    for noise in noises:
        noise.requires_grad = True

    # optim
    optimizer = optim.Adam(latents + noises, lr=args.lr)

    # core loop
    pbar = tqdm(range(args.step))
    latent_path = []
    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr
        noise_strength = latent_std * args.noise * max(
            0, 1 - t / args.noise_ramp)**2

        # add stochastic noise to latent_in
        if not args.no_noise_explore:
            latents_n = latent_noise(latents, noise_strength.item())
        else:
            latents_n = latents
        # latent_n = g_ema.style_forward(latent_n, skip=9 - args.f1_d)

        # inference GAN
        assert isinstance(latents_n, (list))
        img_gen, _ = g_ema(latents_n,
                           input_is_latent=True,
                           noise=noises,
                           inject_index=args.inject_index)

        # loss funcs
        batch, channel, height, width = img_gen.shape
        if height > 256:  # resize to 256 for VGG loss computation
            factor = height // 256

            img_gen = img_gen.reshape(batch, channel, height // factor, factor,
                                      width // factor, factor)
            img_gen = img_gen.mean([3, 5])

        p_loss = percept(img_gen, imgs).sum()
        if args.normalize_vgg_loss:
            p_loss /= imgs.shape[0]
        n_loss = noise_regularize(noises)
        mse_loss = F.mse_loss(img_gen, imgs)

        loss = p_loss + args.noise_regularize * n_loss + args.mse * mse_loss

        # step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        noise_normalize_(noises)

        # log
        if (i + 1) % 100 == 0:
            latent_path.append(
                [latent.detach().clone() for latent in latents_n])
        pbar.set_description((
            f"perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};"
            f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"))

    # get last result
    img_gen, _ = g_ema(latent_path[-1],
                       input_is_latent=True,
                       noise=noises,
                       inject_index=args.inject_index)

    # prepare save path
    if args.grid_search:
        # debug mode
        path_base = Path('inversion') / "grid_search" / args.mse / args.noise
    else:
        path_base = Path('inversion')
        if args.id_aware:
            path_base = path_base / 'id_aware'
    if not path_base.exists():
        path_base.mkdir(parents=True)

    # save results

    img_ar = make_image(img_gen)
    result_file = {}

    # import ipdb
    # ipdb.set_trace()

    for i, input_name in enumerate(args.files):
        filename = '{}_inject{}_{}imgs'.format(
            os.path.splitext(os.path.basename(args.files[i]))[0],
            args.inject_index, len(args.files))

        noise_single = []
        for noise in noises:
            noise_single.append(noise[i:i + 1])

        result_file[input_name] = {
            "img": img_gen[i],
            "noise": noise_single,
        }

        if args.id_aware:
            result_file[input_name].update({"latent_pose": latent_pose[i]})
        else:
            result_file[input_name].update({"latent": latent_in[i]})

        img_name = path_base / '{}.png'.format(filename)
        pil_img = Image.fromarray(img_ar[i])
        pil_img.save(img_name)

    if args.id_aware:
        result_file.update({
            "latent_id": latent_id[0],
        })

    torch.save(result_file, path_base / (filename + '.pt'))
