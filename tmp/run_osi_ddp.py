import argparse
import copy
from tqdm import tqdm
import torch
import torch.distributed as dist
import os
import csv
from typing import Union, Sequence, List
from PIL import Image

from transformers import CLIPModel, CLIPTokenizer
from inversion.inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler
import open_clip

from utils.optim_utils import *
from utils.io_utils import *
from utils.image_utils import *
from watermark import *


class OSIModel(torch.nn.Module):
    def __init__(self, unet, encoder, quant_conv, vae_scaling_factor):
        super().__init__()
        self.unet = unet
        self.encoder = encoder
        self.quant_conv = quant_conv
        self.vae_scaling_factor = vae_scaling_factor

    def forward(self, latents, timestep, prompt_embeds):
        inv_x0 = self.quant_conv(self.encoder(latents))[:, :4] * self.vae_scaling_factor
        noise_pred = self.unet(
            inv_x0,
            timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]

        return inv_x0, noise_pred


def setup_distributed():
    use_ddp = (
        torch.cuda.is_available()
        and "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
        and int(os.environ["WORLD_SIZE"]) > 1
    )
    if not use_ddp:
        return False, 0, 1, 0

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return True, rank, world_size, local_rank


def cleanup_distributed(is_ddp):
    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()


def main(args):
    is_ddp, rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    is_main_process = rank == 0

    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_path, subfolder="scheduler")
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_path,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision="fp16",
    )
    pipe.safety_checker = None
    pipe = pipe.to(device)

    # load OSI model
    OSI_model = OSIModel(
        unet=copy.deepcopy(pipe.unet).to(device),
        encoder=copy.deepcopy(pipe.vae.encoder).to(device),
        quant_conv=copy.deepcopy(pipe.vae.quant_conv).to(device),
        vae_scaling_factor=0.18215,
    )
    load_OSI_weights(OSI_model, args.encoder_path, args.unet_path)

    # assume at the detection time, the original prompt is unknown
    tester_prompt = ""
    text_embeddings = pipe.get_text_embedding(tester_prompt)
    timestep = torch.tensor([999], device=device)

    # reference model for CLIP score
    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(
            args.reference_model,
            pretrained=args.reference_model_pretrain,
            device=device,
        )
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    # dataset
    dataset, prompt_key = get_dataset(args)

    # class for watermark
    if args.chacha:
        watermark = Gaussian_Shading_chacha(args.channel_copy, args.hw_copy, args.fpr, args.user_number)
    else:
        watermark = Gaussian_Shading(args.channel_copy, args.hw_copy, args.fpr, args.user_number)

    os.makedirs(args.output_path, exist_ok=True)
    if args.save_image:
        img_root = os.path.join(args.output_path, "image")
        os.makedirs(img_root, exist_ok=True)

    # assume at the detection time, the original prompt is unknown
    tester_prompt = ""
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    # acc
    acc = []
    # CLIP Scores
    clip_scores = []
    local_rows = []

    sample_indices = list(range(rank, args.num, world_size))
    iterator = tqdm(sample_indices) if is_main_process else sample_indices
    for i in iterator:
        seed = i + args.gen_seed
        current_prompt = dataset[i][prompt_key]

        # generate with watermark
        set_random_seed(seed)
        if args.hw_copy > 1:
            init_latents_w = watermark.create_watermark_and_return_w_XOR()
        else:
            init_latents_w = watermark.create_watermark_and_return_w_IDENTITY()

        outputs = pipe(
            current_prompt,
            num_images_per_prompt=1,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_w,
        )
        image_w = outputs.images[0]

        # distortion
        image_w_distortion = image_distortion(image_w, seed, args)
        if args.save_image:
            dst = os.path.join(img_root, str(i))
            os.makedirs(dst, exist_ok=True)
            image_w_distortion.save(os.path.join(dst, f"{args.distortion_name}.png"))

        # reverse image with OSI model
        image_w_distortion = transform_img(image_w_distortion).unsqueeze(0).to(text_embeddings.dtype).to(device)
        with torch.no_grad():
            _, reversed_latents_w = OSI_model(
                latents=image_w_distortion,
                timestep=timestep,
                prompt_embeds=text_embeddings,
            )
            predict_wm = (torch.sigmoid(reversed_latents_w) >= 0.5).int()

        # acc metric
        acc_metric = watermark.eval_watermark(predict_wm)
        acc.append(acc_metric)

        # CLIP Score
        if args.reference_model is not None:
            socre = measure_similarity(
                [image_w],
                current_prompt,
                ref_model,
                ref_clip_preprocess,
                ref_tokenizer,
                device,
            )
            clip_socre = socre[0].item()
        else:
            clip_socre = 0
        clip_scores.append(clip_socre)
        local_rows.append({"NO": i, args.distortion_name: acc_metric})

    tpr_detection, tpr_traceability = watermark.get_tpr()

    if is_ddp:
        gathered_acc = [None for _ in range(world_size)]
        gathered_clip_scores = [None for _ in range(world_size)]
        gathered_rows = [None for _ in range(world_size)]
        gathered_tpr_detection = [None for _ in range(world_size)]
        gathered_tpr_traceability = [None for _ in range(world_size)]

        dist.all_gather_object(gathered_acc, acc)
        dist.all_gather_object(gathered_clip_scores, clip_scores)
        dist.all_gather_object(gathered_rows, local_rows)
        dist.all_gather_object(gathered_tpr_detection, tpr_detection)
        dist.all_gather_object(gathered_tpr_traceability, tpr_traceability)

        if is_main_process:
            all_acc = [x for part in gathered_acc for x in part]
            all_clip_scores = [x for part in gathered_clip_scores for x in part]
            all_rows = [x for part in gathered_rows for x in part]
            all_rows = sorted(all_rows, key=lambda x: x["NO"])

            total_tpr_detection = sum(gathered_tpr_detection)
            total_tpr_traceability = sum(gathered_tpr_traceability)

            csv_path = os.path.join(args.output_path, f"{args.distortion_name}.csv")
            with open(csv_path, "w", newline="") as csv_file:
                fieldnames = ["NO", args.distortion_name]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)

            save_metrics(args, total_tpr_detection, total_tpr_traceability, all_acc, all_clip_scores)
    else:
        csv_path = os.path.join(args.output_path, f"{args.distortion_name}.csv")
        with open(csv_path, "w", newline="") as csv_file:
            fieldnames = ["NO", args.distortion_name]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(local_rows)

        save_metrics(args, tpr_detection, tpr_traceability, acc, clip_scores)

    cleanup_distributed(is_ddp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One Step Inversion (OSI) DDP")
    parser.add_argument("--num", default=1000, type=int)
    parser.add_argument("--image_length", default=512, type=int)
    parser.add_argument("--guidance_scale", default=7.5, type=float)
    parser.add_argument("--num_inference_steps", default=50, type=int)
    parser.add_argument("--num_inversion_steps", default=None, type=int)
    parser.add_argument("--gen_seed", default=0, type=int)
    parser.add_argument("--channel_copy", default=1, type=int)
    parser.add_argument("--hw_copy", default=8, type=int)
    parser.add_argument("--user_number", default=1000000, type=int)
    parser.add_argument("--fpr", default=0.000001, type=float)
    parser.add_argument("--output_path", default="./output/")
    parser.add_argument("--chacha", action="store_true", help="chacha20 for cipher")
    parser.add_argument("--reference_model", default=None)
    parser.add_argument("--reference_model_pretrain", default=None)
    parser.add_argument("--dataset_path", default="Gustavosta/Stable-Diffusion-Prompts")
    parser.add_argument("--model_path", default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--save_image", action="store_true", help="save generated images")

    # distortion
    parser.add_argument("--jpeg_ratio", default=None, type=int)
    parser.add_argument("--random_crop_ratio", default=None, type=float)
    parser.add_argument("--random_drop_ratio", default=None, type=float)
    parser.add_argument("--gaussian_blur_r", default=None, type=int)
    parser.add_argument("--median_blur_k", default=None, type=int)
    parser.add_argument("--resize_ratio", default=None, type=float)
    parser.add_argument("--gaussian_std", default=None, type=float)
    parser.add_argument("--sp_prob", default=None, type=float)
    parser.add_argument("--brightness_factor", default=None, type=float)
    parser.add_argument("--distortion_name", default="Identity", type=str)

    # osi model weights
    parser.add_argument("--unet_path", default=None, type=str)
    parser.add_argument("--encoder_path", default=None, type=str)

    args = parser.parse_args()

    if args.num_inversion_steps is None:
        args.num_inversion_steps = args.num_inference_steps

    main(args)
