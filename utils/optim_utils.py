import torch
from datasets import load_dataset
from typing import Any, Mapping
import json
import numpy as np
import os
from statistics import mean, stdev


def read_json(filename: str) -> Mapping[str, Any]:
    """Returns a Python dict representation of JSON object at input file."""
    with open(filename) as fp:
        return json.load(fp)


def get_dataset(args):
    if 'laion' in args.dataset_path:
        dataset = load_dataset(args.dataset)['train']
        prompt_key = 'TEXT'
    elif 'coco' in args.dataset_path:
        with open('fid_outputs/coco/meta_data.json') as f:
            dataset = json.load(f)
            dataset = dataset['annotations']
            prompt_key = 'caption'
    else:
        dataset = load_dataset(args.dataset_path)['train']
        prompt_key = 'Prompt'
    return dataset, prompt_key


def save_metrics(args, tpr_detection, tpr_traceability, acc, clip_scores):
    names = {
        'jpeg_ratio': "Jpeg.txt",
        'random_crop_ratio': "RandomCrop.txt",
        'random_drop_ratio': "RandomDrop.txt",
        'gaussian_blur_r': "GauBlur.txt",
        'gaussian_std': "GauNoise.txt",
        'median_blur_k': "MedBlur.txt",
        'resize_ratio': "Resize.txt",
        'sp_prob': "SPNoise.txt",
        'brightness_factor': "Color_Jitter.txt"
    }
    filename = "Identity.txt"
    for option, name in names.items():
        if getattr(args, option) is not None:
            filename = name

    if args.reference_model is not None:
        with open(args.output_path + filename, "a") as file:
            file.write('tpr_detection:' + str(tpr_detection / args.num) + '      ' +
                       'tpr_traceability:' + str(tpr_traceability / args.num) + '      ' +
                       'mean_acc:' + str(mean(acc)) + '      ' + 'std_acc:' + str(stdev(acc)) + '      ' +
                       'mean_clip_score:' + str(mean(clip_scores)) + '      ' + 'std_clip_score:' + str(stdev(clip_scores)) + '      ' +
                       '\n')

    else:
        with open(args.output_path + filename, "a") as file:
            file.write('tpr_detection:' + str(tpr_detection / args.num) + '      ' +
                       'tpr_traceability:' + str(tpr_traceability / args.num) + '      ' +
                       'mean_acc:' + str(mean(acc)) + '      ' + 'std_acc:' + str(stdev(acc)) + '      ' +
                       '\n')

    return


def load_OSI_weights(model, encoder_path, unet_path):
    if encoder_path and os.path.exists(encoder_path):
        print(f"Loading encoder weights from {encoder_path}")
        ckpt = torch.load(encoder_path, map_location='cpu')
        state_dict = {k.replace('module.',''):v for k,v in ckpt.items()}
        
        missing_key, unexpected_key = model.encoder.load_state_dict(state_dict, strict=False)
        print(f'encoder missing key:\n {missing_key}')
        
        missing_key, unexpected_key = model.quant_conv.load_state_dict(state_dict, strict=False)
        print("quant_conv missing keys:", missing_key)

    if unet_path and os.path.exists(unet_path):
        print(f"Loading unet weights from {unet_path}")
        unet_ckpt = torch.load(unet_path, map_location='cpu')
        unet_state_dict = {k.replace('module.',''):v for k,v in unet_ckpt.items()}
        missing_key, unexpected_key = model.unet.load_state_dict(unet_state_dict, strict=False)
        print("unet missing keys:", missing_key)
        print("unet nexpected keys:", unexpected_key)