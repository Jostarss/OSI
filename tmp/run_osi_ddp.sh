#!/usr/bin/env bash

NPROC_PER_NODE=${NPROC_PER_NODE:-8}

torchrun --nproc_per_node="${NPROC_PER_NODE}" run_osi_ddp.py \
    --num 1000 \
    --image_length 512 \
    --guidance_scale 7.5 \
    --num_inference_steps 50 \
    --channel_copy 1 \
    --hw_copy 1 \
    --fpr 0.000001 \
    --output_path ./output/ddp/ \
    --dataset_path /data/hezhenliang/users/tangjia/.cache/huggingface/datasets/Gustavosta___stable-diffusion-prompts/default/0.0.0/d816d4a05cb89bde39dd99284c459801e1e7e69a \
    --model_path /data/hezhenliang/users/tangjia/workspace/R2/model_weights/stable-diffusion-2-1-base/snapshots/1f758383196d38df1dfe523ddb1030f2bfab7741 \
    --unet_path /data/hezhenliang/users/tangjia/workspace/R2/model_weights/OSI/SD21_ep10_unet.pth \
    --encoder_path /data/hezhenliang/users/tangjia/workspace/R2/model_weights/OSI/SD21_ep10_encoder.pth \
    --distortion_name Identity

torchrun --nproc_per_node="${NPROC_PER_NODE}" run_osi_ddp.py \
    --num 1000 \
    --image_length 512 \
    --guidance_scale 7.5 \
    --num_inference_steps 50 \
    --channel_copy 1 \
    --hw_copy 1 \
    --fpr 0.000001 \
    --output_path ./output/ddp/ \
    --dataset_path /data/hezhenliang/users/tangjia/.cache/huggingface/datasets/Gustavosta___stable-diffusion-prompts/default/0.0.0/d816d4a05cb89bde39dd99284c459801e1e7e69a \
    --model_path /data/hezhenliang/users/tangjia/workspace/R2/model_weights/stable-diffusion-2-1-base/snapshots/1f758383196d38df1dfe523ddb1030f2bfab7741 \
    --unet_path /data/hezhenliang/users/tangjia/workspace/R2/model_weights/OSI/SD21_ep10_unet.pth \
    --encoder_path /data/hezhenliang/users/tangjia/workspace/R2/model_weights/OSI/SD21_ep10_encoder.pth \
    --distortion_name RandomCrop \
    --random_crop_ratio 0.6

torchrun --nproc_per_node="${NPROC_PER_NODE}" run_osi_ddp.py \
    --num 1000 \
    --image_length 512 \
    --guidance_scale 7.5 \
    --num_inference_steps 50 \
    --channel_copy 1 \
    --hw_copy 1 \
    --fpr 0.000001 \
    --output_path ./output/ddp/ \
    --dataset_path /data/hezhenliang/users/tangjia/.cache/huggingface/datasets/Gustavosta___stable-diffusion-prompts/default/0.0.0/d816d4a05cb89bde39dd99284c459801e1e7e69a \
    --model_path /data/hezhenliang/users/tangjia/workspace/R2/model_weights/stable-diffusion-2-1-base/snapshots/1f758383196d38df1dfe523ddb1030f2bfab7741 \
    --unet_path /data/hezhenliang/users/tangjia/workspace/R2/model_weights/OSI/SD21_ep10_unet.pth \
    --encoder_path /data/hezhenliang/users/tangjia/workspace/R2/model_weights/OSI/SD21_ep10_encoder.pth \
    --distortion_name RandomDrop \
    --random_drop_ratio 0.8

torchrun --nproc_per_node="${NPROC_PER_NODE}" run_osi_ddp.py \
    --num 1000 \
    --image_length 512 \
    --guidance_scale 7.5 \
    --num_inference_steps 50 \
    --channel_copy 1 \
    --hw_copy 1 \
    --fpr 0.000001 \
    --output_path ./output/ddp/ \
    --dataset_path /data/hezhenliang/users/tangjia/.cache/huggingface/datasets/Gustavosta___stable-diffusion-prompts/default/0.0.0/d816d4a05cb89bde39dd99284c459801e1e7e69a \
    --model_path /data/hezhenliang/users/tangjia/workspace/R2/model_weights/stable-diffusion-2-1-base/snapshots/1f758383196d38df1dfe523ddb1030f2bfab7741 \
    --unet_path /data/hezhenliang/users/tangjia/workspace/R2/model_weights/OSI/SD21_ep10_unet.pth \
    --encoder_path /data/hezhenliang/users/tangjia/workspace/R2/model_weights/OSI/SD21_ep10_encoder.pth \
    --distortion_name Color_Jitter \
    --brightness_factor 6

torchrun --nproc_per_node="${NPROC_PER_NODE}" run_osi_ddp.py \
    --num 1000 \
    --image_length 512 \
    --guidance_scale 7.5 \
    --num_inference_steps 50 \
    --channel_copy 1 \
    --hw_copy 1 \
    --fpr 0.000001 \
    --output_path ./output/ddp/ \
    --dataset_path /data/hezhenliang/users/tangjia/.cache/huggingface/datasets/Gustavosta___stable-diffusion-prompts/default/0.0.0/d816d4a05cb89bde39dd99284c459801e1e7e69a \
    --model_path /data/hezhenliang/users/tangjia/workspace/R2/model_weights/stable-diffusion-2-1-base/snapshots/1f758383196d38df1dfe523ddb1030f2bfab7741 \
    --unet_path /data/hezhenliang/users/tangjia/workspace/R2/model_weights/OSI/SD21_ep10_unet.pth \
    --encoder_path /data/hezhenliang/users/tangjia/workspace/R2/model_weights/OSI/SD21_ep10_encoder.pth \
    --distortion_name Jpeg \
    --jpeg_ratio 25

torchrun --nproc_per_node="${NPROC_PER_NODE}" run_osi_ddp.py \
    --num 1000 \
    --image_length 512 \
    --guidance_scale 7.5 \
    --num_inference_steps 50 \
    --channel_copy 1 \
    --hw_copy 1 \
    --fpr 0.000001 \
    --output_path ./output/ddp/ \
    --dataset_path /data/hezhenliang/users/tangjia/.cache/huggingface/datasets/Gustavosta___stable-diffusion-prompts/default/0.0.0/d816d4a05cb89bde39dd99284c459801e1e7e69a \
    --model_path /data/hezhenliang/users/tangjia/workspace/R2/model_weights/stable-diffusion-2-1-base/snapshots/1f758383196d38df1dfe523ddb1030f2bfab7741 \
    --unet_path /data/hezhenliang/users/tangjia/workspace/R2/model_weights/OSI/SD21_ep10_unet.pth \
    --encoder_path /data/hezhenliang/users/tangjia/workspace/R2/model_weights/OSI/SD21_ep10_encoder.pth \
    --distortion_name GauBlur \
    --gaussian_blur_r 4

torchrun --nproc_per_node="${NPROC_PER_NODE}" run_osi_ddp.py \
    --num 1000 \
    --image_length 512 \
    --guidance_scale 7.5 \
    --num_inference_steps 50 \
    --channel_copy 1 \
    --hw_copy 1 \
    --fpr 0.000001 \
    --output_path ./output/ddp/ \
    --dataset_path /data/hezhenliang/users/tangjia/.cache/huggingface/datasets/Gustavosta___stable-diffusion-prompts/default/0.0.0/d816d4a05cb89bde39dd99284c459801e1e7e69a \
    --model_path /data/hezhenliang/users/tangjia/workspace/R2/model_weights/stable-diffusion-2-1-base/snapshots/1f758383196d38df1dfe523ddb1030f2bfab7741 \
    --unet_path /data/hezhenliang/users/tangjia/workspace/R2/model_weights/OSI/SD21_ep10_unet.pth \
    --encoder_path /data/hezhenliang/users/tangjia/workspace/R2/model_weights/OSI/SD21_ep10_encoder.pth \
    --distortion_name GauNoise \
    --gaussian_std 0.05

torchrun --nproc_per_node="${NPROC_PER_NODE}" run_osi_ddp.py \
    --num 1000 \
    --image_length 512 \
    --guidance_scale 7.5 \
    --num_inference_steps 50 \
    --channel_copy 1 \
    --hw_copy 1 \
    --fpr 0.000001 \
    --output_path ./output/ddp/ \
    --dataset_path /data/hezhenliang/users/tangjia/.cache/huggingface/datasets/Gustavosta___stable-diffusion-prompts/default/0.0.0/d816d4a05cb89bde39dd99284c459801e1e7e69a \
    --model_path /data/hezhenliang/users/tangjia/workspace/R2/model_weights/stable-diffusion-2-1-base/snapshots/1f758383196d38df1dfe523ddb1030f2bfab7741 \
    --unet_path /data/hezhenliang/users/tangjia/workspace/R2/model_weights/OSI/SD21_ep10_unet.pth \
    --encoder_path /data/hezhenliang/users/tangjia/workspace/R2/model_weights/OSI/SD21_ep10_encoder.pth \
    --distortion_name MedBlur \
    --median_blur_k 7

torchrun --nproc_per_node="${NPROC_PER_NODE}" run_osi_ddp.py \
    --num 1000 \
    --image_length 512 \
    --guidance_scale 7.5 \
    --num_inference_steps 50 \
    --channel_copy 1 \
    --hw_copy 1 \
    --fpr 0.000001 \
    --output_path ./output/ddp/ \
    --dataset_path /data/hezhenliang/users/tangjia/.cache/huggingface/datasets/Gustavosta___stable-diffusion-prompts/default/0.0.0/d816d4a05cb89bde39dd99284c459801e1e7e69a \
    --model_path /data/hezhenliang/users/tangjia/workspace/R2/model_weights/stable-diffusion-2-1-base/snapshots/1f758383196d38df1dfe523ddb1030f2bfab7741 \
    --unet_path /data/hezhenliang/users/tangjia/workspace/R2/model_weights/OSI/SD21_ep10_unet.pth \
    --encoder_path /data/hezhenliang/users/tangjia/workspace/R2/model_weights/OSI/SD21_ep10_encoder.pth \
    --distortion_name SPNoise \
    --sp_prob 0.05

torchrun --nproc_per_node="${NPROC_PER_NODE}" run_osi_ddp.py \
    --num 1000 \
    --image_length 512 \
    --guidance_scale 7.5 \
    --num_inference_steps 50 \
    --channel_copy 1 \
    --hw_copy 1 \
    --fpr 0.000001 \
    --output_path ./output/ddp/ \
    --dataset_path /data/hezhenliang/users/tangjia/.cache/huggingface/datasets/Gustavosta___stable-diffusion-prompts/default/0.0.0/d816d4a05cb89bde39dd99284c459801e1e7e69a \
    --model_path /data/hezhenliang/users/tangjia/workspace/R2/model_weights/stable-diffusion-2-1-base/snapshots/1f758383196d38df1dfe523ddb1030f2bfab7741 \
    --unet_path /data/hezhenliang/users/tangjia/workspace/R2/model_weights/OSI/SD21_ep10_unet.pth \
    --encoder_path /data/hezhenliang/users/tangjia/workspace/R2/model_weights/OSI/SD21_ep10_encoder.pth \
    --distortion_name Resize \
    --resize_ratio 0.25
