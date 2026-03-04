MODEL_PATH="path/to/stable_diffusion_model/"
DATASET_PATH="path/to/dataset/"
UNET_PATH="path/to/osi_unet/"
ENCODER_PATH="path/to/osi_encoder/"

python run_osi.py \
    --num 1000 \
    --image_length 512 \
    --guidance_scale 7.5 \
    --num_inference_steps 50 \
    --channel_copy 1 \
    --hw_copy 8 \
    --fpr 0.000001 \
    --output_path ./output/ \
    --dataset_path ${DATASET_PATH} \
    --model_path ${MODEL_PATH} \
    --unet_path ${UNET_PATH} \
    --encoder_path ${ENCODER_PATH} \
    --distortion_name Identity

python run_osi.py \
    --num 1000 \
    --image_length 512 \
    --guidance_scale 7.5 \
    --num_inference_steps 50 \
    --channel_copy 1 \
    --hw_copy 8 \
    --fpr 0.000001 \
    --output_path ./output/ \
    --dataset_path ${DATASET_PATH} \
    --model_path ${MODEL_PATH} \
    --unet_path ${UNET_PATH} \
    --encoder_path ${ENCODER_PATH} \
    --distortion_name RandomCrop \
    --random_crop_ratio 0.6

python run_osi.py \
    --num 1000 \
    --image_length 512 \
    --guidance_scale 7.5 \
    --num_inference_steps 50 \
    --channel_copy 1 \
    --hw_copy 8 \
    --fpr 0.000001 \
    --output_path ./output/ \
    --dataset_path ${DATASET_PATH} \
    --model_path ${MODEL_PATH} \
    --unet_path ${UNET_PATH} \
    --encoder_path ${ENCODER_PATH} \
    --distortion_name RandomDrop \
    --random_drop_ratio 0.8

python run_osi.py \
    --num 1000 \
    --image_length 512 \
    --guidance_scale 7.5 \
    --num_inference_steps 50 \
    --channel_copy 1 \
    --hw_copy 8 \
    --fpr 0.000001 \
    --output_path ./output/ \
    --dataset_path ${DATASET_PATH} \
    --model_path ${MODEL_PATH} \
    --unet_path ${UNET_PATH} \
    --encoder_path ${ENCODER_PATH} \
    --distortion_name Color_Jitter \
    --brightness_factor 6

python run_osi.py \
    --num 1000 \
    --image_length 512 \
    --guidance_scale 7.5 \
    --num_inference_steps 50 \
    --channel_copy 1 \
    --hw_copy 8 \
    --fpr 0.000001 \
    --output_path ./output/ \
    --dataset_path ${DATASET_PATH} \
    --model_path ${MODEL_PATH} \
    --unet_path ${UNET_PATH} \
    --encoder_path ${ENCODER_PATH} \
    --distortion_name Jpeg \
    --jpeg_ratio 25

python run_osi.py \
    --num 1000 \
    --image_length 512 \
    --guidance_scale 7.5 \
    --num_inference_steps 50 \
    --channel_copy 1 \
    --hw_copy 8 \
    --fpr 0.000001 \
    --output_path ./output/ \
    --dataset_path ${DATASET_PATH} \
    --model_path ${MODEL_PATH} \
    --unet_path ${UNET_PATH} \
    --encoder_path ${ENCODER_PATH} \
    --distortion_name GauBlur \
    --gaussian_blur_r 4

python run_osi.py \
    --num 1000 \
    --image_length 512 \
    --guidance_scale 7.5 \
    --num_inference_steps 50 \
    --channel_copy 1 \
    --hw_copy 8 \
    --fpr 0.000001 \
    --output_path ./output/ \
    --dataset_path ${DATASET_PATH} \
    --model_path ${MODEL_PATH} \
    --unet_path ${UNET_PATH} \
    --encoder_path ${ENCODER_PATH} \
    --distortion_name GauNoise \
    --gaussian_std 0.05

python run_osi.py \
    --num 1000 \
    --image_length 512 \
    --guidance_scale 7.5 \
    --num_inference_steps 50 \
    --channel_copy 1 \
    --hw_copy 8 \
    --fpr 0.000001 \
    --output_path ./output/ \
    --dataset_path ${DATASET_PATH} \
    --model_path ${MODEL_PATH} \
    --unet_path ${UNET_PATH} \
    --encoder_path ${ENCODER_PATH} \
    --distortion_name MedBlur \
    --median_blur_k 7

python run_osi.py \
    --num 1000 \
    --image_length 512 \
    --guidance_scale 7.5 \
    --num_inference_steps 50 \
    --channel_copy 1 \
    --hw_copy 8 \
    --fpr 0.000001 \
    --output_path ./output/ \
    --dataset_path ${DATASET_PATH} \
    --model_path ${MODEL_PATH} \
    --unet_path ${UNET_PATH} \
    --encoder_path ${ENCODER_PATH} \
    --distortion_name SPNoise \
    --sp_prob 0.05

python run_osi.py \
    --num 1000 \
    --image_length 512 \
    --guidance_scale 7.5 \
    --num_inference_steps 50 \
    --channel_copy 1 \
    --hw_copy 8 \
    --fpr 0.000001 \
    --output_path ./output/ \
    --dataset_path ${DATASET_PATH} \
    --model_path ${MODEL_PATH} \
    --unet_path ${UNET_PATH} \
    --encoder_path ${ENCODER_PATH} \
    --distortion_name Resize \
    --resize_ratio 0.25
