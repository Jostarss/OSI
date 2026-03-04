# OSI: One-step Inversion Excels in Extracting Diffusion Watermarks

<a href="https://arxiv.org/abs/2602.09494"><img src="https://img.shields.io/badge/arXiv-3A98B9?label=%F0%9F%93%9D&labelColor=FFFDD0" style="height: 28px" /></a>


Official implementation of One-step Inversion (OSI) — a fast and accurate method for extracting the initial noise sign from diffusion-generated images, with exceptional performance in Gaussian Shading style watermark extraction.


## 🛠️ Dependencies

This code is based on `python 3.8.20` and the packages specified in `requirements.txt`.

Create and activate a new conda environment:
```bash
conda create -n osi python=3.8.20
conda activate osi
```

Install dependencies:
```bash
pip install -r requirements.txt
```


## 🚗 Usage

1. Prepare the Stable Diffusion weights and the trained OSI model weights, then update paths in the command/script:
   - `--model_path`: Stable Diffusion checkpoint path
   - `--unet_path` / `--encoder_path`: OSI UNet / encoder checkpoints
   - `--dataset_path`: text prompt dataset or local prompts path

2. Run a single distortion example:

```bash
python run_osi.py \
  --num 1000 \
  --image_length 512 \
  --guidance_scale 7.5 \
  --num_inference_steps 50 \
  --channel_copy 1 \
  --hw_copy 8 \
  --fpr 0.000001 \
  --output_path ./output/ \
  --dataset_path <dataset_path> \
  --model_path <stable_diffusion_model_path> \
  --unet_path <osi_unet_ckpt> \
  --encoder_path <osi_encoder_ckpt> \
  --distortion_name <distortion_name>
```

3. Run all distortion presets in one go:

```bash
bash scripts/run_osi.sh
```


## 📚 Acknowledgements

We borrow the code from [Tree-Ring Watermark](https://github.com/YuxinWenRick/tree-ring-watermark.git) and [Gaussian Shading](https://github.com/bsmhmmlf/Gaussian-Shading.git). We appreciate the authors for sharing their code.


## ✏️ Citation

If you find our work helpful, please consider citing:

```bibtex
@article{chen2026osionestepinversionexcels,
      title={OSI: One-step Inversion Excels in Extracting Diffusion Watermarks}, 
      author={Yuwei Chen, Zhenliang He, Jia Tang, Meina Kan, Shiguang Shan},
      journal={arXiv preprint arXiv:2602.09494},
      year={2026}
}
```

