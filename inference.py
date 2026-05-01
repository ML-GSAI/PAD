import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hy3dshape'))
import torch
import yaml
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.utils.depth.depth_converter import ImageToVAE
from hy3dshape.utils import instantiate_from_config


def load_pipeline(base_model_path, ckpt_cfg_path=None, ckpt_path=None):
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(base_model_path)

    if ckpt_cfg_path and ckpt_path:
        config = yaml.safe_load(open(ckpt_cfg_path))
        model = instantiate_from_config(config['model']['params']['denoiser_cfg'])
        sd = torch.load(ckpt_path, map_location='cpu')
        sd = {k.replace('_forward_module.model.', ''): v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if 'first_stage_model' not in k}
        model.load_state_dict(sd)
        pipeline.model = model.cuda().half()

    return pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to input image or directory of images')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--base_model', default='tencent/Hunyuan3D-2.1', help='Base model path or HuggingFace ID')
    parser.add_argument('--ckpt_config', default=None, help='Path to finetuned model config YAML')
    parser.add_argument('--ckpt', default=None, help='Path to finetuned checkpoint (pytorch_model.bin)')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=5.0)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    pipeline = load_pipeline(args.base_model, args.ckpt_config, args.ckpt)
    image_to_vae = ImageToVAE()

    # Collect images
    if os.path.isdir(args.image):
        exts = ('.png', '.jpg', '.jpeg')
        image_paths = [os.path.join(args.image, f) for f in os.listdir(args.image) if f.lower().endswith(exts)]
    else:
        image_paths = [args.image]

    for image_path in image_paths:
        print(f'Processing {image_path}')
        pts = image_to_vae(image_path)
        pts = pts.to('cuda').half()
        pts = pipeline.vae.encode(pts, sample_posterior=True)

        mesh = pipeline(image=image_path, pts=pts, num_inference_steps=args.steps, guidance_scale=args.guidance_scale)[0]

        stem = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(args.output, f'{stem}.glb')
        mesh.export(out_path)
        print(f'Saved to {out_path}')


if __name__ == '__main__':
    main()
