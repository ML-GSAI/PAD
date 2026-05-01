import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import itertools
import warnings
import json
from typing import Optional
import click

from moge.model import import_model_class_by_version
from moge.utils.io import save_glb, save_ply
from moge.utils.vis import colorize_depth, colorize_normal
from moge.utils.geometry_numpy import depth_occlusion_edge_numpy
import utils3d




class MogeDepth:
    """MoGe depth estimation wrapper class."""
    def __init__(self, version: str = 'v2', device: str = 'cuda', use_fp16: bool = False, pretrained_model_path: Optional[str] = None):
        self.device = torch.device(device)
        self.use_fp16 = use_fp16
        
        DEFAULT_PRETRAINED_MODEL_FOR_EACH_VERSION = {
            "v1": "Ruicheng/moge-vitl",
            "v2": "Ruicheng/moge-2-vitl-normal",
        }
        
        if pretrained_model_path is None:
             pretrained_model_path = DEFAULT_PRETRAINED_MODEL_FOR_EACH_VERSION.get(version, DEFAULT_PRETRAINED_MODEL_FOR_EACH_VERSION['v2'])
             
        self.model = import_model_class_by_version(version).from_pretrained(pretrained_model_path).to(self.device).eval()
        if self.use_fp16:
            self.model.half()

    def predict(self, image: np.ndarray, fov_x: Optional[float] = None, resolution_level: int = 30, num_tokens: int|None = None) -> dict:
        """
        Input: RGB Image (H, W, 3) in range [0, 1]
        Output: Dictionary containing depth, points, mask, intrinisics, normal (if available)
        """
        height, width = image.shape[:2]
        image_tensor = torch.tensor(image, dtype=torch.float32, device=self.device).permute(2, 0, 1).unsqueeze(0)
        
        if self.use_fp16:
             image_tensor = image_tensor.half()

        with torch.no_grad():
            output = self.model.infer(image_tensor, fov_x=fov_x, resolution_level=resolution_level, num_tokens=num_tokens, use_fp16=self.use_fp16)
            # Unpack first batch item
            res = {k: v.squeeze(0).cpu().numpy() for k, v in output.items()}
            return res

    def process_folder(self, 
        input_path: str,
        output_path: str,
        fov_x: Optional[float] = None,
        resize_to: Optional[int] = None,
        resolution_level: int = 9,
        num_tokens: int = 256,
        threshold: float = 0.03,
        save_maps: bool = False,
        save_glb: bool = False,
        save_ply: bool = False,
        show: bool = False
    ):
        include_suffices = ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']
        input_path_obj = Path(input_path)
        if input_path_obj.is_dir():
            image_paths = sorted(itertools.chain(*(input_path_obj.rglob(f'*.{suffix}') for suffix in include_suffices)))
        else:
            image_paths = [input_path_obj]
        
        if len(image_paths) == 0:
            raise FileNotFoundError(f'No image files found in {input_path}')

        if not any([save_maps, save_glb, save_ply]):
            warnings.warn('No output format specified. Defaults to saving all. Please use arguments to specify the output.')
            save_maps = save_glb = save_ply = True

        for image_path in (pbar := tqdm(image_paths, desc='Inference', disable=len(image_paths) <= 1)):
            if not image_path.exists():
                raise FileNotFoundError(f'File {image_path} does not exist.')
                
            image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            
            if resize_to is not None:
                height, width = min(resize_to, int(resize_to * height / width)), min(resize_to, int(resize_to * width / height))
                image = cv2.resize(image, (width, height), cv2.INTER_AREA)
            
            # Normalize to [0, 1]
            image_normalized = image / 255.0

            # Inference
            res = self.predict(image_normalized, fov_x=fov_x, resolution_level=resolution_level, num_tokens=num_tokens)
            
            points = res['points']
            depth = res['depth']
            mask = res['mask']
            intrinsics = res['intrinsics']
            normal = res.get('normal', None) 

            save_path_dir = Path(output_path, image_path.relative_to(input_path_obj).parent if input_path_obj.is_dir() else image_path_obj.parent, image_path.stem)
            save_path_dir.mkdir(exist_ok=True, parents=True)

            # Save images / maps
            if save_maps:
                cv2.imwrite(str(save_path_dir / 'image.jpg'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(save_path_dir / 'depth_vis.png'), cv2.cvtColor(colorize_depth(depth), cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(save_path_dir / 'depth.exr'), depth, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                cv2.imwrite(str(save_path_dir / 'mask.png'), (mask * 255).astype(np.uint8))
                cv2.imwrite(str(save_path_dir / 'points.exr'), cv2.cvtColor(points, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                if normal is not None:
                    cv2.imwrite(str(save_path_dir / 'normal.png'), cv2.cvtColor(colorize_normal(normal), cv2.COLOR_RGB2BGR))
                
                fov_x_val, fov_y_val = utils3d.np.intrinsics_to_fov(intrinsics)
                with open(save_path_dir / 'fov.json', 'w') as f:
                    json.dump({
                        'fov_x': round(float(np.rad2deg(fov_x_val)), 2),
                        'fov_y': round(float(np.rad2deg(fov_y_val)), 2),
                    }, f)

            # Export mesh & verification
            if save_glb or save_ply or show:
                mask_cleaned = mask & ~utils3d.np.depth_map_edge(depth, rtol=threshold)
                if normal is None:
                    faces, vertices, vertex_colors, vertex_uvs = utils3d.np.build_mesh_from_map(
                        points,
                        image.astype(np.float32) / 255,
                        utils3d.np.uv_map(height, width),
                        mask=mask_cleaned,
                        tri=True
                    )
                    vertex_normals = None
                else:
                    faces, vertices, vertex_colors, vertex_uvs, vertex_normals = utils3d.np.build_mesh_from_map(
                        points,
                        image.astype(np.float32) / 255,
                        utils3d.np.uv_map(height, width),
                        normal,
                        mask=mask_cleaned,
                        tri=True
                    )
                # When exporting the model, follow the OpenGL coordinate conventions:
                # - world coordinate system: x right, y up, z backward.
                # - texture coordinate system: (0, 0) for left-bottom, (1, 1) for right-top.
                vertices, vertex_uvs = vertices * [1, -1, -1], vertex_uvs * [1, -1] + [0, 1]
                if normal is not None:
                    vertex_normals = vertex_normals * [1, -1, -1]

            if save_glb:
                save_glb(save_path_dir / 'mesh.glb', vertices, faces, vertex_uvs, image, vertex_normals)

            if save_ply:
                save_ply(save_path_dir / 'pointcloud.ply', vertices, np.zeros((0, 3), dtype=np.int32), vertex_colors, vertex_normals)

            if show:
                import trimesh
                trimesh.Trimesh(
                    vertices=vertices,
                    vertex_colors=vertex_colors,
                    vertex_normals=vertex_normals,
                    faces=faces, 
                    process=False
                ).show()

if __name__ == '__main__':
    
    from PIL import Image
    img = Image.open("/path/to/data.1/assets/demo.png").convert("RGB")
    img = np.array(img)/255.0

    depth_est = MogeDepth()
    res = depth_est.predict(img)
    print(res.keys())

    print(res)
