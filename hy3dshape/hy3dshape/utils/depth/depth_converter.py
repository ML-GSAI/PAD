


from hy3dshape.utils.depth.depth_moge import MogeDepth
import numpy as np
import torch
from PIL import Image

class ImageToVAE():
    def __init__(self, device = "cuda"):
        self.device = device
        self.depth_estimator = MogeDepth(device=device)

    def estimate_normal_from_depth(self, depth_map, fx, fy, cx, cy):
        """
        
        Args:
            depth_map: (H, W) ,
            fx, fy: 
            cx, cy: 
            
        Returns:
            normal_map: (H, W, 3) ,,
        """
        # Fix: ensure depth_map is finite and clip large values
        depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
        
        h, w = depth_map.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        fx = fx * w
        fy = fy * h
        cx = cx * w
        cy = cy * h
        
        # x = (u - cx) * z / fx
        
        Z = -depth_map
        X = (u - cx) * depth_map / fx
        Y = -(v - cy) * depth_map / fy
        
        pts = np.stack([X, Y, Z], axis=-1) 
        
        grad_v, grad_u = np.gradient(pts, axis=(0, 1))
        
        #   grad_v x grad_u  ≈ (-Y) x (+X) = - (Y x X) = - (-Z) = +Z
        normal = np.cross(grad_v, grad_u)

        # Fix: remove nan/inf from normals before normalization
        normal = np.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0)
        
        norm = np.linalg.norm(normal, axis=-1, keepdims=True)
        
        valid_norm = norm > 1e-12
        normal = np.divide(normal, norm, out=np.zeros_like(normal), where=valid_norm)
        
        return normal

    def normalize_to_unit_box(self, V, range_val=(-1, 1), ret_raw=False):
        """
        Normalize the vertices V to fit inside a unit bounding box [-1,1]^3.
        V: (n,3) numpy array or tensor of vertex positions.
        Returns: normalized V (same type as input, or numpy)
        """
        assert V.ndim == 2 and V.shape[1] == 3, "Input V should be (n,3)"
        is_numpy = False
        if isinstance(V, np.ndarray):
            V = torch.from_numpy(V)
            is_numpy = True

        V_min = V.min(dim=0).values
        V_max = V.max(dim=0).values

        
        scale = 1.0 / (V_max - V_min).max() 
        offset = -(V_max + V_min) / 2.0
        target_min, target_max = range_val
        target_center = (target_min + target_max) / 2.0
        scale = scale * (target_max - target_min)
        offset = offset + target_center / scale
        V_normalized = (V + offset) * scale 

        if is_numpy:
            V_normalized = V_normalized.cpu().numpy()
            if isinstance(scale, torch.Tensor): scale = scale.item()
            if isinstance(offset, torch.Tensor): offset = offset.cpu().numpy()

        if ret_raw:
            return V_normalized, scale, offset

        return V_normalized

    def process_image(self, image_input):
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
            image = np.array(image) / 255.0
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
            image = np.array(image) / 255.0
        elif isinstance(image_input, (np.ndarray, np.generic)):
            image = image_input
            if image.ndim == 3 and image.shape[0] == 3: 
                image = image.transpose(1, 2, 0)
            if image.mean() > 1.0:
                image = image / 255.0
            if image.min() < -0.5:
                image = (image + 1.0) / 2.0
        
        elif isinstance(image_input, torch.Tensor):
            image = image_input.detach().cpu().numpy()
            if image.ndim == 4:
                image = image[0]
            if image.ndim == 3 and image.shape[0] == 3:
                image = image.transpose(1, 2, 0)
            if image.mean() > 1.0:
                image = image / 255.0
            if image.min() < -0.5:
                image = (image + 1.0) / 2.0
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")
        
        return image
    
    def backproject(self, depth_map, c2w, fx, fy, cx, cy, mask_th=0.999, is_filter = True):
        h, w = depth_map.shape
        depth = depth_map
        mask = depth_map < mask_th
        u = np.arange(0, w)
        v = np.arange(0, h)
        uu, vv = np.meshgrid(u, v)


        fx = fx * w
        fy = fy * h
        cx = cx * w
        cy = cy * h

        print("backproject: fx, fy, cx, cy, w, h:", fx, fy, cx, cy, w, h)

        z = depth
        x = (uu - cx) * z / fx
        y = (vv - cy) * z / fy
        pts_cam = np.stack([x, -y, -z, np.ones_like(z)], axis=-1).reshape(-1,4)  # N x 4
        mask_flat = mask.reshape(-1)
        if is_filter:
            pts_cam_valid = pts_cam[mask_flat]
        else:
            pts_cam_valid = pts_cam
        # world coordinates
        pts_world_h = (c2w @ pts_cam_valid.T).T  # N x 4
        pts_world = pts_world_h[:, :3] / pts_world_h[:, 3:4]
        return pts_world, mask.reshape(-1)

    def __call__(self, image_input, num_points=81920):
        # returns [1, num_points, 7] tensor

        image = self.process_image(image_input)

        results = self.depth_estimator.predict(image)
        pts = results['points']  # (H, W, 3)
        intrinsic = results['intrinsics']


        fx = intrinsic[0,0]
        fy = intrinsic[1,1]
        cx = intrinsic[0,2]
        cy = intrinsic[1,2]

        depth = results['depth']  # (H, W)
        
        # Filter invalid depth and neighbors
        bad_mask = ~np.isfinite(depth)
        # Dilate 2 iterations
        for _ in range(2):
            dilated = bad_mask.copy()
            dilated[1:, :] |= bad_mask[:-1, :]
            dilated[:-1, :] |= bad_mask[1:, :]
            dilated[:, 1:] |= bad_mask[:, :-1]
            dilated[:, :-1] |= bad_mask[:, 1:]
            bad_mask = dilated

        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        pts = self.backproject(depth, np.eye(4), fx, fy, cx, cy, is_filter=False)[0].reshape(pts.shape)  # (H, W, 3)

        normal = self.estimate_normal_from_depth(
            depth_map=depth,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )

        pts = pts.reshape(-1, 3)
        normal = normal.reshape(-1, 3)
        bad_mask = bad_mask.reshape(-1)
        
        valid_mask = np.isfinite(pts).all(axis=1) & np.isfinite(normal).all(axis=1) & (np.abs(pts) < 1e6).all(axis=1)
        valid_mask = valid_mask & (~bad_mask)

        pts = pts[valid_mask]
        normal = normal[valid_mask]

        if pts.shape[0] == 0:
             # Fallback: create random points if everything is filtered out
             print("Warning: All points filtered out. Returning random noise.")
             pts = np.random.rand(num_points, 3) * 2 - 1
             normal = np.random.rand(num_points, 3)
             normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
             pts = np.concatenate([pts, normal, np.zeros((num_points, 1))], axis=-1)
             return torch.from_numpy(pts).unsqueeze(0).cuda().half()

        pts[:, :3] = self.normalize_to_unit_box(pts[:, :3], range_val=(-0.99,0.99))

        pts = np.concatenate([pts, normal], axis=-1)  # (N, 6)
        pts = np.concatenate([pts, np.zeros((pts.shape[0], 1))], axis=-1)  # (N, 7)

        if pts.shape[0] >= num_points:
            index = np.random.choice(pts.shape[0], size=num_points, replace=False)
            pts = pts[index]
        else:
            index = np.random.choice(pts.shape[0], size=num_points, replace=True)
            pts = pts[index]

        pts = torch.from_numpy(pts).unsqueeze(0).cuda().half()
        return pts
