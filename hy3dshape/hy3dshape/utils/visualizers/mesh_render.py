import pyvista as pv
import os
import glob
import argparse
import numpy as np
import imageio.v2 as imageio
import shutil


def save_ply(filename: str, points: np.ndarray):
    """Save an Nx3 numpy array to an ASCII PLY file."""
    import os
    if not isinstance(points, np.ndarray):
        raise TypeError("points must be a numpy.ndarray")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    # Ensure directory exists
    dirpath = os.path.dirname(filename)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    pts = points.astype(float)
    n = pts.shape[0]

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]

    with open(filename, "w") as f:
        f.write("\n".join(header) + "\n")
        for x, y, z in pts:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")



def concate_video_row(vidlist):
    # vidlist: list of (Frames, H, W, 3)
    # return: (Frames, H, W*len, 3)
    return np.concatenate(vidlist, axis=2)

def concate_video_t(vidlist):
    return np.concatenate(vidlist, axis=0)


def render_glb_to_video(mesh, frames = 60, texture_path=None, window_size=[1024, 1024], output_path=None):
    # ret = (Frames, H, W, 3)
    if texture_path is None:
        texture_path = "/path/to/data"
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
        
    if texture_path and os.path.exists(texture_path):
        try:
            cubemap = pv.cubemap(texture_path)
            plotter.set_environment_texture(cubemap)
        except Exception as e:
            print(f"Warning: Could not load environment texture: {e}")
    
    if isinstance(mesh, str):
        mesh = pv.read(mesh)
    else:
        mesh = pv.wrap(mesh)
        
    # Center the mesh
    center = mesh.center
    mesh.translate((-center[0], -center[1], -center[2]), inplace=True)


    plotter.add_mesh(mesh, pbr=True, metallic=0.0, roughness=0.8)
    

    plotter.camera.up = (0.0, 1.0, 0.0)
    plotter.camera.position = (0.0, 0.0, 1.3) 
    plotter.camera.focal_point = (0.0, 0.0, 0.0)
    plotter.reset_camera()
    plotter.camera.zoom(0.8)



    
    video_frames = []
    angle_step = 360.0 / frames

    plotter.show(auto_close=False)

    for frame in range(frames):
        plotter.camera.azimuth += angle_step
        
        plotter.render()
        
        video_frames.append(plotter.image)
        
    plotter.close()
    
    vid = np.stack(video_frames, axis=0)
    if output_path is not None:
        imageio.mimsave(output_path, vid, fps=30)
    return vid

