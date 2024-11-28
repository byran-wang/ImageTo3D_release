import numpy as np
import torch
import math
import torch.nn.functional as F
from PIL import Image
import os
import trimesh
import nvdiffrast.torch as dr
import argparse
from Utils import make_mesh_tensors, nvdiffrast_render



def get_sphere_camera(elevation_deg = 30, 
                      azimuth_deg = -30, 
                      camera_distance = 2, 
                      fovy_deg = 41.15, 
                      size = [256, 256], # [height, width]
                      ):
    elevation = torch.tensor(elevation_deg) * math.pi / 180
    azimuth = torch.tensor(azimuth_deg) * math.pi / 180
    camera_distance = torch.tensor(camera_distance)
    camera_position = torch.stack(
        [
            camera_distance * torch.cos(elevation) * torch.cos(azimuth),
            camera_distance * torch.cos(elevation) * torch.sin(azimuth),
            camera_distance * torch.sin(elevation),
        ],
        dim=-1,
    )

    center = torch.zeros_like(camera_position)
    up = torch.as_tensor([0, 0, 1], dtype=torch.float32)

    lookat = F.normalize(center - camera_position, dim=-1)
    right = F.normalize(torch.cross(lookat, up), dim=-1)
    up = F.normalize(torch.cross(right, lookat), dim=-1)
    c2w = torch.cat(
        [torch.stack([right, up, -lookat], dim=-1), camera_position[:, None]],
        dim=-1,
    )
    glc2w4x4 = torch.cat(
        [c2w, torch.zeros_like(c2w[:1])], dim=0
    )
    glc2w4x4[3, 3] = 1.0
    glc2cvc = np.array([[1, 0, 0, 0],[0, -1 , 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    cvc2w4x4 = glc2w4x4 @ torch.tensor(glc2cvc).float()

    fovy = torch.deg2rad(torch.FloatTensor([fovy_deg]))
    width, height = size[1], size[0]    
    f = 0.5 * height / torch.tan(0.5 * fovy)
    K = torch.tensor(
        [
            [f, 0, int(0.5 * (width - 1))],
            [0, f, int(0.5 * (height - 1))],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )
    return np.array(cvc2w4x4), np.array(K)

def depth_to_point_cloud(depth_map, K):
    """
    Args:
        depth_map (numpy.ndarray): A 2D array (height, width) containing depth values.
        K (numpy.ndarray): The 3x3 intrinsic camera matrix.
        
    Returns:
        numpy.ndarray: A (N, 3) array where N is the number of valid points in the point cloud, and each row is a 3D point (X, Y, Z).
    """
    
    # Get the image height and width
    height, width = depth_map.shape

    # Generate a grid of pixel coordinates (u, v)
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # Flatten u, v, and depth_map for easier manipulation
    u_flat = u.flatten()
    v_flat = v.flatten()
    depth_flat = depth_map.flatten()

    # Filter out invalid (zero or very close to zero) depth values
    valid = depth_flat > -1000
    u_valid = u_flat[valid]
    v_valid = v_flat[valid]
    depth_valid = depth_flat[valid]

    # Create homogeneous pixel coordinates
    pixel_coords = np.stack([u_valid, v_valid, np.ones_like(u_valid)], axis=0)

    # Compute the inverse of the intrinsic matrix K
    K_inv = np.linalg.inv(K)

    # Unproject the pixel coordinates to camera space
    # point_camera = depth * K_inv * pixel_coords
    points_3d = depth_valid * (K_inv @ pixel_coords)

    # Transpose the result to get an (N, 3) point cloud
    points_3d = points_3d.T
    
    return points_3d

def rotate_camera_xyz(o2c_mat, angle_x_degrees, angle_y_degrees, angle_z_degrees):
    """
    Rotate a camera matrix along the X, Y, and Z axes by the given angles in degrees.
    
    Parameters:
    o2c_mat (numpy.ndarray): 4x4 camera matrix
    angle_x_degrees (float): Rotation angle around X-axis in degrees
    angle_y_degrees (float): Rotation angle around Y-axis in degrees
    angle_z_degrees (float): Rotation angle around Z-axis in degrees
    
    Returns:
    numpy.ndarray: Transformed 4x4 camera matrix
    """
    # Ensure the input is a 4x4 numpy array
    if not isinstance(o2c_mat, np.ndarray) or o2c_mat.shape != (4, 4):
        raise ValueError("Input must be a 4x4 numpy array")
    
    # Convert angles to radians
    angle_x_radians = np.deg2rad(angle_x_degrees)
    angle_y_radians = np.deg2rad(angle_y_degrees)
    angle_z_radians = np.deg2rad(angle_z_degrees)
    
    # Extract translation vector
    tx, ty, tz = o2c_mat[0, 3], o2c_mat[1, 3], o2c_mat[2, 3]
    
    # Create the rotation matrices
    rot_x = np.array([[1, 0, 0, 0],
                      [0, np.cos(angle_x_radians), -np.sin(angle_x_radians), 0],
                      [0, np.sin(angle_x_radians), np.cos(angle_x_radians), 0],
                      [0, 0, 0, 1]])
    
    rot_y = np.array([[np.cos(angle_y_radians), 0, np.sin(angle_y_radians), 0],
                      [0, 1, 0, 0],
                      [-np.sin(angle_y_radians), 0, np.cos(angle_y_radians), 0],
                      [0, 0, 0, 1]])
    
    rot_z = np.array([[np.cos(angle_z_radians), -np.sin(angle_z_radians), 0, 0],
                      [np.sin(angle_z_radians), np.cos(angle_z_radians), 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
    
    # Combine the rotation matrices
    combined_rotation = rot_z @ rot_y @ rot_x
    
    # Transform the camera matrix
    transformed_mat = combined_rotation @ o2c_mat
    
    # Set the new translation vector
    transformed_mat[0, 3] = tx
    transformed_mat[1, 3] = ty
    transformed_mat[2, 3] = tz
    
    return transformed_mat


def main(args):   

    mesh_obj = trimesh.load(args.object_mesh_f)
    if args.rm_small_blobs:
        # Split the mesh into connected components
        print("Splitting the mesh into connected components...")
        components = mesh_obj.split(only_watertight=False)

        # Define the size threshold
        min_faces = args.min_faces  # Adjust based on your mesh's scale and requirements

        # Filter out small components
        large_components = [comp for comp in components if len(comp.faces) >= min_faces]

        # Combine the large components
        if large_components:
            cleaned_mesh1 = trimesh.util.concatenate(large_components)

        else:
            print("No components are large enough to keep.")
        mesh_obj = cleaned_mesh1

    
    cmd = f"rm -rf {args.out_dir} && mkdir -p {args.out_dir}"

    os.system(cmd)
    image_size = args.image_size # [480, 640]

    elevation_deg, camera_distance, fovy_deg = args.elevation_deg, args.camera_distance, args.fovy_deg
    azimuth_degs = np.float32(np.linspace(0, 360, args.image_numbers))
    for i, azimuth_deg in enumerate(azimuth_degs):
        print(f"Rendering {i+1}/{len(azimuth_degs)}")
        cvc2w4x4, K =  get_sphere_camera(elevation_deg = elevation_deg, azimuth_deg = azimuth_deg, camera_distance = camera_distance, fovy_deg = fovy_deg, size = image_size)
        w2cvc4x4 = np.linalg.inv(cvc2w4x4)
        
        # y: 左右,-往右转，+往左转； x：上下，-往上转，+往下转
        # w2cvc4x4 = rotate_camera_xyz(w2cvc4x4, angle_x_degrees=-10, angle_y_degrees=0, angle_z_degrees=0)

        w2cvc4x4 = torch.tensor(w2cvc4x4, dtype=torch.float)[None].to('cuda')
        mesh_tensors = make_mesh_tensors(mesh_obj)
        rgb_r, depth_r, normal_r = nvdiffrast_render(K=K, H=image_size[0], W=image_size[1], ob_in_cams=w2cvc4x4, context='cuda', get_normal=True, 
                                                    glctx=dr.RasterizeCudaContext(), mesh_tensors=mesh_tensors, output_size=image_size, bbox2d=None, use_light=True)    

        rgb_r = rgb_r[0].cpu().numpy()
        normal_r = normal_r[0].cpu().numpy()
        normal_r = (normal_r + 1) / 2
        normal_r = 1 - normal_r

        # Step 2: Normalize and scale to [0, 255]
        # Ensure that both images are in the range [0, 1]
        # If `normal_r` has values outside [0, 1], you may need to normalize it accordingly
        rgb_scaled = np.clip(rgb_r, 0.0, 1.0)
        normal_scaled = np.clip(normal_r, 0.0, 1.0)

        # Scale to [0, 255] and convert to uint8
        rgb_uint8 = (rgb_scaled * 255).astype(np.uint8)        # Shape: (480, 640, 3)
        normal_uint8 = (normal_scaled * 255).astype(np.uint8)  # Shape: (480, 640, 3)

        object_mask_np = np.zeros_like(normal_uint8[...,0])
        object_mask_np[normal_scaled[...,0] != 0.5] = 255
        normal_uint8[object_mask_np == 0] = 255
        normal_uint8 = np.concatenate([normal_uint8, object_mask_np[:, :, None]], axis=2)
        rgb_uint8[object_mask_np == 0] = 255
        rgb_uint8 = np.concatenate([rgb_uint8, object_mask_np[:, :, None]], axis=2)

        
        # Step 3: Concatenate side-by-side (rgb_r on the left, normal_r on the right)
        merged_image = np.concatenate((rgb_uint8, normal_uint8), axis=1)  # Shape: (480, 1920, 3)

        # Step 4: Convert to PIL Image and save as PNG
        merged_pil = Image.fromarray(merged_image)
        merged_f = f"{args.out_dir}/{i:04d}.png"
        merged_pil.save(merged_f) # Save the image as 'merged_rgb_normal.png'`       
           
    
    cmd = f'''/usr/bin/ffmpeg -y -framerate 5 -i {args.out_dir}/%04d.png -c:v libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" {args.out_dir}/output.mp4'''
    os.system(cmd)
    print(f"Saved video to {args.out_dir}/output.mp4")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--object_mesh_f", type=str, default="", help="object mesh file")
    parser.add_argument("--image_size", type=str, default=[512, 512], help="image [height, width]")
    parser.add_argument("--image_numbers", type=int, default=60, help="image rendering total number")
    parser.add_argument("--elevation_deg", type=float, default=30.0)
    parser.add_argument("--camera_distance", type=float, default=3.0)
    parser.add_argument("--fovy_deg", type=float, default=41.15)
    parser.add_argument("--min_faces", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="outputs", help="output directory")
    parser.add_argument("--rm_small_blobs", action="store_true")

    args, extras = parser.parse_known_args()            
    main(args)