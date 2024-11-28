# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch,trimesh,logging
import nvdiffrast.torch as dr
import torch.nn.functional as F
import cv2
import numpy as np

glcam_in_cvcam = np.array([[1,0,0,0],
                          [0,-1,0,0],
                          [0,0,-1,0],
                          [0,0,0,1]]).astype(float)

def make_mesh_tensors(mesh, device='cuda', max_tex_size=None):
  mesh_tensors = {}
  if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
    img = np.array(mesh.visual.material.image.convert('RGB'))
    img = img[...,:3]
    if max_tex_size is not None:
      max_size = max(img.shape[0], img.shape[1])
      if max_size>max_tex_size:
        scale = 1/max_size * max_tex_size
        img = cv2.resize(img, fx=scale, fy=scale, dsize=None)
    mesh_tensors['tex'] = torch.as_tensor(img, device=device, dtype=torch.float)[None]/255.0
    mesh_tensors['uv_idx']  = torch.as_tensor(mesh.faces, device=device, dtype=torch.int)
    uv = torch.as_tensor(mesh.visual.uv, device=device, dtype=torch.float)
    uv[:,1] = 1 - uv[:,1]
    mesh_tensors['uv']  = uv
  else:
    if mesh.visual.vertex_colors is None:
      print.info(f"WARN: mesh doesn't have vertex_colors, assigning a pure color")
      mesh.visual.vertex_colors = np.tile(np.array([128,128,128]).reshape(1,3), (len(mesh.vertices), 1))
    mesh_tensors['vertex_color'] = torch.as_tensor(mesh.visual.vertex_colors[...,:3], device=device, dtype=torch.float)/255.0

  mesh_tensors.update({
    'pos': torch.tensor(mesh.vertices, device=device, dtype=torch.float),
    'faces': torch.tensor(mesh.faces, device=device, dtype=torch.int),
    'vnormals': torch.tensor(mesh.vertex_normals, device=device, dtype=torch.float),
  })
  return mesh_tensors


def nvdiffrast_render(K=None, H=None, W=None, ob_in_cams=None, glctx=None, context='cuda', get_normal=False, mesh_tensors=None, mesh=None, projection_mat=None, bbox2d=None, output_size=None, use_light=False, light_color=None, light_dir=np.array([0,0,1]), light_pos=np.array([0,0,0]), w_ambient=0.8, w_diffuse=0.5, extra={}):
  '''Just plain rendering, not support any gradient
  @K: (3,3) np array
  @ob_in_cams: (N,4,4) torch tensor, openCV camera
  @projection_mat: np array (4,4)
  @output_size: (height, width)
  @bbox2d: (N,4) (umin,vmin,umax,vmax) if only roi need to render.
  @light_dir: in cam space
  @light_pos: in cam space
  '''
  if glctx is None:
    if context == 'gl':
      glctx = dr.RasterizeGLContext()
    elif context=='cuda':
      glctx = dr.RasterizeCudaContext()
    else:
      raise NotImplementedError
    print.info("created context")

  if mesh_tensors is None:
    mesh_tensors = make_mesh_tensors(mesh)
  pos = mesh_tensors['pos']
  vnormals = mesh_tensors['vnormals']
  pos_idx = mesh_tensors['faces']
  has_tex = 'tex' in mesh_tensors

  ob_in_glcams = torch.tensor(glcam_in_cvcam, device='cuda', dtype=torch.float)[None]@ob_in_cams
  if projection_mat is None:
    projection_mat = projection_matrix_from_intrinsics(K, height=H, width=W, znear=0.1, zfar=100)
  projection_mat = torch.as_tensor(projection_mat.reshape(-1,4,4), device='cuda', dtype=torch.float)
  mtx = projection_mat@ob_in_glcams

  if output_size is None:
    output_size = np.asarray([H,W])

  pts_cam = transform_pts(pos, ob_in_cams)
  pos_homo = to_homo_torch(pos)
  pos_clip = (mtx[:,None]@pos_homo[None,...,None])[...,0]
  if bbox2d is not None:
    l = bbox2d[:,0]
    t = H-bbox2d[:,1]
    r = bbox2d[:,2]
    b = H-bbox2d[:,3]
    tf = torch.eye(4, dtype=torch.float, device='cuda').reshape(1,4,4).expand(len(ob_in_cams),4,4).contiguous()
    tf[:,0,0] = W/(r-l)
    tf[:,1,1] = H/(t-b)
    tf[:,3,0] = (W-r-l)/(r-l)
    tf[:,3,1] = (H-t-b)/(t-b)
    pos_clip = pos_clip@tf
  rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=np.asarray(output_size))
  xyz_map, _ = dr.interpolate(pts_cam, rast_out, pos_idx)
  depth = xyz_map[...,2]
  if has_tex:
    texc, _ = dr.interpolate(mesh_tensors['uv'], rast_out, mesh_tensors['uv_idx'])
    color = dr.texture(mesh_tensors['tex'], texc, filter_mode='linear')
  else:
    color, _ = dr.interpolate(mesh_tensors['vertex_color'], rast_out, pos_idx)

  if use_light:
    get_normal = True
  if get_normal:
    vnormals_cam = transform_dirs(vnormals, ob_in_cams)
    normal_map, _ = dr.interpolate(vnormals_cam, rast_out, pos_idx)
    normal_map = F.normalize(normal_map, dim=-1)
    normal_map = torch.flip(normal_map, dims=[1])
  else:
    normal_map = None

  if use_light:
    if light_dir is not None:
      light_dir_neg = -torch.as_tensor(light_dir, dtype=torch.float, device='cuda')
    else:
      light_dir_neg = torch.as_tensor(light_pos, dtype=torch.float, device='cuda').reshape(1,1,3) - pts_cam
    diffuse_intensity = (F.normalize(vnormals_cam, dim=-1) * F.normalize(light_dir_neg, dim=-1)).sum(dim=-1).clip(0, 1)[...,None]
    diffuse_intensity_map, _ = dr.interpolate(diffuse_intensity, rast_out, pos_idx)  # (N_pose, H, W, 1)
    if light_color is None:
      light_color = color
    else:
      light_color = torch.as_tensor(light_color, device='cuda', dtype=torch.float)
    color = color*w_ambient + diffuse_intensity_map*light_color*w_diffuse

  color = color.clip(0,1)
  color = color * torch.clamp(rast_out[..., -1:], 0, 1) # Mask out background using alpha
  color = torch.flip(color, dims=[1])   # Flip Y coordinates
  depth = torch.flip(depth, dims=[1])
  extra['xyz_map'] = torch.flip(xyz_map, dims=[1])
  return color, depth, normal_map






def to_homo(pts):
  '''
  @pts: (N,3 or 2) will homogeneliaze the last dimension
  '''
  assert len(pts.shape)==2, f'pts.shape: {pts.shape}'
  homo = np.concatenate((pts, np.ones((pts.shape[0],1))),axis=-1)
  return homo


def to_homo_torch(pts):
  '''
  @pts: shape can be (...,N,3 or 2) or (N,3) will homogeneliaze the last dimension
  '''
  ones = torch.ones((*pts.shape[:-1],1), dtype=torch.float, device=pts.device)
  homo = torch.cat((pts, ones),dim=-1)
  return homo


def transform_pts(pts,tf):
  """Transform 2d or 3d points
  @pts: (...,N_pts,3)
  @tf: (...,4,4)
  """
  if len(tf.shape)>=3 and tf.shape[-3]!=pts.shape[-2]:
    tf = tf[...,None,:,:]
  return (tf[...,:-1,:-1]@pts[...,None] + tf[...,:-1,-1:])[...,0]


def transform_dirs(dirs,tf):
  """
  @dirs: (...,3)
  @tf: (...,4,4)
  """
  if len(tf.shape)>=3 and tf.shape[-3]!=dirs.shape[-2]:
    tf = tf[...,None,:,:]
  return (tf[...,:3,:3]@dirs[...,None])[...,0]




def projection_matrix_from_intrinsics(K, height, width, znear, zfar, window_coords='y_down'):
  """Conversion of Hartley-Zisserman intrinsic matrix to OpenGL proj. matrix.

  Ref:
  1) https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL
  2) https://github.com/strawlab/opengl-hz/blob/master/src/calib_test_utils.py

  :param K: 3x3 ndarray with the intrinsic camera matrix.
  :param x0 The X coordinate of the camera image origin (typically 0).
  :param y0: The Y coordinate of the camera image origin (typically 0).
  :param w: Image width.
  :param h: Image height.
  :param nc: Near clipping plane.
  :param fc: Far clipping plane.
  :param window_coords: 'y_up' or 'y_down'.
  :return: 4x4 ndarray with the OpenGL projection matrix.
  """
  x0 = 0
  y0 = 0
  w = width
  h = height
  nc = znear
  fc = zfar

  depth = float(fc - nc)
  q = -(fc + nc) / depth
  qn = -2 * (fc * nc) / depth

  # Draw our images upside down, so that all the pixel-based coordinate
  # systems are the same.
  if window_coords == 'y_up':
    proj = np.array([
      [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
      [0, -2 * K[1, 1] / h, (-2 * K[1, 2] + h + 2 * y0) / h, 0],
      [0, 0, q, qn],  # Sets near and far planes (glPerspective).
      [0, 0, -1, 0]
      ])

  # Draw the images upright and modify the projection matrix so that OpenGL
  # will generate window coords that compensate for the flipped image coords.
  elif window_coords == 'y_down':
    proj = np.array([
      [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
      [0, 2 * K[1, 1] / h, (2 * K[1, 2] - h + 2 * y0) / h, 0],
      [0, 0, q, qn],  # Sets near and far planes (glPerspective).
      [0, 0, -1, 0]
      ])
  else:
    raise NotImplementedError

  return proj
