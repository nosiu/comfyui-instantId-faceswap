import numpy as np
import cv2
import math
import torch
import math
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from .ip_adapter.instantId import CrossAttentionPatch

def draw_kps(w, h, kps, color_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)], alphas=[1, 1, 1, 1, 1]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)
    out_img = np.zeros([int(h), int(w), 3], dtype=np.uint8)

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]
        alpha = alphas[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))

        polygon = cv2.ellipse2Poly(
            (int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1
        )

        limb_img = np.zeros_like(out_img)
        cv2.fillConvexPoly(limb_img, polygon, color)
        out_img = cv2.addWeighted(out_img, 1, limb_img, float(alpha) * 0.6, 0)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        alpha = alphas[idx_kp] 
        x = kp[0]
        y = kp[1]
        kp_img = out_img.copy()
        cv2.circle(kp_img, (int(x), int(y)), 10, color, -1)
        out_img = cv2.addWeighted(out_img, 1 - float(alpha), kp_img, float(alpha), 0)

    return out_img.astype(np.uint8)


# based on https://github.com/laksjdjf/IPAdapter-ComfyUI/blob/main/ip_adapter.py#L19
def set_model_patch_replace(model, patch_kwargs, key):
  attn = "attn2"
  to = model.model_options["transformer_options"].copy()
  if "patches_replace" not in to:
    to["patches_replace"] = {}
  else:
    to["patches_replace"] = to["patches_replace"].copy()

  if attn not in to["patches_replace"]:
    to["patches_replace"][attn] = {}
  else:
    to["patches_replace"][attn] = to["patches_replace"][attn].copy()
  if key not in to["patches_replace"][attn]:
    to["patches_replace"][attn][key] = CrossAttentionPatch(**patch_kwargs)
    model.model_options["transformer_options"] = to
  else:
    to["patches_replace"][attn][key].set_new_condition(**patch_kwargs)


def resize_to_fit_area(original_width, original_height, area_width, area_height):
  base_pixels = 8
  max_area= area_width * area_height
  aspect_ratio = original_width / original_height

  scale_factor = math.sqrt(max_area / (original_width * original_height))
  new_width = int(original_width * scale_factor)
  new_height = int(original_height * scale_factor)

  new_width = new_width // base_pixels * base_pixels
  new_height = new_height // base_pixels * base_pixels

  if new_width * new_height > max_area:
      new_width = math.floor(math.sqrt(max_area * aspect_ratio)) // base_pixels * base_pixels
      new_height = math.floor(new_width / aspect_ratio) // base_pixels * base_pixels

  return (new_width, new_height)


def get_mask_bbox_with_padding(mask_image, pad_top, pad_right, pad_bottom, pad_left):
  mask_segments = torch.nonzero(mask_image == 1, as_tuple=False)
  if torch.count_nonzero(mask_segments).item() == 0:
    raise Exception("Draw a mask on pose image")

  m_y1 = torch.min(mask_segments[:, 0]).item()
  m_y2 = torch.max(mask_segments[:, 0]).item()
  m_x1 = torch.min(mask_segments[:, 1]).item()
  m_x2 = torch.max(mask_segments[:, 1]).item()

  height, width = mask_image.shape

  p_x1 = max(0, m_x1 - pad_left)
  p_y1 = max(0, m_y1 - pad_top)
  p_x2 = min(width, m_x2 + pad_right)
  p_y2 = min(height, m_y2 + pad_bottom)

  return int(p_x1), int(p_y1), int(p_x2), int(p_y2)


def get_kps_from_image(image, insightface):
  np_pose_image = (255.0 * image.cpu().numpy().squeeze()).clip(0, 255).astype(np.uint8)
  face_info = insightface.get(cv2.cvtColor(np_pose_image, cv2.COLOR_RGB2BGR))
  assert len(face_info) > 0, "No face detected in pose image"
  face_info = sorted(face_info, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))[-1] # only use the maximum face
  return face_info["kps"]


def get_angle(a=(0, 0), b=(0, 0), round_angle=False):
    # a, b - eyes
    angle = math.atan2(b[1] - a[1], b[0] - a[0]) * 180 / math.pi
    if round_angle:
        angle = round(angle / 90) * 90
        if angle == 360: angle = 0

    return angle


def calculate_size_after_rotation(width, height, angle):
    angle_rad = math.radians(angle)
    
    new_width = abs(width * math.cos(angle_rad)) + abs(height * math.sin(angle_rad))
    new_height = abs(width * math.sin(angle_rad)) + abs(height * math.cos(angle_rad))
    
    return (int(np.ceil(new_width)), int(np.ceil(new_height))) #+ 1?


def image_rotate_with_pad(image, clockwise, angle):
  if not clockwise: angle *= -1

  image = image.squeeze(0)
  image = image.permute(2, 0, 1)
  image = TF.rotate(image, angle, fill=0, expand=True)
  image = image.permute(1, 2, 0)
  image = image.unsqueeze(0)
  return image


def kps_rotate_2d(points, original_width, original_height, new_width, new_height, angle):
    angle_rad = math.radians(angle)
    
    original_center_x = original_width / 2
    original_center_y = original_height / 2
    
    new_center_x = new_width / 2
    new_center_y = new_height / 2
    
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    
    rotated_points = []
    
    for point in points:
        x, y = point
        
        translated_x = x - original_center_x
        translated_y = y - original_center_y
        
        rotated_x = translated_x * cos_angle - translated_y * sin_angle
        rotated_y = translated_x * sin_angle + translated_y * cos_angle
        
        final_x = int(round(rotated_x + new_center_x))
        final_y = int(round(rotated_y + new_center_y))
        
        rotated_points.append([final_x, final_y])
    
    return rotated_points


def kps_rotate_3d(points, angleXDeg, angleYDeg, angleZDeg):
    angleX = math.radians(angleXDeg)
    angleY = math.radians(angleYDeg)
    angleZ = math.radians(angleZDeg)

    center = np.mean(points, axis=0)

    translated_points = np.array([point - center for point in points])

    def rotate_x(point, angle):
        cos_theta = math.cos(angle)
        sin_theta = math.sin(angle)
        return [
            int(point[0]),
            int(point[1] * cos_theta - point[2] * sin_theta),
            int(point[1] * sin_theta + point[2] * cos_theta)
        ]

    def rotate_y(point, angle):
        cos_theta = math.cos(angle)
        sin_theta = math.sin(angle)
        return [
            int(point[0] * cos_theta + point[2] * sin_theta),
            int(point[1]),
            int(-point[0] * sin_theta + point[2] * cos_theta)
        ]

    def rotate_z(point, angle):
        cos_theta = math.cos(angle)
        sin_theta = math.sin(angle)
        return [
            int(point[0] * cos_theta - point[1] * sin_theta),
            int(point[0] * sin_theta + point[1] * cos_theta),
            int(point[2])
        ]

    rotated_points = [
        rotate_z(rotate_y(rotate_x(point, angleX), angleY), angleZ)
        for point in translated_points
    ]

    return [point + center for point in rotated_points]


def kps3d_to_kps2d (kps):
  if len(kps['array'][0]) == 3:
    kps2d = {
      'width': kps['width'],
      'height': kps['height'],
      'opacities': kps['opacities'][:],
      'array': []
    }

    for x, y, _ in kps['array']:
       kps2d['array'].append([x, y])

    return kps2d
  return kps


def get_bbox_from_kps (kps_data, grow_by):
  kps = np.array(kps_data['array'])
  minX, minY = np.min(kps, axis=0)
  maxX, maxY = np.max(kps, axis=0)
  width = (maxX - minX) / grow_by
  height = ((maxY - minY) / grow_by)

  return [
    [
      int(max(np.ceil(minX - (width)), 0)),
      int(max(np.ceil(minY - (height)), 0))
    ],
    [
      int(min(np.ceil(maxX + (width)), kps_data['width'])),
      int(min(np.ceil(maxY + (height)), kps_data['height']))
    ]   
  ]