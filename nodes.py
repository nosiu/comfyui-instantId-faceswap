import os
import cv2
import torch
import numpy as np
import json
import comfy.utils
import folder_paths
from urllib.parse import urlparse, parse_qs
from server import PromptServer
from aiohttp import web
from comfy_execution.graph_utils import GraphBuilder
from .ip_adapter.resampler import Resampler
from .ip_adapter.instantId import InstantId
from insightface.app import FaceAnalysis
from .utils import draw_kps, set_model_patch_replace, resize_to_fit_area, \
  kps_rotate_2d, kps_rotate_3d, kps3d_to_kps2d, calculate_size_after_rotation, \
  get_mask_bbox_with_padding,get_kps_from_image,get_angle, image_rotate_with_pad, \
  get_bbox_from_kps


folder_paths.folder_names_and_paths["ipadapter"] = ([os.path.join(folder_paths.models_dir, "ipadapter")], folder_paths.supported_pt_extensions)
INSIGHTFACE_PATH = os.path.join(folder_paths.models_dir, "insightface")
CATEGORY_NAME = "InstantId Faceswap"
MAX_RESOLUTION = 16384


#==============================================================================
# get key points and landmarks position as 3d points and send it back to frontend when requested
routes = PromptServer.instance.routes
@routes.post('/get_keypoints_for_instantId')
async def proxy_handle(request):
  post = await request.json()

  try:
    app = FaceAnalysis(
      name="antelopev2",
      root=INSIGHTFACE_PATH,
      providers=["CPUExecutionProvider", "CUDAExecutionProvider"]
    )

    app.prepare(ctx_id=0, det_size=(640, 640))
    parsed_url = urlparse(post['image'])
    queries = parse_qs(parsed_url.query)
    path = os.path.join(folder_paths.get_directory_by_type(queries['type'][0]), queries['filename'][0])
    image = cv2.imread(path)
    faces = app.get(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if len(faces) == 0:
      raise Exception("No face detected")

    face = sorted(faces, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))[-1] # only use the maximum face

    landmarks_3d = face.landmark_3d_68
    # KPS
    left_eye = np.mean(landmarks_3d[36:42], axis=0)
    right_eye = np.mean(landmarks_3d[42:48], axis=0)
    nose_tip = np.array([landmarks_3d[33][0], landmarks_3d[33][1], landmarks_3d[30][2]])
    left_mouth = landmarks_3d[48]
    right_mouth = landmarks_3d[54]
    
    return web.json_response({
      "status": "ok",
      "data": {
        "kps": [ left_eye.tolist(), right_eye.tolist(), nose_tip.tolist(), left_mouth.tolist(), right_mouth.tolist() ],
        "jawline": landmarks_3d[0:17].tolist(), 
        "eyebrow_left": landmarks_3d[17:22].tolist(),
        "eyebrow_right": landmarks_3d[22:27].tolist(),
        "nose_bridge": landmarks_3d[27:31].tolist(), 
        "nose_lower": landmarks_3d[31:36].tolist(), 
        "eye_left": landmarks_3d[36:42].tolist(),
        "eye_right": landmarks_3d[42:48].tolist(),
        "mouth_outer": landmarks_3d[48:60].tolist(),
        "mouth_inner": landmarks_3d[60:68].tolist(),
        }
    })
  except Exception as error:
      if str(error) == "No face detected":
        return web.json_response({
          "error": "No face detected"
        })
      else:
        raise error


#==============================================================================
class FaceEmbed:
  def __init__(self):
    pass

  @classmethod
  def INPUT_TYPES(self):
    return {
      "required": {
        "insightface":  ("INSIGHTFACE_APP",),
        "face_image":  ("IMAGE",)
      },
      "optional": {
        "face_embeds": ("FACE_EMBED",)
      }
    }

  RETURN_TYPES = ("FACE_EMBED",)
  RETURN_NAMES = ("face embeds",)
  FUNCTION = "make_face_embed"
  CATEGORY = CATEGORY_NAME

  def make_face_embed(self, insightface, face_image, face_embeds = None):
    face_image = (255.0 * face_image.cpu().numpy().squeeze()).clip(0, 255).astype(np.uint8)
    face_info = insightface.get(cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))

    assert len(face_info) > 0, "No face detected for face embed"

    face_info = sorted(face_info, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))[-1] # only use the maximum face
    face_emb = torch.tensor(face_info["embedding"], dtype=torch.float32).unsqueeze(0)

    if face_embeds is None:
      return (face_emb,)

    face_embeds = torch.cat((face_embeds, face_emb), dim=-2)
    return (face_embeds,)


#==============================================================================
class FaceEmbedCombine:
  def __init__(self):
    pass

  @classmethod
  def INPUT_TYPES(self):
    return {
      "required": {
        "resampler":  ("RESAMPLER",),
        "face_embeds":  ("FACE_EMBED",)
      },
    }

  RETURN_TYPES = ("FACE_CONDITIONING",)
  RETURN_NAMES = ("face conditioning",)
  FUNCTION = "combine_face_embed"
  CATEGORY = CATEGORY_NAME

  def combine_face_embed(self, resampler, face_embeds):
    embeds = torch.mean(face_embeds, dim=0, dtype=torch.float32).unsqueeze(0)
    embeds = embeds.reshape([1, -1, 512])
    conditionings = resampler(embeds).to(comfy.model_management.get_torch_device())
    return (conditionings,)


#==============================================================================
class AngleFromFace:
  rotate_modes = ["none", "loseless", "any"]
  def __init__(self):
      pass

  @classmethod
  def INPUT_TYPES(self):
    return {
      "required": {
        "insightface": ("INSIGHTFACE_APP",),
        "image": ("IMAGE", { "tooltip": "Pose image." }),
        "mask": ("MASK",),
        "rotate_mode": (self.rotate_modes,),
        "pad_top": ("INT", {"default": 100, "min": 0, "step": 1, "max": MAX_RESOLUTION}),
        "pad_right": ("INT", {"default": 100, "min": 0, "step": 1, "max": MAX_RESOLUTION}),
        "pad_bottom": ("INT", {"default": 100, "min": 0, "step": 1, "max": MAX_RESOLUTION}),
        "pad_left": ("INT", {"default": 100, "min": 0, "step": 1, "max": MAX_RESOLUTION}),
      },
    }

  RETURN_TYPES = ("FLOAT",)
  RETURN_NAMES = ("angle",)
  FUNCTION = "get_angle"
  CATEGORY = CATEGORY_NAME

  def get_angle(
        self, insightface, image, mask, rotate_mode,
        pad_top, pad_right, pad_bottom, pad_left
    ):

    p_x1, p_y1, p_x2, p_y2 = get_mask_bbox_with_padding(mask.squeeze(0), pad_top, pad_right, pad_bottom, pad_left)
    image = image[:, p_y1:p_y2, p_x1:p_x2]
    kps = get_kps_from_image(image, insightface)

    angle = 0.
    if rotate_mode != "none" :
      angle = get_angle(
        kps[0], kps[1],
        round_angle = True if rotate_mode == "loseless" else False
      )
    return (angle,)


#==============================================================================
class AngleFromKps:
  rotate_modes = ["none", "loseless", "any"]
  def __init__(self):
      pass

  @classmethod
  def INPUT_TYPES(self):
    return {
      "required": {
        "kps_data": ("KPS_DATA",),
        "rotate_mode": (self.rotate_modes,)
      },
    }

  RETURN_TYPES = ("FLOAT",)
  RETURN_NAMES = ("angle",)
  FUNCTION = "get_angle"
  CATEGORY = CATEGORY_NAME

  def get_angle(self, kps_data, rotate_mode):
    kps_data = json.loads(kps_data)
    angle = 0.
    if rotate_mode != "none" :
      angle = get_angle(
        kps_data['array'][0], kps_data['array'][1],
        round_angle = True if rotate_mode == "loseless" else False
      )
    return (angle,)


#==============================================================================
class ComposeRotated:
  def __init__(self):
    pass

  @classmethod
  def INPUT_TYPES(self):
    return {
      "required": {
        "original_image": ("IMAGE",),
        "rotated_image": ("IMAGE",),
      }
  }

  RETURN_TYPES = ("IMAGE",)
  RETURN_NAMES = ("image",)
  FUNCTION = "compose_rotate"
  CATEGORY = CATEGORY_NAME

  def compose_rotate(self, original_image, rotated_image):
    original_width, original_height = original_image.shape[2], original_image.shape[1]
    rotated_width, rotated_height = rotated_image.shape[2], rotated_image.shape[1]

    if rotated_width != original_width:
      pad_x1 = (rotated_width - original_width) // 2
      pad_x2 = pad_x1 * -1
    else:
      pad_x1 = 0
      pad_x2 = original_width

    if rotated_height != original_height:
      pad_y1 = (rotated_height - original_height) // 2
      pad_y2 = pad_y1 * -1
    else:
      pad_y1 = 0
      pad_y2 = original_height

    image = rotated_image[:, pad_y1:pad_y2, pad_x1:pad_x2, :]
    return (image,)


#==============================================================================
class RotateImage:
  @classmethod
  def INPUT_TYPES(self):
    return {
      "required": {
        "image": ("IMAGE",),
        "angle": ("FLOAT", {"default": 0.0, "min": -360.0, "step": 0.1, "max": 360.0},),
        "counter_clockwise": ("BOOLEAN", {"default": True},),
      }
    }

  RETURN_TYPES = ("IMAGE",)
  RETURN_NAMES = ("rotated_image", "rotated_mask",)
  FUNCTION = "rotate_and_pad_image"
  CATEGORY = CATEGORY_NAME

  def rotate_and_pad_image(self, image, angle, counter_clockwise):
    if angle == 0 or angle == 360:
      return (image,)

    image = image_rotate_with_pad(image, counter_clockwise, angle)

    return (image,)
  

#==============================================================================
class LoadInstantIdAdapter:
  def __init__(self):
    pass

  @classmethod
  def INPUT_TYPES(self):
    return {
      "required": {
        "ipadapter":  (folder_paths.get_filename_list("ipadapter"), { "tooltip": "The default folder where the adapter is searched for is: models/ipadapter." }),
      }
  }

  RETURN_TYPES = ("INSTANTID_ADAPTER", "RESAMPLER", )
  RETURN_NAMES = ("InstantId_adapter", "resampler",)
  FUNCTION = "load_instantId_adapter"
  CATEGORY = CATEGORY_NAME

  def load_instantId_adapter(self, ipadapter):
    ipadapter_path = folder_paths.get_full_path("ipadapter", ipadapter)
    model = comfy.utils.load_torch_file(ipadapter_path, safe_load=True)
    instantId = InstantId(model['ip_adapter'])

    resampler = Resampler(
      dim=1280,
      depth=4,
      dim_head=64,
      heads=20,
      num_queries=16,
      embedding_dim=512,
      output_dim=2048,
      ff_mult=4
    )
    resampler.load_state_dict(model["image_proj"])
    return (instantId, resampler)


#==============================================================================
class InstantIdAdapterApply:
  def __init__(self):
    pass

  @classmethod
  def INPUT_TYPES(self):
    return {
      "required": {
        "model": ("MODEL", ),
        "instantId_adapter": ("INSTANTID_ADAPTER", ),
        "face_conditioning": ("FACE_CONDITIONING", ),
        "strength": ("FLOAT", {"default": 0.8, "min": 0, "step": 0.1, "max": 10},),
      }
    }

  RETURN_TYPES = ("MODEL",)
  RETURN_NAMES = ("model",)
  FUNCTION = "apply_instantId_adapter"
  CATEGORY = CATEGORY_NAME

  def apply_instantId_adapter(self, model, instantId_adapter, face_conditioning, strength):
    if strength == 0: return (model,)

    instantId = instantId_adapter.to(comfy.model_management.get_torch_device())
    patch_kwargs = {
      "instantId": instantId,
      "scale": strength,
      "cond": face_conditioning,
      "number": 0
    }

    m = model.clone()

    for id in [4,5,7,8]:
      block_indices = range(2) if id in [4, 5] else range(10)
      for index in block_indices:
        set_model_patch_replace(m, patch_kwargs, ("input", id, index))
        patch_kwargs["number"] += 1
      block_indices = range(2) if id in [3, 4, 5] else range(10)
      for index in block_indices:
        set_model_patch_replace(m, patch_kwargs, ("output", id, index))
        patch_kwargs["number"] += 1
    for index in range(10):
      set_model_patch_replace(m, patch_kwargs, ("middle", 1, index))
      patch_kwargs["number"] += 1

    return (m,)


#==============================================================================
# based on ControlNetApplyAdvance from ComfyUi/nodes.py
class ControlNetInstantIdApply:
  @classmethod
  def INPUT_TYPES(self):
    return {
      "required": {
        "positive": ("CONDITIONING", ),
        "negative": ("CONDITIONING", ),
        "face_conditioning": ("FACE_CONDITIONING", ),
        "control_net": ("CONTROL_NET", ),
        "image": ("IMAGE", ),
        "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01})
      }
    }

  RETURN_TYPES = ("CONDITIONING", "CONDITIONING", )
  RETURN_NAMES = ("positive", "negative",)
  FUNCTION = "apply_controlnet"
  CATEGORY = CATEGORY_NAME

  def apply_controlnet(self, positive, negative, face_conditioning, control_net, image, strength):
    if strength == 0:
        return (positive, negative)

    control_hint = image.movedim(-1,1)
    cnets = {}

    out = []
    for conditioning, isPositive in zip([positive, negative], [True, False]):
      c = []
      for t in conditioning:
        d = t[1].copy()

        prev_cnet = d.get("control", None)
        if prev_cnet in cnets:
          c_net = cnets[prev_cnet]
        else:
          c_net = control_net.copy().set_cond_hint(control_hint, strength)
          c_net.set_previous_controlnet(prev_cnet)
          cnets[prev_cnet] = c_net

        if isPositive:
          d["cross_attn_controlnet"] = face_conditioning.to(comfy.model_management.intermediate_device())
        else :
          d["cross_attn_controlnet"] = torch.zeros_like(face_conditioning).to(comfy.model_management.intermediate_device())
        d["control"] = c_net
        d["control_apply_to_uncond"] = False

        n = [t[0], d]
        c.append(n)
      out.append(c)
    return (out[0], out[1],)


#==============================================================================
class InstantIdAndControlnetApply:
  def __init__(self):
    pass

  @classmethod
  def INPUT_TYPES(self):
    return {
      "required": {
        "model": ("MODEL", ),
        "ipadapter_path":  (folder_paths.get_filename_list("ipadapter"), { "tooltip": "The default folder where the adapter is searched for is: models/ipadapter." }),
        "control_net_name": (folder_paths.get_filename_list("controlnet"), ),
        "face_embed": ("FACE_EMBED", ),
        "control_image": ("IMAGE", ),
        "adapter_strength": ("FLOAT", {"default": 0.5, "min": 0, "step": 0.1, "max": 10},),
        "control_net_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 10.0, "step": 0.01}),
        "positive": ("CONDITIONING", ),
        "negative": ("CONDITIONING", )
      }
    }

  RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING",)
  RETURN_NAMES = ("model", "positive", "negative",)
  FUNCTION = "apply_instantId_adapter_and_controlnet"
  CATEGORY = CATEGORY_NAME

  def apply_instantId_adapter_and_controlnet(
      self, model, ipadapter_path, control_net_name, face_embed, control_image,
      adapter_strength, control_net_strength, positive, negative
  ):
    graph = GraphBuilder()
    loadInstantIdAdapter = graph.node(
      "LoadInstantIdAdapter", ipadapter=ipadapter_path
    )
    faceEmbedCombine = graph.node(
      "FaceEmbedCombine", resampler=loadInstantIdAdapter.out(1), face_embeds=face_embed
     )
    loadControlNet = graph.node(
      "ControlNetLoader", control_net_name = control_net_name
    )
    instantIdApply = graph.node(
      "InstantIdAdapterApply", model=model, instantId_adapter=loadInstantIdAdapter.out(0),
      face_conditioning=faceEmbedCombine.out(0), strength=adapter_strength
    )
    controlNetInstantIdApply = graph.node(
      "ControlNetInstantIdApply", positive=positive, negative=negative,
      face_conditioning=faceEmbedCombine.out(0), control_net=loadControlNet.out(0),
      image=control_image, strength=control_net_strength
    )

    return {
      "result": (instantIdApply.out(0), controlNetInstantIdApply.out(0), controlNetInstantIdApply.out(1),),
      "expand":graph.finalize()
    }


#==============================================================================
class PreprocessImageAdvanced:
  resize_modes = ["auto", "free", "scale by width", "scale by height"]
  upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

  def __init__(self):
      pass

  @classmethod
  def INPUT_TYPES(self):
    return {
      "required": {
        "image": ("IMAGE", { "tooltip": "Pose image." }),
        "mask": ("MASK",),
        "width": ("INT", {"default": 1024, "min": 0, "step": 1, "max": MAX_RESOLUTION}),
        "height": ("INT", {"default": 1024, "min": 0, "step": 1, "max": MAX_RESOLUTION}),
        "resize_mode": (self.resize_modes,),
        "upscale_method": (self.upscale_methods,),
        "pad_top": ("INT", {"default": 100, "min": 0, "step": 1, "max": MAX_RESOLUTION}),
        "pad_right": ("INT", {"default": 100, "min": 0, "step": 1, "max": MAX_RESOLUTION}),
        "pad_bottom": ("INT", {"default": 100, "min": 0, "step": 1, "max": MAX_RESOLUTION}),
        "pad_left": ("INT", {"default": 100, "min": 0, "step": 1, "max": MAX_RESOLUTION}),
      },
      "optional": {
        "insightface": ("INSIGHTFACE_APP",),
      }
    }

  RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "INT", "INT", "INT", "INT", "INT", "INT",)
  RETURN_NAMES = ("resized_image", "mask", "control_image", "x", "y", "original_width", "original_height", "new_width", "new_height",)
  FUNCTION = "preprocess_image"
  CATEGORY = CATEGORY_NAME

  def preprocess_image(
        self, image, mask, width, height, resize_mode, upscale_method,
        pad_top, pad_right, pad_bottom, pad_left, insightface = None
    ):

    p_x1, p_y1, p_x2, p_y2 = get_mask_bbox_with_padding(mask.squeeze(0), pad_top, pad_right, pad_bottom, pad_left)
    mask = mask[:, p_y1:p_y2, p_x1:p_x2]
    image = image[:, p_y1:p_y2, p_x1:p_x2]
    kps = get_kps_from_image(image, insightface) if insightface else None
    _, original_height, original_width, _ = image.shape

    if resize_mode == "auto":
       width, height = resize_to_fit_area(int(p_x2 - p_x1), int(p_y2 - p_y1), width, height)
    else:
      if resize_mode != "free":
        ratio = original_width / original_height
        if resize_mode == "scale by width":
          height = int(width / ratio)
        if resize_mode == "scale by height":
          width = int(height * ratio)

    width = (width // 8) * 8
    height = (height // 8) * 8

    mask = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
    image = image.movedim(-1,1)
    mask = mask.movedim(-1,1)

    mask = comfy.utils.common_upscale(mask, width, height, "bilinear", "disabled")
    image = comfy.utils.common_upscale(image, width, height, upscale_method, "disabled")

    mask = mask.movedim(1,-1)
    mask = mask[:, :, :, 0]
    image = image.movedim(1,-1)
    _, new_height, new_width = mask.shape

    if kps is not None:
      kps *= [image.shape[2]  / original_width, image.shape[1] / original_height]
      control_image = draw_kps(width, height, kps)
      control_image = (torch.from_numpy(control_image).float() / 255.0).unsqueeze(0)

    return (
      image, mask,
      control_image if kps is not None else None,
      p_x1, p_y1, original_width, original_height,
      new_width, new_height,
    )


#==============================================================================
class PreprocessImage(PreprocessImageAdvanced):
  def __init__(self):
    pass

  @classmethod
  def INPUT_TYPES(self):
    return {
      "required": {
        "image": ("IMAGE",),
        "mask": ("MASK",),
        "width": ("INT", {"default": 1024, "min": 0, "step": 1, "max": MAX_RESOLUTION}),
        "height": ("INT", {"default": 1024, "min": 0, "step": 1, "max": MAX_RESOLUTION}),
        "resize_mode": (self.resize_modes,),
        "pad": ("INT", {"default": 100, "min": 0, "step": 1, "max": MAX_RESOLUTION}),
      },
      "optional": {
        "insightface": ("INSIGHTFACE_APP",),
      }
    }

  FUNCTION = "preprocess_image_simple"
  CATEGORY = CATEGORY_NAME

  def preprocess_image_simple(self, image, mask, width, height, resize_mode, pad, insightface = None):
    return self.preprocess_image(
       image, mask, width, height, resize_mode, "bilinear", pad, pad, pad, pad, insightface
    )


#==============================================================================
class LoadInsightface:
  def __init__(self):
    pass

  @classmethod
  def INPUT_TYPES(self):
    return {}

  RETURN_TYPES = ("INSIGHTFACE_APP",)
  RETURN_NAMES = ("insightface",)
  FUNCTION = "load_insightface"
  CATEGORY = CATEGORY_NAME

  def load_insightface(self):
    app = FaceAnalysis(
      name="antelopev2",
      root=INSIGHTFACE_PATH,
      providers=["CPUExecutionProvider", "CUDAExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return (app,)


#==============================================================================
class KpsDraw:
  @classmethod
  def INPUT_TYPES(self):
    return {
      "required": {
        "width": ("INT", {"default": 1024, "min": 0, "step": 1, "max": MAX_RESOLUTION}),
        "height": ("INT", {"default": 1024, "min": 0, "step": 1, "max": MAX_RESOLUTION}),
        "kps": ("HIDDEN_STRING_JSON", ),
      },
      "optional": {
        "image_reference": ("IMAGE",),
      }
    }

  RETURN_TYPES = ("KPS_DATA",)
  RETURN_NAMES = ("kps_data",)
  FUNCTION = "draw_kps"
  CATEGORY = CATEGORY_NAME

  def draw_kps(self, width, height, kps, image_reference = None):
    return (kps,)
  
 
#==============================================================================
class Kps3dFromImage:
  @classmethod
  def INPUT_TYPES(self):
    return {
      "required": {
        "width": ("INT", {"default": 1024, "min": 0, "step": 1, "max": MAX_RESOLUTION}),
        "height": ("INT", {"default": 1024, "min": 0, "step": 1, "max": MAX_RESOLUTION}),
        "kps": ("HIDDEN_STRING_JSON", ),
      },
       "optional": {
        "image": ("IMAGE",),
      }
    }

  RETURN_TYPES = ("KPS_DATA_3D", "KPS_DATA",)
  RETURN_NAMES = ("kps_data_3d", "kps_data")
  FUNCTION = "make_kps"
  CATEGORY = CATEGORY_NAME

  def make_kps(self, width, height, kps, image):

    kps_2d = kps3d_to_kps2d(json.loads(kps))

    return (kps, json.dumps(kps_2d),)


#==============================================================================
class KpsMaker:
  @classmethod
  def INPUT_TYPES(self):
    return {
      "required": {
        "kps_data": ("KPS_DATA",),
      }
    }

  RETURN_TYPES = ("IMAGE",)
  RETURN_NAMES = ("control_image",)
  FUNCTION = "draw_kps"
  CATEGORY = CATEGORY_NAME

  def draw_kps(self, kps_data):
    kps_data = json.loads(kps_data)

    control_image = draw_kps(kps_data['width'], kps_data["height"], kps_data["array"], alphas = kps_data["opacities"])
    control_image = (torch.from_numpy(control_image).float() / 255.0).unsqueeze(0)

    return (control_image, )
  

#==============================================================================
class Kps2dRandomizer:
  @classmethod
  def INPUT_TYPES(self):
    return {
      "required": {
        "kps_data": ("KPS_DATA",),
        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed used for randomizing KPS"}),
        "angle_min": ("INT", {"default": 0, "min": -180, "step": 1, "max": 180}),
        "angle_max": ("INT", {"default": 0, "min": -180, "step": 1, "max": 180}),
        "scale_min": ("FLOAT", {"default": 1, "min": 0.1, "step": 0.01, "max": 5}),
        "scale_max": ("FLOAT", {"default": 1, "min": 0.1, "step": 0.01, "max": 5}),
        "translate_x": ("INT", {"default": 0, "min": 0, "step": 1, "max": MAX_RESOLUTION}),
        "translate_y": ("INT", {"default": 0, "min": 0, "step": 1, "max": MAX_RESOLUTION}),
        "border": ("INT", {"default": 0, "min": 0, "step": 1, "max": MAX_RESOLUTION}),
      }
    }

  RETURN_TYPES = ("KPS_DATA",)
  RETURN_NAMES = ("kps_data",)
  FUNCTION = "rand_kps"
  CATEGORY = CATEGORY_NAME

  def rand_kps(self, kps_data, seed, angle_min, angle_max, scale_min, scale_max, translate_x, translate_y, border):

    torch.manual_seed(seed)
    kps_data = json.loads(kps_data)

    angle = 0
    scale = 1
    width = kps_data['width']
    height = kps_data['height']

    # get random angle
    if angle_min != 0 and angle_max != 0:
      angle = torch.randint(angle_min, angle_max + 1, (1,)).item()

    # get random scale
    if scale_min != 1 and scale_max != 1:
      scale = (scale_max - scale_min) * torch.rand(1).item() + scale_min

    # get random translate_x and translate_y
    if translate_x != 0:
        random_translate_x = torch.randint(-int(translate_x), int(translate_x) + 1, (1,)).item()
    else:
        random_translate_x = 0

    if translate_y != 0:
        random_translate_y = torch.randint(-int(translate_y), int(translate_y) + 1, (1,)).item()
    else:
        random_translate_y = 0

    # rotate
    if angle != 0:
      centroid = np.mean(np.array(kps_data["array"]), axis=0)
      angle_rad = np.radians(angle)

      rotated_points = []
      for x, y in kps_data["array"]:
        translated_x = x - centroid[0]
        translated_y = y - centroid[1]
        
        rotated_x = translated_x * np.cos(angle_rad) - translated_y * np.sin(angle_rad)
        rotated_y = translated_x * np.sin(angle_rad) + translated_y * np.cos(angle_rad)

        rotated_points.append([rotated_x + centroid[0], rotated_y + centroid[1]])

      kps_data["array"] = rotated_points
  
    # translate
    if random_translate_x != 0 or random_translate_y != 0:
        translated_points = []
        for x, y in kps_data["array"]:
            translated_points.append([x + random_translate_x, y + random_translate_y])
        kps_data["array"] = translated_points

    # scale
    if scale_min != 1 and scale_max != 1:
        scaled_points = []
        centroid = np.mean(np.array(kps_data["array"]), axis=0)
        for x, y in kps_data["array"]:
            scaled_points.append([
                centroid[0] + (x - centroid[0]) * scale,
                centroid[1] + (y - centroid[1]) * scale
            ])
        kps_data["array"] = scaled_points

    # check border
    x_values = [x for x, _ in kps_data["array"]]
    y_values = [y for _, y in kps_data["array"]]

    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)

    shift_x = 0
    shift_y = 0

    if min_x < border:
      shift_x = border - min_x 
    elif max_x > width - border:
      shift_x = (width - border) - max_x 

    if min_y < border:
        shift_y = border - min_y 
    elif max_y > height - border:
        shift_y = (height - border) - max_y 



    final_output = []
    for x, y in kps_data["array"]:
        shifted_x = x + shift_x
        shifted_y = y + shift_y
        final_output.append([int(shifted_x), int(shifted_y)])

    kps_data["array"] = final_output

    return (json.dumps(kps_data), )

#==============================================================================
class Kps3dRandomizer:
  @classmethod
  def INPUT_TYPES(self):
    return {
      "required": {
        "kps_data_3d": ("KPS_DATA_3D",),
        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed used for randomizing KPS"}),
        "rotate_x": ("INT", {"default": 0, "min": -180, "step": 1, "max": 180}),
        "rotate_y": ("INT", {"default": 0, "min": -180, "step": 1, "max": 180}),
        "rotate_z": ("INT", {"default": 0, "min": -180, "step": 1, "max": 180})
      }
    }

  RETURN_TYPES = ("KPS_DATA",)
  RETURN_NAMES = ("kps_data",)
  FUNCTION = "rand_kps"
  CATEGORY = CATEGORY_NAME

  def rand_kps(self, kps_data_3d, seed, rotate_x, rotate_y, rotate_z):
    torch.manual_seed(seed)
    kps_data = json.loads(kps_data_3d)

    angle_x = 0
    if rotate_x != 0:
      angle_x = torch.randint(-rotate_x, rotate_x + 1, (1,)).item()
    angle_y = 0
    if rotate_y != 0:
      angle_y = torch.randint(-rotate_y, rotate_y + 1, (1,)).item()
    angle_z = 0
    if rotate_x != 0:
      angle_z = torch.randint(-rotate_z, rotate_z + 1, (1,)).item()

    angle_x += kps_data['rotateX']
    angle_y += kps_data['rotateY']
    angle_z += kps_data['rotateZ']
    if angle_x != 0 or angle_y != 0 or angle_z != 0: 
      points = kps_rotate_3d(kps_data['array'], angle_x, angle_y, angle_z)
    else:
      points = kps_data['array']

    kps_data['array'] = points
    kps_data = kps3d_to_kps2d(kps_data)
  
    return (json.dumps(kps_data), )


#==============================================================================
class Kps2dScaleBy:
  @classmethod
  def INPUT_TYPES(self):
    return {
      "required": {
        "kps_data": ("KPS_DATA",),
        "scale": ("FLOAT", {"default": 1, "min": 0, "max": 100}),
      }
    }

  RETURN_TYPES = ("KPS_DATA",)
  RETURN_NAMES = ("kps_data",)
  FUNCTION = "scale_kps_by"
  CATEGORY = CATEGORY_NAME

  def scale_kps_by(self, kps_data, scale):
    kps_data = json.loads(kps_data)

    points = kps_data['array']
    kps_data['width'] = int(kps_data['width'] * scale)
    kps_data['height'] = int(kps_data['height'] * scale)
    for i, point in enumerate(points):
      kps_data['array'][i][0] = int(point[0] * scale)
      kps_data['array'][i][1] = int(point[1] * scale)
  
    return (json.dumps(kps_data), )


#==============================================================================
class Kps2dScale:
  @classmethod
  def INPUT_TYPES(self):
    return {
      "required": {
        "kps_data": ("KPS_DATA",),
        "width": ("INT", {"default": 1024, "min": 0, "max": MAX_RESOLUTION}),
        "height": ("INT", {"default": 1024, "min": 0, "max": MAX_RESOLUTION}),
      }
    }

  RETURN_TYPES = ("KPS_DATA",)
  RETURN_NAMES = ("kps_data",)
  FUNCTION = "scale_kps"
  CATEGORY = CATEGORY_NAME

  def scale_kps(self, kps_data, width, height):
    kps_data = json.loads(kps_data)

    points = kps_data['array']
    scaleX =  width / kps_data['width']
    scaleY =  height / kps_data['height']
    kps_data['width'] = int(kps_data['width'] * scaleX)
    kps_data['height'] = int(kps_data['height'] * scaleY)

    for i, point in enumerate(points):
      kps_data['array'][i][0] = int(point[0] * scaleX)
      kps_data['array'][i][1] = int(point[1] * scaleY)
  
    return (json.dumps(kps_data), )
  

#==============================================================================
class Kps2dRotate:
  @classmethod
  def INPUT_TYPES(self):
    return {
      "required": {
        "kps_data": ("KPS_DATA",),
        "angle": ("FLOAT", {"default": 0.0, "min": -360.0, "step": 0.1, "max": 360.0},),
        "counter_clockwise": ("BOOLEAN", {"default": True},),
      }
    }

  RETURN_TYPES = ("KPS_DATA",)
  RETURN_NAMES = ("kps_data",)
  FUNCTION = "rotate_kps"
  CATEGORY = CATEGORY_NAME

  def rotate_kps(self, kps_data, angle, counter_clockwise):
    if angle == 0 or angle == 360:
      return (kps_data,)

    if counter_clockwise: angle = -angle

    kps_data = json.loads(kps_data)

    points = kps_data['array']
    new_width, new_height = calculate_size_after_rotation(kps_data['width'], kps_data['height'], angle)

    kps_data['array'] = kps_rotate_2d(points, kps_data['width'], kps_data['height'], int(new_width), int(new_height), angle)
    kps_data['width'] = int(new_width)
    kps_data['height'] = int(new_height)

    return (json.dumps(kps_data), )
  
#==============================================================================
class Kps2dCrop:
  @classmethod
  def INPUT_TYPES(self):
    return {
      "required": {
        "kps_data": ("KPS_DATA",),
        "x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
        "y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
        "width": ("INT", {"default": 1024, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
        "height": ("INT", {"default": 1024, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
      }
    }

  RETURN_TYPES = ("KPS_DATA",)
  RETURN_NAMES = ("kps_data",)
  FUNCTION = "crop_kps"
  CATEGORY = CATEGORY_NAME

  def crop_kps(self, kps_data, x, y, width, height):
    kps_data = json.loads(kps_data)

    kps_data['width'] = width
    kps_data['height'] = height

    points = kps_data['array']

    for i, point in enumerate(points):
      kps_data['array'][i][0] = point[0] - x#
      kps_data['array'][i][1] = point[1] - y#

    return (json.dumps(kps_data), )
  

#==============================================================================
class MaskFromKps:
  @classmethod
  def INPUT_TYPES(self):
    return {
      "required": {
        "kps_data": ("KPS_DATA",),
        "grow_by": ("INT", {"default": 4, "min": 1, "max": 10, "step": 1}),
      }
    }

  RETURN_TYPES = ("MASK",)
  RETURN_NAMES = ("mask",)
  FUNCTION = "creat_mask"
  CATEGORY = CATEGORY_NAME

  def creat_mask(self, kps_data, grow_by):
    kps_data = json.loads(kps_data)
    bbox = get_bbox_from_kps(kps_data, grow_by)
    mask = torch.zeros((kps_data['height'], kps_data['width']))
    mask[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] = 1

    return (mask.unsqueeze(0), )

NODE_CLASS_MAPPINGS = {
  "LoadInsightface": LoadInsightface,
  "LoadInstantIdAdapter": LoadInstantIdAdapter,
  "InstantIdAdapterApply": InstantIdAdapterApply,
  "ControlNetInstantIdApply": ControlNetInstantIdApply,
  "InstantIdAndControlnetApply": InstantIdAndControlnetApply,
  "PreprocessImage": PreprocessImage,
  "PreprocessImageAdvanced": PreprocessImageAdvanced,
  "AngleFromFace": AngleFromFace,
  "AngleFromKps": AngleFromKps,
  "RotateImage": RotateImage,
  "ComposeRotated": ComposeRotated,
  "KpsDraw": KpsDraw,
  "Kps3dFromImage": Kps3dFromImage,
  "KpsMaker": KpsMaker,
  "Kps2dRandomizer": Kps2dRandomizer,
  "Kps3dRandomizer": Kps3dRandomizer,
  "KpsScale": Kps2dScale,
  "KpsScaleBy": Kps2dScaleBy,
  "KpsRotate": Kps2dRotate,
  "KpsCrop": Kps2dCrop,
  "MaskFromKps": MaskFromKps,
  "FaceEmbed": FaceEmbed,
  "FaceEmbedCombine": FaceEmbedCombine

}

NODE_DISPLAY_NAME_MAPPINGS = {
  "LoadInsightface": "Load insightface",
  "LoadInstantIdAdapter": "Load instantId adapter",
  "InstantIdAdapterApply": "Apply instantId adapter",
  "ControlNetInstantIdApply": "Apply instantId ControlNet",
  "InstantIdAndControlnetApply": "Apply instantId and ControlNet",
  "PreprocessImage": "Preprocess image for instantId",
  "PreprocessImagAdvancese": "Preprocess image for instantId (Advanced)",
  "AngleFromFace": "Get Angle from face",
  "AngleFromKps": "Get Angle from KPS data",
  "RotateImage": "Rotate Image",
  "ComposeRotated": "Remove rotation padding",
  "KpsDraw": "Draw KPS",
  "Kps3dFromImage": "3d KPS from image",
  "KpsMaker": "Create KPS Image",
  "Kps2dRandomizer": "Randomize 2d KPS",
  "Kps3dRandomizer": "Randomize 3d KPS",
  "Kps2dScaleBy": "Scale 2d KPS by",
  "Kps2dScale": "Scale 2d KPS",
  "KpsRotate": "Rotate 2d KPS",
  "KpsCrop": "Crop 2d KPS",
  "MaskFromKps": "Create mask from Kps",
  "FaceEmbed": "FaceEmbed for instantId",
  "FaceEmbedCombine": "FaceEmbed Combine"
}