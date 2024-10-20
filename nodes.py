import os
import cv2
import torch
import numpy as np
import folder_paths
import node_helpers
import comfy.utils
from PIL import Image
from .ip_adapter.resampler import Resampler
from .ip_adapter.instantId import InstantId
from insightface.app import FaceAnalysis
from comfy_execution.graph_utils import GraphBuilder
from .utils import draw_kps, set_model_patch_replace, resize_to_fit_area, get_mask_bbox_with_padding, get_kps_from_image, get_angle, rotate_with_pad


folder_paths.folder_names_and_paths["ipadapter"] = ([os.path.join(folder_paths.models_dir, "ipadapter")], folder_paths.supported_pt_extensions)
INSIGHTFACE_PATH = os.path.join(folder_paths.models_dir, "insightface")
CATEGORY_NAME = "InstantId Faceswap"
MAX_RESOLUTION = 16384


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
class KpsMaker:
  @classmethod
  def INPUT_TYPES(self):
    return {
      "required": {
        "image": ("STRING",),
        "width": ("INT", {"default": 1024, "min": 0, "step": 1, "max": MAX_RESOLUTION}),
        "height": ("INT", {"default": 1024, "min": 0, "step": 1, "max": MAX_RESOLUTION}),
      },
      "optional": {
        "image_reference": ("IMAGE",),
      }
    }

  RETURN_TYPES = ("IMAGE", "MASK",)
  RETURN_NAMES = ("control_image", "mask",)
  FUNCTION = "draw_kps"
  CATEGORY = CATEGORY_NAME

  def draw_kps(self, image, width, height, image_reference = None):
    if "clipspace" not in image:
      image_path = os.path.join(
        folder_paths.get_input_directory(),
        "faceswap_controls",
        image
      )
    else: # with mask - saved in different directory
      image_path = os.path.join(
        folder_paths.get_input_directory(),
        image[:-8]
      )

    pil_image = node_helpers.pillow(Image.open, image_path)
    image = pil_image.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0)

    if "A" in pil_image.getbands():
      mask = np.array(pil_image.getchannel("A")).astype(np.float32) / 255.0
      mask = 1. - torch.from_numpy(mask)
    else:
      mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
    mask = mask.unsqueeze(0)
    return (image, mask,)


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

    image = rotate_with_pad(image, counter_clockwise, angle)
    return (image,)


NODE_CLASS_MAPPINGS = {
  "LoadInsightface": LoadInsightface,
  "LoadInstantIdAdapter": LoadInstantIdAdapter,
  "InstantIdAdapterApply": InstantIdAdapterApply,
  "ControlNetInstantIdApply": ControlNetInstantIdApply,
  "InstantIdAndControlnetApply": InstantIdAndControlnetApply,
  "PreprocessImage": PreprocessImage,
  "PreprocessImageAdvanced": PreprocessImageAdvanced,
  "AngleFromFace": AngleFromFace,
  "RotateImage": RotateImage,
  "ComposeRotated": ComposeRotated,
  "KpsMaker": KpsMaker,
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
  "RotateImage": "Rotate Image",
  "ComposeRotated": "Remove rotation padding",
  "KpsMaker": "Draw KPS",
  "FaceEmbed": "FaceEmbed for instantId",
  "FaceEmbedCombine": "FaceEmbed Combine"
}