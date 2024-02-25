from diffusers.models import ControlNetModel
from diffusers import LCMScheduler
from PIL import Image, ImageFilter
from time import perf_counter
import os
import cv2
import torch
import numpy as np
import folder_paths
import comfy.utils
from math import atan2, pi, ceil
from comfy.cli_args import args, LatentPreviewMethod
from comfy.latent_formats import SDXL
from comfy.model_management import xformers_enabled, vae_dtype, get_free_memory
from latent_preview import get_previewer
from insightface.app import FaceAnalysis
from .pipeline_stable_diffusion_xl_instantid_inpaint import StableDiffusionXLInstantIDInpaintPipeline, draw_kps

folder_paths.folder_names_and_paths["ipadapter"] = ([os.path.join(folder_paths.models_dir, "ipadapter")], folder_paths.supported_pt_extensions)
INSIGHTFACE_PATH = os.path.join(folder_paths.models_dir, "insightface")

device = "cuda" if torch.cuda.is_available() else "cpu"

CATEGORY_NAME = "InstantId Faceswap"

DEBUG = False
def debug(msg):
    if DEBUG: print(msg)

def image_to_tensor(image):
    img_array = np.array(image)
    # add batch dim and normalise values to 0 - 1
    img_tensor = (torch.from_numpy(img_array).float() / 255.0).unsqueeze(0)
    return img_tensor

def tensor_to_numpy(tensor):
    # squeeze batch dim and normalise values to 0 - 255
    return (255.0 * tensor.cpu().numpy().squeeze()).clip(0, 255).astype(np.uint8)

def resize_img(input_image, max_side=1280, min_side=1024,
               mode=Image.BILINEAR, base_pixel_number=64):

    if not isinstance(input_image, Image.Image): # Tensor to PIL.Image
        input_image = Image.fromarray(tensor_to_numpy(input_image))

    w, h = input_image.size

    ratio = min_side / min(h, w)
    w, h = round(ratio*w), round(ratio*h)
    ratio = max_side / max(h, w)
    input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
    w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
    h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    return input_image

def prepareMaskAndPoseAndControlImage(pose_image, mask_image, insightface, padding = 50, resize = 1280, blur_mask = 10):
        p_x1, p_y1, p_x2, p_y2 = getMaskBboxWithPadding(mask_image, padding)

        image = pose_image[p_y1:p_y2, p_x1:p_x2]
        mask_image = mask_image[p_y1:p_y2, p_x1:p_x2]

        original_height, original_width, _ = image.shape

        kps = getKpsFromImage(image, insightface)

        mask = Image.fromarray(mask_image.astype(np.uint8))
        image = Image.fromarray(image.astype(np.uint8))

        mask_blur_offset = int(blur_mask / 4) if blur_mask > 0 else 0
        resized_mask = mask.resize((original_width - int(blur_mask), original_height - int(blur_mask)))
        mask_with_blur = Image.new("RGB", (original_width, original_height), (0, 0, 0))
        mask_with_blur.paste(resized_mask, (mask_blur_offset, mask_blur_offset))
        mask_with_blur = mask_with_blur.filter(ImageFilter.GaussianBlur(radius = blur_mask))
        mask = mask_with_blur.convert("L")

        if resize != "Don't":
            resize = int(resize)
            mask = resize_img(mask, resize)
            image = resize_img(image, resize)
            new_width, new_height = image.size
            kps *= [new_width / original_width, new_height / original_height]
        control_image = draw_kps(image, kps)

        # (mask, pose, control), (original positon face + padding: x, y, w, h)
        return (mask, image, control_image), (p_x1, p_y1, original_width, original_height)

def getMaskBboxWithPadding(mask_image, padding):
    mask_segments = np.where(mask_image == 255)
    m_x1 = int(np.min(mask_segments[1]))
    m_x2 = int(np.max(mask_segments[1]))
    m_y1 = int(np.min(mask_segments[0]))
    m_y2 = int(np.max(mask_segments[0]))

    height = mask_image.shape[0]
    width = mask_image.shape[1]

    p_x1 = max(0, m_x1 - padding)
    p_y1 = max(0, m_y1 - padding)
    p_x2 = min(width, m_x2 + padding)
    p_y2 = min(height,m_y2 + padding)

    return int(p_x1), int(p_y1), int(p_x2), int(p_y2)

def getAngle(a=(0, 0), b=(0, 0), round_angle=False):
    angle = atan2(b[1] - a[1], b[0] - a[0]) * 180 / pi
    if not round_angle: return angle
    rounded_angle = round(angle / 90) * 90
    if rounded_angle == 360: rounded_angle = 0
    return rounded_angle

def getKpsFromImage(np_pose_image, insightface):
    face_info = insightface.get(cv2.cvtColor(np_pose_image, cv2.COLOR_RGB2BGR))
    assert len(face_info) > 0, "No face detected in pose image"
    face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1] # only use the maximum face
    return face_info['kps']

class FaceEmbed:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
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

    def make_face_embed(self, insightface, face_image, face_embeds = ()):
        face_image = tensor_to_numpy(face_image)
        face_info = insightface.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
        assert len(face_info) > 0, "No face detected for face embed"
        face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1] # only use the maximum face
        face_emb = face_info['embedding']
        face_embeds = (*face_embeds, face_emb)
        return (face_embeds,)

class LoadLCMLora:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lcm_lora": (folder_paths.get_filename_list("loras"), ),
            }
        }

    RETURN_TYPES = ("LCM_LORA",)
    RETURN_NAMES = ("LCM Lora",)
    FUNCTION = "load_lcm_lora"
    CATEGORY = CATEGORY_NAME

    def load_lcm_lora(self, lcm_lora):
        lora = folder_paths.get_full_path("loras", lcm_lora)
        return [lora]

class SetupPipeline:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "checkpoint": (folder_paths.get_filename_list("checkpoints"),),
                "controlnet": (folder_paths.get_folder_paths("controlnet"),),
                "controlnet_name": ("STRING", {"multiline": False, "forceInput": False, "default": "/ControlNetModel"}),
                "ipadapter":  (folder_paths.get_filename_list("ipadapter"),),
            },
            "optional": {
                "LCM_lora": ("LCM_LORA",)
            }
        }
    RETURN_TYPES = ("FACESWAP_PIPE", "INSIGHTFACE_APP",)
    RETURN_NAMES = ("inpaint pipe", "insightface",)
    FUNCTION = "create_pipeline"
    CATEGORY = CATEGORY_NAME

    def create_pipeline(self, checkpoint, controlnet, controlnet_name, ipadapter, LCM_lora = None):
        checkpoint = folder_paths.get_full_path("checkpoints", checkpoint)
        controlnet_full_path = controlnet + controlnet_name
        controlnet = folder_paths.get_full_path("controlnet", controlnet_name)
        ipadapter = folder_paths.get_full_path("ipadapter", ipadapter)


        app = FaceAnalysis(name="antelopev2",
                           root=INSIGHTFACE_PATH,
                           providers=['CPUExecutionProvider', 'CUDAExecutionProvider']
        )

        app.prepare(ctx_id=0, det_size=(640, 640))

        controlnet = ControlNetModel.from_pretrained(
            controlnet_full_path,
            torch_dtype=torch.float16
        )

        pipe = StableDiffusionXLInstantIDInpaintPipeline.from_single_file(
            checkpoint,
            controlnet=controlnet,
            torch_dtype=torch.float16 ,
            use_safetensors=True
        )

        pipe.load_ip_adapter_instantid(ipadapter)

        if LCM_lora is not None:
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            pipe.load_lora_weights(LCM_lora)
            pipe.fuse_lora()

        if xformers_enabled(): pipe.enable_xformers_memory_efficient_attention()

        return (pipe, app,)


OFFLOAD_TYPES = {
    "NONE": "don't",
    "BEFORE_DECODING": "before decoding",
    "AT_THE_END": "at the end"
}

ROTATE_FACE_TYPES = {
    "LOSELESS": "loseless",
    "ALWAYS": "always",
    "NONE": "don't"
}

class GenerationInpaint:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "face_embeds": ("FACE_EMBED",),
                "inpaint_pipe": ("FACESWAP_PIPE",),
                "insightface": ("INSIGHTFACE_APP",),
                "padding": ("INT", {"default": 40, "min": 0, "max": 500, "step": 1, "display": "input"}),
                "ip_adapter_scale": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1, "round": 1, "display": "input"}),
                "controlnet_conditioning_scale": ("FLOAT", {"default": 0.8, "min": 0, "step": 0.1, "max": 1.0}),
                "guidance_scale": ("FLOAT", {"default": 0, "min": 0, "max": 10, "step": 0.1, "display": "input"}),
                "steps": ("INT", {"default": 2, "min": 1, "max": 100, "step": 1, "display": "slider"}),
                "mask_strength": ("FLOAT", {"default": 0.99, "min": 0.01, "max": 1.00, "step": 0.01, "round": 0.01, "display": "input"}),
                "blur_mask": ("INT", {"default": 0,  "min": 0, "max": 200, "step": 1,  "display": "input"}),
                "resize": (["1280", "1024", "768", "Don't"],),
                "rotate_face": ([
                    ROTATE_FACE_TYPES['NONE'],
                    ROTATE_FACE_TYPES['LOSELESS'],
                    ROTATE_FACE_TYPES['ALWAYS']
                ],),
                "offload": ([
                    OFFLOAD_TYPES["NONE"],
                    OFFLOAD_TYPES["BEFORE_DECODING"],
                    OFFLOAD_TYPES["AT_THE_END"]
                ],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),

            },
            "optional": {
                "positive": ("STRING", {"multiline": True, "forceInput": False}),
                "negative": ("STRING", {"multiline": True, "forceInput": False}),
                "negative2": ("STRING", {"multiline": True, "forceInput": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "faceswap"
    CATEGORY = CATEGORY_NAME

    def faceswap(
        self,
        image,
        mask,
        face_embeds,
        inpaint_pipe,
        insightface,
        padding,
        ip_adapter_scale,
        controlnet_conditioning_scale,
        guidance_scale,
        steps,
        mask_strength,
        blur_mask,
        resize,
        offload,
        rotate_face,
        seed,
        positive = "",
        negative = "",
        negative2 = ""
    ):
        mask_image = tensor_to_numpy(mask)
        pose_image = tensor_to_numpy(image)
        ip_adapter_scale = ip_adapter_scale / 100

        width = pose_image.shape[1]
        height = pose_image.shape[0]

        face_emb = np.concatenate(face_embeds)

        angle = None
        if rotate_face != ROTATE_FACE_TYPES['NONE']:
            p_x1, p_y1, p_x2, p_y2 = getMaskBboxWithPadding(mask_image, padding)
            im = pose_image[p_y1:p_y2, p_x1:p_x2]
            kps = getKpsFromImage(im, insightface)
            angle = getAngle(
                kps[0], kps[1],
                round_angle = True if rotate_face == ROTATE_FACE_TYPES['LOSELESS'] else False
            )
            debug("face rotation angle: " + str(angle))
            if angle != 0:
                pose_image = Image.fromarray(pose_image)
                pose_image = pose_image.rotate(angle, expand=True, fillcolor=(0,0,0,0))
                pose_image = np.array(pose_image)
                mask_image = Image.fromarray(mask_image).rotate(angle, expand=True, fillcolor=(0,))
                mask_image = np.array(mask_image)

        images, position = prepareMaskAndPoseAndControlImage(pose_image, mask_image, insightface, padding, resize, blur_mask)
        mask_image, ref_image, control_image = images

        generator = torch.Generator(device=device).manual_seed(seed)

        previewer = None

        if args.preview_method != LatentPreviewMethod.NoPreviews:
            previewer = get_previewer(device, SDXL())

        pbar = comfy.utils.ProgressBar(steps-1)

        def progress_fn(_, step, _1, dict):
            preview_bytes = None
            if previewer:
                preview_bytes = previewer.decode_latent_to_preview_image("JPEG", dict["latents"].float()) # first arg is unused
            pbar.update_absolute(step, steps-1, preview_bytes)
            return dict

        debug("Offload type: " + offload)
        debug("GPU memory before pipe: " + f"{(get_free_memory() / 1024 / 1024):.3f}" + " MB")

        t1 = perf_counter()
        inpaint_pipe.to(device)
        t2 = perf_counter()
        debug("moving pipe to GPU took: " + str(t2 - t1) + " s")

        latent = inpaint_pipe(
            prompt=positive,
            negative_prompt=negative,
            negative_prompt_2=negative2,
            image_embeds=face_emb,
            control_image=control_image,
            image=ref_image,
            #TODO control_mask (?)
            mask_image=mask_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            ip_adapter_scale=ip_adapter_scale,
            strength=mask_strength,
            num_inference_steps=int(ceil((steps / mask_strength))),
            generator=generator,
            guidance_scale=guidance_scale,
            callback_on_step_end=progress_fn
        )

        if offload != OFFLOAD_TYPES['NONE']:
            t1 = perf_counter()
            inpaint_pipe.unet.to("cpu")
            inpaint_pipe.controlnet.to("cpu")
            inpaint_pipe.text_encoder.to("cpu")
            inpaint_pipe.text_encoder_2.to("cpu")
            inpaint_pipe.image_proj_model.to("cpu")
            t2 = perf_counter()
            debug("moving UNET, CONTROLNET, TEXT_ENCODER, TEXT_ENCODER_2, IMAGE_PROJ_MODEL to CPU took: " + str(t2 - t1) + " s")
            torch.cuda.empty_cache()

        face = inpaint_pipe.decodeVae(latent, vae_dtype() == torch.float32)

        x, y, w, h = position

        resized_face = face.resize((w, h), resample=Image.LANCZOS)
        resized_mask = mask_image.resize((w, h))
        pose_image = Image.fromarray(pose_image)

        pose_image.paste(resized_face, (x, y), mask=resized_mask)

        if angle is not None:
            pose_image = pose_image.rotate(-angle, expand=True, fillcolor=(0,0,0,0), resample=Image.BICUBIC)
            new_width, new_height = pose_image.size
            # TODO CROP only  the original face and paste it into the original image?
            pose_image = pose_image.crop(box=(
                (new_width - width) / 2,
                (new_height - height) / 2,
                (new_width - (new_width - width) / 2),
                (new_height - (new_height - height) / 2)
            ))

        if offload == OFFLOAD_TYPES['AT_THE_END']:
            inpaint_pipe.to("cpu", silence_dtype_warnings=True)
            inpaint_pipe.image_proj_model.to("cpu")
            torch.cuda.empty_cache()

        debug("GPU memory at the end: " + f"{(get_free_memory() / 1024 / 1024):.3f}" + " MB")
        return (image_to_tensor(pose_image),)

NODE_CLASS_MAPPINGS = {
    "FaceSwapSetupPipeline": SetupPipeline,
    "FaceSwapGenerationInpaint": GenerationInpaint,
    "FaceEmbed": FaceEmbed,
    "LCMLora": LoadLCMLora,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceSwapSetupPipeline": "Faceswap setup",
    "FaceSwapGenerationInpaint": "Faceswap generate",
    "FaceEmbed": "Faceswap face embed",
    "LCMLora": "Faceswap LCM Lora",
}