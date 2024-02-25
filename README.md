# ComfyUI InstantID Faceswapper v0.0.5
<sub>[About](#comfyui-instantid-faceswapper) | [Installation guide](#installation-guide) | [Custom nodes](#custom-nodes) | [Workflows](#workflows) | [Workflow script](#workflow-script-beta) | [Tips](#tips) | [Changelog](#changelog)</sub>

Implementation of [faceswap](https://github.com/nosiu/InstantID-faceswap/tree/main) based on [InstantID](https://github.com/InstantID/InstantID) for ComfyUI. \
Allows usage of [LCM Lora](https://huggingface.co/latent-consistency/lcm-lora-sdxl) which can produce good results in only a few generation steps.
</br>
**Works ONLY with SDXL checkpoints.**
</br>
</br>
![image](https://github.com/nosiu/comfyui-instantId-faceswap/assets/5691179/b69e11cf-ea77-4f41-95cc-c0ea84269e7b)
![image](https://github.com/nosiu/comfyui-instantId-faceswap/assets/5691179/597a0b1d-21fd-44ac-945a-9df4fd73eda4)



## Installation guide
<sub>[About](#comfyui-instantid-faceswapper) | [Installation guide](#installation-guide) | [Custom nodes](#custom-nodes) | [Workflows](#workflows) | [Workflow script](#workflow-script-beta) | [Tips](#tips) | [Changelog](#changelog)</sub>

1. Clone or download this repository and put it into **ComfyUI/custom_nodes**
2. Open commandline in the  **ComfyUI/custom_nodes/comfyui-instantId-faceswap/** folder and type `pip install -r requirements.txt` to install dependencies
3. Manually download required files and create required folders:
    - [antelopev2 models](https://huggingface.co/DIAMONIK7777/antelopev2/tree/main)
      and put them into **ComfyUI/models/insightface/models/antelopev2** folder
       -  1k3d68.onnx
       -  2d106det.onnx
       -  genderage.onnx
       -  glintr100.onnx
       -  scrfd_10g_bnkps.onnx

    - [IpAdapter and ControlNet](https://huggingface.co/InstantX/InstantID/tree/main)
       - ip-adapter.bin - put it into **ComfyUI/models/ipadapter**
       - ControlNetModel/diffusion_pytorch_model.safetensors - put it into **ComfyUI/models/controlnet/ControlNetModel**
       - ControlNetModel/config.json - put it into **ComfyUI/models/controlnet/ControlNetModel**

    - [LCM Lora](https://huggingface.co/latent-consistency/lcm-lora-sdxl/tree/main) *Optional (but higly recomended)
       - pytorch_lora_weights.safetensors - put it into **ComfyUI/models/loras**

Newly added files hierarchy should look like this:
```
ComfyUI
\---models
    \---ipadapter
           ipadapter.bin
    \---controlnet
        \---ControlNetModel
              config.json
              diffusion_pytorch_model.safetensors
    \---insightface
        \---models
            \antelopev2
                  1k3d68.onnx
                  2d106det.onnx
                  genderage.onnx
                  glintr100.onnx
                  scrfd_10g_bnkps.onnx
```

*Note You don't need to add the 'ipadapter', 'controlnet', and 'lora' folders to this specific location if you already have them somewhere else.
Instead, You can edit `ComfyUI/extra_model_paths.yaml` and add folders containing those files to the config.

## Custom nodes
<sub>[About](#comfyui-instantid-faceswapper) | [Installation guide](#installation-guide) | [Custom nodes](#custom-nodes) | [Workflows](#workflows) | [Workflow script](#workflow-script-beta) | [Tips](#tips) | [Changelog](#changelog)</sub>

### Faceswap LCM Lora
   params:
   - **lcm_lora** - path to your LCM lora inside the folder with Loras

### Faceswap setup
   Loads everything and sets up the pipeline \
   params:
   - **checkpoint** - your SDXL checkpoint (do not use checkpoints for inpainting!)
   - **controlnet** - folder where your ControlNetModel is located
   - **controlnet_name** - folder with ControlNetModel if you renamed the original folder then type a new name here if you followed the instructions then leave "/ControlNetModel"
   - **ipadapter** - ip adapter from the instruction

### Faceswap face embed
   Prepares face embeds for the generation, you can chain face embeds to improve results \
   params:
   - **face_image** - input image from which to extract embed data
   - **face_embed** - face_embed(s)

### Faceswap generate
   Generates new face from input Image based on input mask \
   params:
   - **padding** - how much the image region sent to the pipeline will be enlarged by mask bbox with padding.
   - **ip_adapter_scale** - strength of ip adapter.
   - **controlnet conditioning scale** - strength of controlnet.
   - **guidance_scale** - guidance scale value encourages the model to generate images closely linked to the text prompt at the expense of lower image quality. Guidance scale is enabled when `guidance_scale` > 1.
   - **steps** - how many steps generation will take
   - **resize** - Maximum value to which the cut region of the image will be scaled. Larger values should yield better results but will be slower. To disable, select `don't`.
   - **mask_strength** - strength of mask.
   - **blur_mask** - how much blur add to a mask before composing it into the the result picture.
   - **rotate_face** - This option rotates the image before processing and rotates it back afterward to keep the face straight.
      - **loseless** - This option rotates the image by multiples of 90 angles (90, 180, 270). You shouldn't lose any quality.
      - **always** - This option rotates the image by any angle in an attempt to keep the face straight.
      **Note:** You will lose some quality of the original image.
      - **don't** - This option does nothing; it won't rotate the image at all.
   - **offload** - if you are experiencing memory issues during the decoding process, this option might help.
      - **don't** - do nothing. This is the fastest option and does not move memory.
      - **before decoding** - moves a significant amount of memory from VRAM to RAM to save memory before decoding. Use this option if you are running out of memory after reaching 100% during generation.
      - **at the end** - same as **before decoding** and moves the entire pipeline into RAM after decoding. Use this option if you are doing something else besides faceswapping.
   - **seed** - seed send to pipeline
   - **control_after_generate** - what to do with seed
   - **positive** - positive prompts
   - **negative** - negative prompts, works only when `guidance_scale` > 1
   - **negative2** - negative prompts

## Workflows
<sub>[About](#comfyui-instantid-faceswapper) | [Installation guide](#installation-guide) | [Custom nodes](#custom-nodes) | [Workflows](#workflows) | [Workflow script](#workflow-script-beta) | [Tips](#tips) | [Changelog](#changelog)</sub>

You can find example workflows in the /workflows folder.

## Workflow script (beta)
<sub>[About](#comfyui-instantid-faceswapper) | [Installation guide](#installation-guide) | [Custom nodes](#custom-nodes) | [Workflows](#workflows) | [Workflow script](#workflow-script-beta) | [Tips](#tips) | [Changelog](#changelog)</sub>

The simple script, `workflow_generate.py`, can generate workflows based on the face images contained in a specific folder. The script will automatically generate appropriate nodes and connect them together. Keep in mind that there is no validation to check if there is a face in the image.

You can copy the `workflow_generate.py` script anywhere you want for easier access; it has no dependencies inside the custom_node.

Only files with extensions: jpg, jpeg, bmp, png, gif, webp, and jiff will be included in the workflow.

The script will not upload reference images into the `ComfyUI/input` folder. As a result, **you won't be able to preview those images.**

If you move, rename, delete image files, or modify paths in any way, the workflow will stop working.

**You may see warnings (errors) in the console while loading generated workflows, ignore those.**
### Usage
arguments:
- input folder absolute path
- (optional) output workflow file name (default: "workflow")

### Example
This command will generate 'albert.json' workflow, which should include all the required nodes for face reference images in the 'C:\Users\Admin\Desktop\ALBERT' folder.

```
workflow_generate.py C:\Users\Admin\Desktop\ALBERT albert
```
![image](https://github.com/nosiu/comfyui-instantId-faceswap/assets/5691179/1b0f0306-5207-4447-9844-d148aa234450)

## Tips
<sub>[About](#comfyui-instantid-faceswapper) | [Installation guide](#installation-guide) | [Custom nodes](#custom-nodes) | [Workflows](#workflows) | [Workflow script](#workflow-script-beta) | [Tips](#tips) | [Changelog](#changelog)</sub>

- If instead of face you are getting artifacts try using **resize** option with a high value if that doesn't help, try reducing the **padding** parameter.
- By using LCM lora you can generate good images in 2-3 steps (10+ otherwise).
- If the result is too different in color from the original, try reducing the number of steps and/or the ip_adapter_scale value.
- If you get `No face detected in pose image` error try to increase padding. It means the current mask + padding is not enough to detect the face in the input image by insightface.

## Changelog
<sub>[About](#comfyui-instantid-faceswapper) | [Installation guide](#installation-guide) | [Custom nodes](#custom-nodes) | [Workflows](#workflows) | [Workflow script](#workflow-script-beta) | [Tips](#tips) | [Changelog](#changelog)</sub>

- ### 0.0.5 (25.02.2024)
   - The `mask_strength` parameter has been fixed; it now functions correctly. Previously, it was stuck at *0.9999* regardless of the chosen value.
   - The `ip_adapter_scale` parameter has been fixed. If you were using the xformers, this parameter could be stuck at *50*.
   - Changed the method of processing face_embed(s).
   - Added the `rotate_face` parameter. It will attempt to rotate the image to keep the face straight before processing and rotate it back to the original position afterward.

- ### 0.0.4 (14.02.2024)
   - To save memory, you can run Comfy with the `--fp16-vae` argument to disable the default VAE upcasting to float32.
   - Merged the old `resize` and `resize_to` options into just `resize` for the Faceswap generate node. To emulate the old behavior where resize was unchecked, select `don't`.
   - Added a manual offload mechanism to save GPU memory.
   - Changed the minimum and maximum values for `mask_strength` to range from 0.00 to 1.00.
- ### 0.0.3 (07.02.2024)
   - Fixed an error that caused new face_embeds to be added when editing previous ones
- ### 0.0.2 (05.02.2024)
   - Introducing workflow generator script - [more information here](#workflow-script-beta)
   - Updating the dependency diffusers to version 0.26.x. Run either:
   ```bash
   pip install -r requirements.txt
   ```
   or
   ```bash
   pip install -U diffusers~=0.26.0
   ```

- ### 0.0.1 (01.02.2024)
  - Progress bar and latent preview added for generation node

