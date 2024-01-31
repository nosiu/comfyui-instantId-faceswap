# ComfyUI InstantID Faceswapper
Implementation of [faceswap](https://github.com/nosiu/InstantID-faceswap/tree/main) based on [InstantID](https://github.com/InstantID/InstantID) for ComfyUI. \
Allows usage of [LCM Lora](https://huggingface.co/latent-consistency/lcm-lora-sdxl) which can produce good results in only a few generation steps. \
**Works ONLY with SDXL checkpoints.** \
![image](https://github.com/nosiu/comfyui-instantId-faceswap/assets/5691179/b69e11cf-ea77-4f41-95cc-c0ea84269e7b)
![image](https://github.com/nosiu/comfyui-instantId-faceswap/assets/5691179/597a0b1d-21fd-44ac-945a-9df4fd73eda4)



## Installation guide
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
*Note You don't need to add the 'ipadapter,' 'controlnet,' 'insightface' and 'lora' folders to this specific location if you already have them somewhere else.
Instead, You can edit `ComfyUI/extra_model_paths.yaml` and add folders containting those files to the config.

## Custom nodes
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
   - **padding** - how much the image region sent to the pipeline will be enlarged by (mask bbox with padding
   - **ip_adapter_scale** - strength of ip adapter
   - **controlnet conditioning scale** - strength of controlnet
   - **guidance_scale** - guidance scale value encourages the model to generate images closely linked to the text prompt at the expense of lower image quality. Guidance scale is enabled when guidance_scale > 1.
   - **steps** - how many steps generation will take
   - **resize** - determines if the face region should be resized. Highly recommended, SDXL doesn't work well with small pictures well.
   - **resize_to** - only when **resize** is set to True. Maximum value to which the cut region of the image will be scaled (larger should give better results but will be slower)
   - **mask_strength** - strenght of mask
   - **blur_mask** - how much blur add to a mask before composing it into the the result picture
   - **seed** - seed send to pipeline
   - **control_after_generate** - what to do with seed
   - **positive** - positive prompts
   - **negative** - negative prompts, works only when **guidance_scale** > 1
   - **negative2** - negative prompts

## Workflows
You can find example workflows in the /workflows folder.

## Tips
- If instead of face you are getting artifacts try using **resize** option with a high value if that doesn't help, try reducing the **padding** parameter.
- By using LCM lora you can generate good images in 2-3 steps (10+ otherwise).
- If the result is too different in color from the original, try reducing the number of steps and/or the ip_adapter_scale value.
- If you get `No face detected in pose image` error try to increase padding. It means the current mask + padding is not enough to detect the face in the input image by insightface.
