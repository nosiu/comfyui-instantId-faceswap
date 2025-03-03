# ComfyUI InstantID FaceSwap v0.1.1
<sub>[About](#comfyui-instantid-faceswap-v011) | [Installation guide](#installation-guide) | [Custom nodes](#custom-nodes) | [Workflows](#workflows) | [Tips](#tips) | [Changelog](#changelog)</sub>

Implementation of [faceswap](https://github.com/nosiu/InstantID-faceswap/tree/main) based on [InstantID](https://github.com/InstantID/InstantID) for ComfyUI. \
Since version 0.1.0 it also allows generating people based on text.
</br>
**Works ONLY with SDXL checkpoints**
</br>
</br>
![image](https://github.com/user-attachments/assets/0c97dccf-ac8a-43f7-b50b-8bbf7ed81049)

![image](https://github.com/user-attachments/assets/bbc88aaf-fba4-43f1-80ea-fece379308db)



## Installation guide
<sub>[About](#comfyui-instantid-faceswap-v011) | [Installation guide](#installation-guide) | [Custom nodes](#custom-nodes) | [Workflows](#workflows) | [Tips](#tips) | [Changelog](#changelog)</sub>

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
       - ControlNetModel/diffusion_pytorch_model.safetensors and ControlNetModel/config.json  - put those files in new folder in  **ComfyUI/models/controlnet**

Newly added files hierarchy should look like this:
```
ComfyUI
\---models
    \---ipadapter
           ipadapter.bin
    \---controlnet
        \--- FOLDER_YOU_CREATED
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

*Note You don't need to add the 'ipadapter', and 'controlnet' folders to this specific location if you already have them somewhere else (also you can rename ipadapter.bin and ControlNetModel to something of your liking).
Instead, You can edit `ComfyUI/extra_model_paths.yaml` and add folders containing those files to the config.

## Custom nodes
<sub>[About](#comfyui-instantid-faceswap-v011) | [Installation guide](#installation-guide) | [Custom nodes](#custom-nodes) | [Workflows](#workflows) | [Tips](#tips) | [Changelog](#changelog)</sub>

- ### Load Insightface:
   Loads Insightface. Models need to be in a specific location. Check the  [Installation guide](#installation-guide) for details.

- ### Load instantId adapter:
   Loads the InstantId adapter and resampler. The model needs to be in a specific location. Check the [Installation guide](#installation-guide) for details. The resampler is used to prepare face embeds for ControlNet and the adapter.

- ### Apply instantId adapter:
   Applies the InstantId adapter to the model. This is optional—you can achieve good results without using this node.

   **Params:**
   - **checkpoint** - SDXL checkpoint
   - **instantId_adapter** - intantId adapter
   - **face_conditioning** - face conditioning prepared by the resampler
   - **strength** - strength of the instantId adapter

- ### Apply instantId ControlNet:
   Applies InstantId ControlNet.

   **Params:**
   - **positive**  - positive prompts
   - **negative**  - negative prompts
   - **face_conditioning** - face conditioning prepared by the resampler
   - **control_net** - instantId Controlnet
   - **strength** - strength of instantId ControlNet

- ### Apply instantId and ControlNet:
    A subgraph node that bundles several operations into a single node for convenience. It includes the following nodes: LoadInstantIdAdapter, FaceEmbedCombine, ControlNetLoader, InstantIdAdapterApply, and ControlNetInstantIdApply.

    This node streamlines the process by loading the InstantId adapter, combining face embeddings, loading the ControlNet, and applying both the InstantId adapter and ControlNet in one step.

- ### FaceEmbed for instantId
   Prepares face embeds for generation. You can chain multiple face embeds.

   **Params:**
   - **insightface** - insightface
   - **face_image** - input image from which to extract embed data
   - **face_embeds** (*optional*) - additional face embed(s)

- ### FaceEmbed Combine
   Prepares face embeds for ControlNet and the adapter.

   **Params:**
   - **resampler** - resampler
   - **face_embeds** - face_embeds

- ### Get Angle from face
   Returns the angle (in degrees) by which the image must be rotated counterclockwise to align the face. Since there can be more than one face in the image, face search is performed only in the area of the drawn mask, enlarged by the pad parameter.

   **Note:** If the face is rotated by an extreme angle, insightface won't be able to find the correct position of face keypoints, so the rotation angle might not always be accurate. In these cases, manually draw your own KPS.


   **Params:**
   - **insightface** - insightface
   - **image** - image with the face to rotate
   - **mask** - mask
   - **rotate_mode** - available options:
      - *none* - returns 0
      - *loseless* - returns the closest angle to 90, 180, 270 degrees
      - *any* - returns a specific angle by which the image should be rotated
   - **pad_top** - how many pixels to enlarge the mask upwards
   - **pad_right** - how many pixels to enlarge the mask to the right
   - **pad_bottom** - how many pixels to enlarge the mask downwards
   - **pad_left**  -  how many pixels to enlarge the mask to the left


- ### Get Angle from KPS data
   Returns the angle (in degrees) by which the image must be rotated counterclockwise to align the face.

   **Params:**
   - **rotate_mode** - available options:
      - *none* - returns 0
      - *loseless* - returns the closest angle to 90, 180, 270 degrees
      - *any* - returns a specific angle by which the image should be rotated


- ### Rotate Image
   Rotates the image by the given angle and expands it.

   **Params:**
  - **image** - image
  - **angle** - angle
  - **counter_clockwise** - direction

- ### Remove rotation padding
   Removes the expanded region added by two rotations (first to align the face, and second to return to the original position).

   **Params:**
  - **original_image** - image before rotation
  - **rotated_image** - rotated image

- ### Draw KPS
   Allows you to draw your own keypoints (KPS), useful when you get the error `"No face detected in pose image"` or when using InstantId to generate images from prompts only. Click and drag the KPS to move them around.

   When you place your KPS in the desired position, this node will show the angle by which the image should be rotated to align the face.

   You can adjust the opacity of each keypoint to sacrifice likeness, for example, when adding "glasses".

   **Shortcuts:**\
      **CTRL + DRAG** - move around\
      **CTRL + WHEEL** - zoom in / out\
      **ALT + WHEEL** - decrease / increase distance of other points from blue point (nose kps)

   **Params:**
   - **image_reference** (optional) - an image that serves as a background to more accurately match the appropriate points. If provided, the resulting image will have the width and height of this image.
   - **width** - width of the image (disabled if `image_reference` is provided)
   - **height** - height of the image (disabled if `image_reference` is provided)

- ### 3d KPS from image
   Allows you to extract 3D keypoints (KPS) of a face from an image. This is useful when generating content from text prompts and wanting to rotate the face while preserving the distance between the eyes, nose, and mouth.

   To use it, connect the `image` node, then click the **"Get KPS From Image"** button. Afterward, you can adjust the scale, position, and rotation of the face by clicking the **"Change KPS"** button.

   **IMPORTANT:** 
      - Clicking Get KPS From Image will run InsightFace to extract KPS data, so it’s best not to use this in the middle of the generation process (this depends on your system's performance).
      - You cannot manually change the distance between KPS in this node.

   Once your KPS are placed in the desired position, the node will show the angle by which the image should be rotated to align the face.

   You can adjust the opacity of each keypoint to sacrifice likeness, for example, when adding "glasses".

   **Shortcuts:**\
      **CTRL + DRAG** - move around\
      **CTRL + WHEEL** - zoom in / out\
      **ALT + WHEEL** - scale KPS

   **Params:**
   - **image** - an image of face from which KPS will be calculated
   - **width** - width of the image
   - **height** - height of the image


- ### Preprocess image for instantId:
   Cuts out the mask area wrapped in a square, enlarges it in each direction by the `pad` parameter, and resizes it (to dimensions rounded down to multiples of 8). It also creates a control image for InstantId ControlNet.

   **Note:** If the face is rotated by an extreme angle, the prepared `control_image` may be drawn incorrectly.

   If the `insightface` param is not provided, it will not create a control image, and you can use this node as a regular node for inpainting (to cut the masked region with padding and later compose it).

   **Params:**
   - **image** - your pose image (the image in which the face will be swapped)
   - **mask** - drawn mask (the area to be changed must contain the face; you can also mask other features like hair or hats and change them later with prompts)
   - **insightface** (optional) - loaded insightface
   - **width** - width of the image in pixels (check `resize_mode`)
   - **height** - height of the image in pixels, check `resize_mode`
   - **resize_mode** - available options:
      - *auto* - automatically calculates the image size so that the area is `width` x ` height`.
         For SDXL, you probably want to use this option with:
         **width: 1024, height: 1024**
      - *scale by width* -  ignores provided `height` and calculates it based on the aspect ratio
      - *scale by height* - ignores provided `width` and calculates it based on the aspect ratio
      - *free* - uses the provided `width` and `height`
   - **pad** - how many pixels to enlarge the mask in each direction

- ### Preprocess image for instantId (Advanced):
   Same as **Preprocess Image for InstantId** with five additional parameters.

   **Params:**
   - **upscale_method**  - *nearest-exact*, *bilinear*, *area*, *bicubic*, *lanczos*
   - **pad_top** - how many pixels to enlarge the mask upwards
   - **pad_right** - how many pixels to enlarge the mask to the right
   - **pad_bottom** - how many pixels to enlarge the mask downwards
   - **pad_left**  -  how many pixels to enlarge the mask to the left


- ### Randomize 2d KPS
   Randomizes the position, angle, and rotation of the KPS based on the provided parameters.

   **Params:**
   - **angle_min** - minimum rotation angle. The rotation point is the center of the KPS,
   - **angle_max** - maximum rotation angle, the rotation point is the center of the KPS,
   - **scale_min** - minimum scaling value relative to the KPS center (1 means no scaling),
   - **scale_max** - maximum scaling value relative to the KPS center (1 means no scaling),
   - **translate_x** - value by which to shift the KPS along the X axis. For example, setting it to 200 will randomly select a shift value between -200 and 200,
   - **translate_y** - value by which to shift the KPS along the Y axis. For example, setting it to 200 will randomly select a shift value between -200 and 200,
   - **border** - edge threshold; if the KPS gets too close to the image border during translation, rotation, or scaling, it will be repositioned to this value


- ### Randomize 3d KPS
   Randomizes the rotation of the KPS around three axes. Setting any of the parameters will randomly select a rotation angle around the corresponding axis. The rotation point is the center of the KPS.
   Example: Setting rotate_x to 20 will rotate the KPS by a random angle between -20 and 20 degrees.

   **Params:**
   - **rotate_x** - rotation angle around the X axis,
   - **rotaet_y** - rotation angle around the Y axis,
   - **rotate_z** - rotation angle around the Z axis,


- ### Scale 2d KPS by
   Scales the KPS data by a given factor.

   **Params:**
   - **scale**: scaling factor


- ### Scale 2d KPS
   Scales the KPS data to the specified width and height.

   **Params:**
   - **width**: desired width,
   - **height**: desired height


- ### Rotate 2d KPS
   Rotates the KPS by the given angle and expands it.

   **Params:**
   - **angle**: rotation angle


 - ### Crop 2d KPS
   Crops the KPS.

   **Params:**
   - **x**: X coordinate of the top-left corner,
   - **y**: Y coordinate of the top-left corner,
   - **width**:  width of the cropped area,
   - **height**: height of the cropped area,


 - ### Create KPS Image
   Creates a control_image from `kps_data`


 - ### Create mask from Kps
   Creates a mask based on the KPS position.

   **Params:**
   - **grow_by** - expands the mask by adding extra space on all sides. The additional margin on each side is equal to the mask’s dimension divided by this value. For example, if the KPS width is 20 pixels and grow_by is set to 10, an extra 20/10 (i.e., 2 pixels) will be added to the left and right sides — resulting in a new width of 20 + 2 + 2 = 24 pixels. The same applies to the height.

## Workflows
<sub>[About](#comfyui-instantid-faceswap-v011) | [Installation guide](#installation-guide) | [Custom nodes](#custom-nodes) | [Workflows](#workflows) | [Tips](#tips) | [Changelog](#changelog)</sub>


You can find example workflows in the `/workflows` folder.
Nodes colors legend: \
**yellow** - node from this extension,\
**blue** - inputs, load your controlnets, models, images ...\
**purple** - you might want to configure those\
**cyan** - output images\
**green** - positive prompts\
**red** - negative prompts

If you set the mask blur options remember that it will shrink the area you masked

### simple.json
Face swap: Set your pose image, draw a mask, set your face reference (the face that will replace the masked area in the pose image), and that's it.

### simple_with_adapter.json
Same as `simple.json` with an additional node `Apply instantId adapter`

### simple_two_embeds.json
Same as `simple.json`, but allows you to provide two face references. You can use this to merge two different faces or just provide a second reference for the first face.

### draw_kps.json
Face swap: Set your pose image, draw a mask, set your face reference (the face that will replace the masked area in the pose image), and then click the "draw KPS" button on the `Draw KPS` node to set your KPS.

<details>
  <summary>View Example Image</summary
                                 
  ![DRAW KPS](https://github.com/user-attachments/assets/9c87fa80-bb51-4df5-aca8-8cfee3d1668b)
</details>

### draw_kps_rotate.json
Same as `draw_kps.json`, but it will also rotate the pose image. After setting your KPS, you should set the angle by which you want to rotate the image to align the face properly.

<details>
  <summary>View Example Image</summary
                                 
  ![KPS SET ANGLE](https://github.com/user-attachments/assets/665dbfdd-79ce-47a0-9004-40c0cf48596c)
</details>

### auto_rotate.json
Same as `simple_with_adapter.json`, but it will automatically detect the angle of rotation based on the mask and padding set in the `Get Angle from Face` node.

### promp2image.json
Generates an image based only on the face reference and prompts. Set your face reference, draw the KPS where the face should be drawn, and add prompts like "man sitting in the park."

### promp2image_detail_pass.json
Same as `prompt2image.json`, but this one expects the KPS you draw to be small, so the face is not detailed (or may even be deformed). Second pass should fix the face.


### prompts2img_2faces_enhancement.json
A workflow that generates two faces in one image and enhances them one by one.
Set your face references and KPS for one image, then set a second KPS in another region of the picture. Good results depend on your prompts.

<details>
  <summary>View Example Image</summary>
    
  ![Two KPS one flow](https://github.com/user-attachments/assets/fbaa38df-3400-401d-b644-087723e6488c)
</details>

### inpaint.json
Since you can use the `Preprocess Image for InstantId` and `Preprocess Image for InstantId (Advanced)` nodes to resize your images with a mask, this workflow is useful for inpainting in general. This workflow shows you how to do it.

<details>
  <summary>View Example Image</summary>
    
  ![basic inpaint](https://github.com/user-attachments/assets/bda258b1-a988-47f5-beb6-105289c990ac)

</details>

### prop2image_latent_upscale.json
Similar to `promp2image_detail_pass.json`, this workflow allows you to draw your KPS. The workflow will run a first pass at 50%, upscale the latent by `1.4`, and finish with a detail pass on the face area.

### prop2image_latent_upscale_with_2d_randomizer.json
Same as `prop2image_latent_upscale.json`, but it will randomize the position of the face within the image.

### prop2image_latent_upscale_with_3d_and_2d_randomizer.json
Similar to `prop2image_latent_upscale_with_2d_randomizer.json`, but instead of drawing your KPS, you retrieve them from a face image (in 3d).
Click the **"Get KPS from image"** button on the 3D KPS from Image node, then use the **"Change KPS"** button to adjust the position of your KPS. You can also randomize the 3D rotation in this workflow.

### prop2image_latent_upscale_with_3d_and_2d_randomizer_with_rotation.json
Exactly the same as `prop2image_latent_upscale_with_3d_and_2d_randomizer.json`, with one small addition: you can set the face rotation method during the last pass. The workflow will rotate the face to a "straight" position, process the image, and then composite it into the final result.

The default rotation is set to "any," so you might encounter some artifacts from the rotation. You can adjust this setting as needed.

## Tips
<sub>[About](#comfyui-instantid-faceswap-v011) | [Installation guide](#installation-guide) | [Custom nodes](#custom-nodes) | [Workflows](#workflows) | [Tips](#tips) | [Changelog](#changelog)</sub>

- Most workflows require you to draw a mask on the pose image.
- If you encounter the error `No face detected in pose image`, try drawing a larger mask or increasing the `pad` parameter or draw KPS yourself.
- You can adjust the opacity of each keypoint to preserve original features or rely more on your prompts without sacrificing the overall likeness of the face.
- You can modify more than just the face — add accessories like a hat, change hair, or even alter expressions.
- If you're changing a lot of elements unrelated to the face, it's a good idea to add a second pass focused primarily on the face area to enhance detail.
- To improve results, you can integrate other extensions such as ControlNet for inpainting, Fooocus inpaint, FaceShaper, Expression Lora, and many more.
- To understand the relationship between ControlNet and the adapter, check the official paper linked in the instantId repository: https://github.com/instantX-research/InstantID?tab=readme-ov-file

## Changelog
<sub>[About](#comfyui-instantid-faceswap-v011) | [Installation guide](#installation-guide) | [Custom nodes](#custom-nodes) | [Workflows](#workflows) | [Tips](#tips) | [Changelog](#changelog)</sub>

- ### 0.1.1 (03.03.2025)
   - Introduced new nodes to ensure proper KPS size.
   - Removed the ability to draw masks on the KPS node (a separate node is available for this).
   - Added the ability to add transparency to individual KPS, providing similar functionality as `ControlNet Scale` but for specific parts of the face (e.g., adding glasses).
   - The generation of KPS control images has been entirely moved to the backend (Python). You can still draw KPS manually, but this change reduces the creation of temporary files.
   - Drawn KPS positions are now saved into the workflow.
   - Added the ability to import 3D KPS positions from the face, allowing rotation, scaling, and movement of those points while preserving the distance between the eyes, nose, and mouth.
   - Added options to randomize position, rotation, and KPS scaling to diversify final images.
   - Added the ability to randomize 3D face rotation to further diversify results.
   - To diversify results even further, you can use tools like [comfyui-text-randomizer](https://github.com/nosiu/comfyui-text-randomizer), which was created as a side project during the development of this repository.

   **Note:** Some old workflows will not be compatible with this version.


- ### 0.1.0 (20.10.2024)
   - The code was rewritten from scratch and now uses the ComfyUI backend. This allows you to chain LORAs or ControlNets as needed, providing greater control over the entire process.
For example, you can now draw your own KPS, enabling both text-to-image and image-to-image generation.
   - Removed most dependencies (including Diffusers).
   - Removed all old nodes and introduced new ones.
   - The script that automatically generated workflows based on all faces in a specific catalog has been removed.

   **Note:** Old workflows will not work with this version.


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

