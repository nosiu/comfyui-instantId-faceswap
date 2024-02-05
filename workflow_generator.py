import json
import os
import sys

NODE_WIDTH = 400
NODE_HEIGHT = 500
NODE_SMALL_WIDTH = NODE_WIDTH
NODE_SMALL_HEIGHT = 80
PADDING = 40
IMAGES_PER_ROW = 10

if len(sys.argv) < 1:
  print("provide directory path argument")
  exit()

path = sys.argv[1]
filename = "workflow"

if len(sys.argv) > 2:
  filename = sys.argv[2]

images = []
extensions = [".jpg", ".jpeg", ".bmp", ".png", ".gif", ".webp", ".jiff"]
try:
  for file in os.listdir(path):
      if file.endswith(tuple(extensions)):
          images.append(os.path.join(path, file))
except:
  print("Couldn't find folder")
  exit()

workflow = {
  "last_node_id": 1,
  "last_link_id": 0,
  "nodes": [],
  "links": [],
  "groups": [],
  "config": {},
  "extra": {},
  "version": 0.4
}

def add_link(from_el, to_el, type):
  workflow["last_link_id"] += 1
  link = [
    workflow["last_link_id"],
    from_el["id"],
    0,
    to_el["id"],
    0
  ]

  for index, el in enumerate(from_el["outputs"]):
    if el["type"] == type:
      link[2] = index
      el["links"].append(workflow["last_link_id"])
      break

  for index, el in enumerate(to_el["inputs"]):
    if el["type"] == type:
      link[4] = index
      el["link"] = workflow["last_link_id"]
      break

  workflow["links"].append(link)

x = 0
y = 0

x += NODE_WIDTH + PADDING

for index, image in enumerate(images):
  image = {
    "id": workflow["last_node_id"],
    "type": "LoadImage",
    "pos": [x, y],
    "size": {"0": NODE_WIDTH, "1": NODE_HEIGHT,
    },
    "flags": {},
    "order": 0,
    "mode": 0,
    "outputs": [
      {
        "name": "IMAGE",
        "type": "IMAGE",
        "links": [],
        "shape": 3,
        "slot_index": 0
      },
      {
        "name": "MASK",
        "type": "MASK",
        "links": [],
        "shape": 3,
        "slot_index": 1
      }
    ],
    "title": "Face image ",
    "properties": {
      "Node name for S&R": "LoadImage"
    },
    "widgets_values": [image, "image"],
    "color": "#223",
    "bgcolor": "#335"
  }

  workflow["last_node_id"] += 1

  face_embed = {
    "id": workflow["last_node_id"],
    "type": "FaceEmbed",
    "pos": [x, y + NODE_HEIGHT + PADDING],
    "size": {"0": NODE_SMALL_WIDTH, "1": NODE_SMALL_HEIGHT},
    "flags": {},
    "order": 5,
    "mode": 0,
    "inputs": [
      {
        "name": "insightface",
        "type": "INSIGHTFACE_APP",
        "link": ""
      },
      {
        "name": "face_image",
        "type": "IMAGE",
        "link": ""
      },
      {
        "name": "face_embeds",
        "type": "FACE_EMBED"
      }
    ],
    "outputs": [
      {
        "name": "face embeds",
        "type": "FACE_EMBED",
        "links": [],
        "shape": 3,
        "slot_index": 0
      }
    ],
    "title": "Face embed ",
    "properties": {
      "Node name for S&R": "FaceEmbed"
    },
    "color": "#223",
    "bgcolor": "#335"
  }

  x += NODE_WIDTH + PADDING
  add_link(image, face_embed, "IMAGE")
  workflow["nodes"].append(image)
  workflow["nodes"].append(face_embed)

  workflow["last_node_id"] += 1

  if index > 0:
    add_link(workflow["nodes"][index * 2 - 1], face_embed, "FACE_EMBED")

  if (index + 1) % IMAGES_PER_ROW == 0:
    y += NODE_HEIGHT + NODE_SMALL_HEIGHT + (PADDING * 2)
    x = NODE_WIDTH + PADDING
# ----------------------------------+
x = 0
lora_loader_node = {
  "id": workflow["last_node_id"],
  "type": "LCMLora",
  "pos": [x, y],
  "size": {"0": NODE_SMALL_WIDTH, "1": NODE_SMALL_HEIGHT},
  "flags": {},
  "order": 6,
  "mode": 0,
  "outputs": [
    {
      "name": "LCM Lora",
      "type": "LCM_LORA",
      "links": [],
      "shape": 3,
      "slot_index": 0
    }
  ],
  "title": "Find LCM Lora",
  "properties": {
    "Node name for S&R": "LCMLora"
  },
  "widgets_values": ["pytorch_lora_weights.safetensors"],
  "color": "#322",
  "bgcolor": "#533"
}

workflow["last_node_id"] += 1

load_data_node = {
  "id": workflow["last_node_id"],
  "type": "FaceSwapSetupPipeline",
  "pos": [x, y + PADDING + NODE_SMALL_HEIGHT],
  "size": {"0": NODE_WIDTH, "1": NODE_HEIGHT},
  "flags": {},
  "order": 1,
  "mode": 0,
  "inputs": [
    {
      "name": "LCM_lora",
      "type": "LCM_LORA",
    }
  ],
  "outputs": [
    {
      "name": "inpaint pipe",
      "type": "FACESWAP_PIPE",
      "links": [],
      "shape": 3,
      "slot_index": 0
    },
    {
      "name": "insightface",
      "type": "INSIGHTFACE_APP",
      "links": [],
      "shape": 3,
      "slot_index": 1
    }
  ],
  "title": "Load EVERYTHING",
  "properties": {
    "Node name for S&R": "SetupPipeline"
  },
  "widgets_values": ["model.safetensors", "C://controlnet", "/ControlNetModel", "ip-adapter.bin"],
  "color": "#322",
  "bgcolor": "#533"
}

workflow["last_node_id"] += 1
# ----------------------------------+
y += NODE_HEIGHT + NODE_SMALL_HEIGHT + (PADDING * 2)
x = 0

upload_ref_image_node = {
  "id": workflow["last_node_id"],
  "type": "LoadImage",
  "pos": [x, y],
  "size": {"0": NODE_WIDTH,"1": NODE_HEIGHT},
  "flags": {},
  "order": 0,
  "mode": 0,
  "outputs": [
    {
      "name": "IMAGE",
      "type": "IMAGE",
      "links": [],
      "shape": 3,
      "slot_index": 0
    },
    {
      "name": "MASK",
      "type": "MASK",
      "links": [],
      "shape": 3,
      "slot_index": 1
    }
  ],
  "title": "Face image ",
  "properties": {
    "Node name for S&R": "LoadImage"
  },
  "widgets_values": [image, ""],
  "color": "#223",
  "bgcolor": "#335"
}

workflow["last_node_id"] += 1
# ----------------------------------+
x += NODE_WIDTH + PADDING

generation_node = {
  "id":  workflow["last_node_id"],
  "type": "FaceSwapGenerationInpaint",
  "pos": [
    x,
    y,
  ],
  "size": {
    "0": NODE_WIDTH,
    "1": NODE_HEIGHT
  },
  "flags": {},
  "order": 7,
  "mode": 0,
  "inputs": [
    {
      "name": "image",
      "type": "IMAGE",
      "link": ""
    },
    {
      "name": "mask",
      "type": "MASK",
      "link": ""
    },
    {
      "name": "face_embeds",
      "type": "FACE_EMBED",
      "link": ""
    },
    {
      "name": "inpaint_pipe",
      "type": "FACESWAP_PIPE",
      "link": ""
    },
    {
      "name": "insightface",
      "type": "INSIGHTFACE_APP",
      "link": ""
    }
  ],
  "outputs": [
    {
      "name": "IMAGE",
      "type": "IMAGE",
      "links": [],
      "shape": 3,
      "slot_index": 0
    }
  ],
  "properties": {
    "Node name for S&R": "GenerationInpaint"
  },
  "widgets_values": [
    70,
    10,
    0.8,
    0,
    3,
    0.9,
    10,
    "true",
    1024,
    769162424730269,
    "randomize",
    "",
    "",
    ""
  ],
  "color": "#323",
  "bgcolor": "#535"
}

workflow["last_node_id"] += 1
# ----------------------------------+
x += NODE_WIDTH + PADDING

preview_image_generated_node = {
  "id": workflow["last_node_id"],
  "type": "PreviewImage",
  "pos": [
    x,
    y
  ],
  "size": {
    "0": NODE_WIDTH,
    "1": NODE_HEIGHT
  },
  "flags": {},
  "order": 4,
  "mode": 0,
  "inputs": [
    {
      "name": "images",
      "type": "IMAGE",
    }
  ],
  "title": "Generated Image",
  "properties": {
    "Node name for S&R": "PreviewImage"
  },
  "color": "#432",
  "bgcolor": "#653"
}

workflow["last_node_id"] += 1
# ----------------------------------+
x += NODE_WIDTH + PADDING

preview_image_original_node = {
  "id": workflow["last_node_id"],
  "type": "PreviewImage",
  "pos": [x,y],
  "size": {"0": NODE_WIDTH, "1": NODE_HEIGHT},
  "flags": {},
  "order": 4,
  "mode": 0,
  "inputs": [
    {
      "name": "images",
      "type": "IMAGE",
    }
  ],
  "title": "Original Image",
  "properties": {
    "Node name for S&R": "PreviewImage"
  },
  "color": "#432",
  "bgcolor": "#653"
}
# ----------------------------------+

for index, node in enumerate(workflow["nodes"]):
  if index % 2 == 1:
    add_link(load_data_node, node, "INSIGHTFACE_APP")

last_face_embed_node = workflow["nodes"][len(workflow["nodes"]) - 1]

workflow["nodes"].append(lora_loader_node)
workflow["nodes"].append(load_data_node)
workflow["nodes"].append(upload_ref_image_node)
workflow["nodes"].append(generation_node)
workflow["nodes"].append(preview_image_generated_node)
workflow["nodes"].append(preview_image_original_node)

add_link(lora_loader_node, load_data_node, "LCM_LORA")
add_link(load_data_node, generation_node, "INSIGHTFACE_APP")
add_link(load_data_node, generation_node, "FACESWAP_PIPE")
add_link(upload_ref_image_node, generation_node, "IMAGE")
add_link(upload_ref_image_node, generation_node, "MASK")
add_link(generation_node, preview_image_generated_node, "IMAGE")
add_link(upload_ref_image_node, preview_image_original_node, "IMAGE")
add_link(last_face_embed_node, generation_node, "FACE_EMBED")

with open(filename + ".json", 'w') as fp:
  json.dump(workflow, fp)