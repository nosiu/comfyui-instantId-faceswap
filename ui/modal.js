import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { createShader, vertexShaderSrc, fragmentShaderSrc } from "./shaders.js"

const checkWebGlSupport = () => {
  const canvas = document.createElement("canvas");
  const gl = canvas.getContext("webgl2")
  return !!gl
}

const dataURLToBlob = (dataURL) => {
  const parts = dataURL.split(";base64,");
  const contentType = parts[0].split(":")[1];
  const byteString = atob(parts[1]);
  const arrayBuffer = new ArrayBuffer(byteString.length);
  const uint8Array = new Uint8Array(arrayBuffer);
  for (let i = 0; i < byteString.length; i++) {
    uint8Array[i] = byteString.charCodeAt(i);
  }
  return new Blob([arrayBuffer], { type: contentType });
}

class KPSDialog {
  constructor(w, h, imageWidget, kps, referenceImage, angleWidget) {
    this.isDragging = false;
    this.draggedPointIndex = null;
    this.mousedown_x = undefined;
    this.mousedown_y = undefined;
    this.mousedown_pan_x = undefined;
    this.mousedown_pan_y = undefined;
    this.pan_x = 0
    this.pan_y = 0
    this.cursorX = undefined
    this.cursorY = undefined
    this.zoom_ratio = 1
    this.min_zoom = undefined
    this.hasImage = referenceImage ? true  : false
    if (this.hasImage) {
      this.opacity = "0.6"
    } else {
      this.opacity = "1"
    }
    this._kps = kps
    this.canvasWidth = w
    this.canvasHeight = h
    this.setElements()
    this.element.style.display = "block"
    this.initializeCanvasPanZoom()
    this.setControl()
    this.attachListeners()
    this.canvas.style.opacity = this.opacity
    this.imageWidget = imageWidget
    this.angleWidget = angleWidget
    this.kps = kps.array.length ? JSON.parse(JSON.stringify(kps.array)) : this.getDefaultKps()
    this.draw()
    if (this.hasImage) this.drawImage(referenceImage)
  }

  getDefaultKps () {
    const halfWidth = this.canvasWidth / 2
    const halfHeight = this.canvasHeight / 2
    return [
      [halfWidth - halfWidth / 2, halfHeight - halfHeight / 2],
      [halfWidth + halfWidth / 2, halfHeight - halfHeight / 2],
      [halfWidth, halfHeight],
      [halfWidth - halfWidth / 2, halfHeight + halfHeight / 2],
      [halfWidth + halfWidth / 2, halfHeight + halfHeight / 2],
    ]
  }

  setElements () {
    this.element = document.createElement("div")
    this.element.style.display = "none"
    this.element.style.width = "80vw"
    this.element.style.height = "80vh"
    this.element.style.zIndex = 8888
    this.element.classList.add('comfy-modal')
    this.element.classList.add('kps-sandbox')

    document.body.appendChild(this.element)

    this.setCanvas()
  }

  setControl () {
    const buttonBar = document.createElement("div")
    buttonBar.style.position = "absolute"
    buttonBar.style.bottom = "0"
    buttonBar.style.height = "50px"
    buttonBar.style.left = "20px"
    buttonBar.style.right = "20px"
    buttonBar.style.pointerEvents = "none"
    buttonBar.appendChild(this.createButton("Save", true, this.save.bind(this)))
    buttonBar.appendChild(this.createButton("Cancel", true, this.closeModal.bind(this)))
    buttonBar.appendChild(this.createButton("Reset pan & zoom", false, () => {
      this.initializeCanvasPanZoom()
    }))
    buttonBar.appendChild(this.createButton("Reset KPS", false, () => {
      this.kps = this.getDefaultKps()
      this.draw()
    }))
    if (this.hasImage) {
      buttonBar.appendChild(this.createOpacitySlider("Opacity", (event) => {
        this.opacity = event.target.value
        this.canvas.style.opacity = event.target.value

      }))
    }
    buttonBar.appendChild(this.createZoomSlider("Zoom", (event) => {
      this.zoom_ratio = parseFloat(event.target.value)
      this.invalidatePanZoom()

    }))
    this.element.appendChild(buttonBar)
  }

  createButton(name, isRight, callback) {
    const button = document.createElement("button");
    button.innerText = name;
    button.style.pointerEvents = "auto";
    button.addEventListener("click", callback);
    if (isRight) {
      button.style.cssFloat = "right";
      button.style.marginLeft = "4px";
    } else {
      button.style.cssFloat = "left";
      button.style.marginRight = "4px";
    }
    return button;
  }

  createOpacitySlider( name, callback) {
    const divElement = document.createElement("div");
    divElement.style.cssFloat = "left";
    divElement.style.fontFamily = "sans-serif";
    divElement.style.marginRight = "4px";
    divElement.style.color = "var(--input-text)";
    divElement.style.backgroundColor = "var(--comfy-input-bg)";
    divElement.style.borderRadius = "8px";
    divElement.style.borderColor = "var(--border-color)";
    divElement.style.borderStyle = "solid";
    divElement.style.fontSize = "15px";
    divElement.style.height = "21px";
    divElement.style.padding = "1px 6px";
    divElement.style.display = "flex";
    divElement.style.position = "relative";
    divElement.style.top = "2px";
    divElement.style.pointerEvents = "auto";

    const opacity_slider_input = document.createElement("input");
    opacity_slider_input.setAttribute("type", "range");
    opacity_slider_input.setAttribute("min", "0.1");
    opacity_slider_input.setAttribute("max", "1.0");
    opacity_slider_input.setAttribute("step", "0.01");
    opacity_slider_input.setAttribute("value", this.opacity);
    const labelElement = document.createElement("label");
    labelElement.textContent = name;
    divElement.appendChild(labelElement);
    divElement.appendChild(opacity_slider_input);
    opacity_slider_input.addEventListener("input", callback);
    return divElement;
  }

  createZoomSlider( name, callback) {
    const divElement = document.createElement("div");
    divElement.id = "instantIdZoomSlider";
    divElement.style.cssFloat = "left";
    divElement.style.fontFamily = "sans-serif";
    divElement.style.marginRight = "4px";
    divElement.style.color = "var(--input-text)";
    divElement.style.backgroundColor = "var(--comfy-input-bg)";
    divElement.style.borderRadius = "8px";
    divElement.style.borderColor = "var(--border-color)";
    divElement.style.borderStyle = "solid";
    divElement.style.fontSize = "15px";
    divElement.style.height = "21px";
    divElement.style.padding = "1px 6px";
    divElement.style.display = "flex";
    divElement.style.position = "relative";
    divElement.style.top = "2px";
    divElement.style.pointerEvents = "auto";

    const zoom_slider_input = document.createElement("input");
    zoom_slider_input.setAttribute("type", "range");
    zoom_slider_input.setAttribute("min", `${this.min_zoom}`);
    zoom_slider_input.setAttribute("max", "2");
    zoom_slider_input.setAttribute("step", "0.1");
    zoom_slider_input.setAttribute("value", `${this.zoom_ratio}`);
    const labelElement = document.createElement("label");
    labelElement.textContent = name;
    divElement.appendChild(labelElement);
    divElement.appendChild(zoom_slider_input);
    zoom_slider_input.addEventListener("input", callback);
    return divElement;
  }

  setCanvas () {
    if (!this.element.querySelector("canvas")) {
      this.canvas = document.createElement("canvas")
      this.canvas.style.position = "absolute"
      this.canvas.style.pointerEvents = "auto"
      this.canvas.style.zIndex = "-1"
      this.element.appendChild(this.canvas)

      this.imageCanvas = document.createElement("canvas")
      this.imageCanvas.style.position = "absolute"
      this.imageCanvas.style.zIndex = "-2"
      this.imageCanvas.style.pointerEvents = "none"
      this.element.appendChild(this.imageCanvas)

    }
    this.canvas.width = this.canvasWidth
    this.canvas.height = this.canvasHeight
    this.imageCanvas.width = this.canvasWidth
    this.imageCanvas.height = this.canvasHeight
  }

  initializeCanvasPanZoom() {
    let drawWidth = this.canvasWidth;
    let drawHeight = this.canvasHeight;
    let width = this.element.clientWidth;
    let height = this.element.clientHeight;

    if (this.canvasWidth > width) {
      drawWidth = width;
      drawHeight = drawWidth / this.canvasWidth * this.canvasHeight;
    }
    if (drawHeight > height) {
      drawHeight = height;
      drawWidth = drawHeight / this.canvasHeight * this.canvasWidth;
    }
    this.zoom_ratio = drawWidth / this.canvasWidth;
    this.min_zoom = drawWidth / this.canvasWidth
    const canvasX = (width - drawWidth) / 2;
    const canvasY = (height - drawHeight) / 2;
    this.pan_x = canvasX;
    this.pan_y = canvasY;
    this.invalidatePanZoom();
  }

  invalidatePanZoom () {
    let raw_width = this.canvasWidth * this.zoom_ratio;
    let raw_height = this.canvasHeight * this.zoom_ratio;
    if (this.pan_x + raw_width < 10) {
      this.pan_x = 10 - raw_width;
    }
    if (this.pan_y + raw_height < 10) {
      this.pan_y = 10 - raw_height;
    }
		this.canvas.style.width = `${raw_width}px`;
		this.canvas.style.height = `${raw_height}px`;
		this.canvas.style.left = `${this.pan_x}px`;
		this.canvas.style.top = `${this.pan_y}px`;

		this.imageCanvas.style.width = `${raw_width}px`;
		this.imageCanvas.style.height = `${raw_height}px`;
		this.imageCanvas.style.left = `${this.pan_x}px`;
		this.imageCanvas.style.top = `${this.pan_y}px`;
		}

  attachListeners () {
    this.canvas.addEventListener('mousedown', this.mouseDown.bind(this))
    this.canvas.addEventListener('mousemove', this.mouseMove.bind(this))
    this.canvas.addEventListener('mouseup', this.mouseUp.bind(this))
    this.element.addEventListener('wheel', this.wheel.bind(this))
    this.element.addEventListener('DOMMouseScroll', (e) => e.preventDefault()) // thanks firefox.
    this.element.addEventListener('keydown', (event) => {
      event.preventDefault();
      if (event.key === "Escape") {
        this.closeModal()
      } else if (event.key === "ENTER") {
        this.save()
      }
    })
  }

  closeModal () {
    document.body.removeChild(this.element)
  }

  save () {
    const canvasCpy = document.createElement('canvas')
    canvasCpy.width = this.canvasWidth
    canvasCpy.height = this.canvasHeight
    this.drawKps(canvasCpy)
    const dataURL = canvasCpy.toDataURL("image/png", 1);
    const blob = dataURLToBlob(dataURL);
    const body = new FormData();

    const fileName = `fs-${performance.now()}.png`
    body.append("image", blob , fileName);
    body.append("subfolder", "faceswap_controls");
    body.append("type", "input");

    api.fetchApi("/upload/image", {
      method: "POST",
      body: body
    }).then(() => {
      this._kps.array = [...this.kps]
      this.imageWidget.value = fileName
      this.imageWidget.callback()

      const a = this.kps[0]
      const b = this.kps[1]
      let angle = Math.atan2(b[1] - a[1], b[0] - a[0]) * 180 / Math.PI;


      this.angleWidget.value = angle

    }).catch((error) => {
      console.error("ERROR: ComfyUI.instantid-faceswap:" , error);
    }).finally(() => {
      this.closeModal()
    });
  }

  changePointsPosition(closer = false, step = 10) {
    step /= this.zoom_ratio
    const center = this.kps[2]
    return this.kps.map((point, index) => {
        if (index === 2) return point
        const direction = [center[0] - point[0], center[1] - point[1]];

        const magnitude = Math.sqrt(direction[0] * direction[0] + direction[1] * direction[1])
        const unitVector = [direction[0] / magnitude, direction[1] / magnitude]
        const moveVector = [unitVector[0] * step, unitVector[1] * step]

        if (closer) {
            if (magnitude < 50) return point
            return [
              point[0] + moveVector[0],
              point[1] + moveVector[1]
            ]
        } else {
            return [
              point[0] - moveVector[0],
              point[1] - moveVector[1]
            ]
        }
    })
}

  mouseDown (event) {
    event.preventDefault()
    const { offsetX: mouseX, offsetY: mouseY } = event

		if (event.ctrlKey) {
		  if (event.buttons == 1) {
			this.mousedown_x = event.clientX
			this.mousedown_y = event.clientY
			this.mousedown_pan_x = this.pan_x
			this.mousedown_pan_y = this.pan_y
		  }
		  return;
		} else {
      let maxX = -Infinity
      let maxY = -Infinity

			this.kps.forEach((kp, idx) => {
			  let [x, y] = kp;
			  x *= this.zoom_ratio
			  y *= this.zoom_ratio
        maxX = x > maxX ? x : maxX
        maxY = y > maxY ? y : maxY
			  const distance = Math.sqrt((x - mouseX) ** 2 + (y - mouseY) ** 2)
			  if (distance < 20 * (this.zoom_ratio)) {
          this.isDragging = true;
          this.draggedPointIndex = idx;
          return;
			  }
			});
      maxX += 20 * this.zoom_ratio
      maxY += 20 * this.zoom_ratio
      if ((mouseX >= maxX) && (mouseY >= maxY) && (mouseX < maxX + 60 * this.zoom_ratio) && (mouseY < maxY + 60 * this.zoom_ratio)){
        this.mousedown_x = event.clientX;
        this.mousedown_y = event.clientY;
        this.isDragging = true;
        this.draggedPointIndex = -1;
      }
		}
  }

  mouseMove (event) {
    event.preventDefault();
    const { offsetX: mouseX, offsetY: mouseY } = event;
		this.cursorX = event.pageX;
		this.cursorY = event.pageY;
		if (event.ctrlKey) {
		  if (event.buttons == 1) {
        if (this.mousedown_x) {
          let deltaX = this.mousedown_x - event.clientX;
          let deltaY = this.mousedown_y - event.clientY;
          this.pan_x = this.mousedown_pan_x - deltaX;
          this.pan_y = this.mousedown_pan_y - deltaY;
          this.invalidatePanZoom();
        }
      }
		}
		if (this.isDragging) {
			const transformedX = (mouseX) / this.zoom_ratio;
			const transformedY = (mouseY) / this.zoom_ratio;
      if(this.draggedPointIndex !== null && this.draggedPointIndex > -1) {
			  this.kps[this.draggedPointIndex] = [transformedX, transformedY];
      } else if (this.draggedPointIndex === -1) {
        let deltaX = this.mousedown_x - event.clientX
        let deltaY = this.mousedown_y - event.clientY
        this.mousedown_x = event.clientX
        this.mousedown_y = event.clientY
        this.kps.forEach(el => {
          el[0] -= deltaX / this.zoom_ratio
          el[1] -= deltaY / this.zoom_ratio
        })
      }
      this.draw();
		}
  }

  mouseUp (event) {
    event.preventDefault();
		this.mousedown_x = null;
		this.mousedown_y = null;
		this.isDragging = false;
		this.draggedPointIndex = null;
  }

  wheel (event) {
    event.preventDefault();
		if (event.ctrlKey) {
		  if (event.deltaY < 0) {
			  this.zoom_ratio = Math.min(2, this.zoom_ratio + 0.2);
		  } else {
        this.zoom_ratio = Math.max(this.min_zoom, this.zoom_ratio - 0.2)
		  }
      document.querySelector("#instantIdZoomSlider input").value = `${this.zoom_ratio}`
		  this.invalidatePanZoom();
		}
    else if (event.altKey) {
      this.kps = this.changePointsPosition(event.deltaY > 0)
      this.draw()
    }
  }

  draw () {
    this.drawKps()
    this.drawMoveAll()
  }

  drawKps (canvas = this.canvas) {
    const color_list = [
      'rgba(255, 0, 0, 0.6)',
      'rgba(0, 255, 0, 0.6)',
      'rgba(0, 0, 255, 0.6)',
      'rgba(255, 255, 0, 0.6)',
      'rgba(255, 0, 255, 0.6)'
    ]
    const ctx = canvas.getContext("2d")
    const stickWidth = 10;
    const limbSeq = [[0, 2], [1, 2], [3, 2], [4, 2]];

    ctx.clearRect(0, 0, this.canvasWidth, this.canvasHeight);
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, this.canvasWidth, this.canvasHeight);
    ctx.save();
    limbSeq.forEach((limb) => {
      const kp1 = this.kps[limb[0]];
      const kp2 = this.kps[limb[1]];
      const color = color_list[limb[0]];

      const x = [kp1[0], kp2[0]];
      const y = [kp1[1], kp2[1]];
      const length = Math.sqrt((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2);
      const angle = Math.atan2(y[1] - y[0], x[1] - x[0]);

      const num_points = 20;
      const polygon = [];

      const midX = (x[0] + x[1]) / 2;
      const midY = (y[0] + y[1]) / 2;

      for (let i = 0; i <= num_points; i++) {
        const theta = (i / num_points) * Math.PI * 2;
        const dx = (length / 2) * Math.cos(theta);
        const dy = (stickWidth / 2) * Math.sin(theta);
        const rx = Math.cos(angle) * dx - Math.sin(angle) * dy + midX;
        const ry = Math.sin(angle) * dx + Math.cos(angle) * dy + midY;
        polygon.push([rx, ry]);
      }

      ctx.beginPath();
      ctx.moveTo(polygon[0][0], polygon[0][1]);
      for (let i = 1; i < polygon.length; i++) {
        ctx.lineTo(polygon[i][0], polygon[i][1]);
      }
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.fill();
    });

    this.kps.forEach((kp, idx) => {
      const [x, y] = kp;
      const color = color_list[idx].replace('0.6', '1');
      ctx.beginPath();
      ctx.arc(x, y, 10, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    });
    ctx.restore();
  }

  drawMoveAll () {
    const pad = 20
    const w = 60
    const ctx = this.canvas.getContext('2d')
    let x = this.kps.reduce((a, b) => a[0] > b[0] ? a : b)[0] + pad
    let y = this.kps.reduce((a, b) => a[1] > b[1] ? a : b)[1] + pad

    ctx.beginPath();
    ctx.fillStyle = "rgb(8, 105, 216)";

    ctx.beginPath();
    ctx.rect(x, y, w, w);
    ctx.fill();

    ctx.font = '40px Arial';
    ctx.fillStyle = 'white';
    ctx.textAlign = 'center';
    ctx.fillText('M', x + 30, y + 40);
  }

  drawImage (ref_image) {
  if (WEBGL2_SUPPORTED) {
    const gl = this.imageCanvas.getContext("webgl2");
    this.drawImageWebGL2(gl, ref_image)
    return
  }
  const ctx = this.imageCanvas.getContext("2d")
  ctx.drawImage(ref_image, 0, 0);
  }

  drawImageWebGL2 (gl, image) {
    const program = gl.createProgram();
    const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSrc);
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSrc);

    if (!vertexShader || !fragmentShader) {
      return;
    }

    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error(gl.getProgramInfoLog(program));
      return;
    }

    gl.useProgram(program);

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([
        -1,  1,  0, 1,
        -1, -1,  0, 0,
         1,  1,  1, 1,
         1, -1,  1, 0,
      ]),
      gl.STATIC_DRAW
    );

    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 4 * 4, 0)
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 4 * 4, 2 * 4)
    gl.enableVertexAttribArray(1);

    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.activeTexture(gl.TEXTURE0);
    gl.uniform1i(gl.getUniformLocation(program, 'uSampler'), 0)

    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, this.imageCanvas.width, this.imageCanvas.height, 0, gl.RGB, gl.UNSIGNED_BYTE, image);
    gl.generateMipmap(gl.TEXTURE_2D);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
}

/*
webgl2 is used to render the reference background
with masked images it is impossible to get pixel values
*/
const WEBGL2_SUPPORTED = checkWebGlSupport()

app.registerExtension({
  name: "ComfyUI.instantid-faceswap",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeType.comfyClass == "KpsMaker") {
      nodeType.prototype.showImage = async function () {
        const img = new Image();
        let name = this.imageWidget.value
        let folder_separator = name.lastIndexOf("/");
        let subfolder = "faceswap_controls";
        if (folder_separator > -1) {
          subfolder = name.substring(0, folder_separator);
          name = name.substring(folder_separator + 1);
        }
        img.src = api.apiURL(
          `/view?filename=${name}&type=input&subfolder=${subfolder}`
        )
        await img.decode();
        this.imgs = [img];
        this.setSizeForImage();
        app.graph.setDirtyCanvas(true);
      }

      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function() {
        const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
        this.imageWidget = this.widgets.find(w => w.name === "image");
        this.imageWidget.callback = this.showImage.bind(this)
        this.imageWidget.disabled = true
        this.kps = {array: []}

        requestAnimationFrame(() => {
          if (this.imageWidget.value) {
            this.showImage();
          }
        })

        const angleWidget = this.addWidget("string", "angle", "", () => {})

        const openDialogWidget = this.addWidget("button", "drawbtn", "", () => {
          let w = this.widgets[1].value
          let h = this.widgets[2].value
          let reference_image
          if (this.getInputNode(0)?.imgs?.length) {
            const inputNode = this.getInputNode(0)
            reference_image = inputNode.imgs[inputNode.imageIndex]
            w = reference_image.width
            h = reference_image.height
          }
          if (w > 0 && h > 0) {
            new KPSDialog(w, h, this.imageWidget, this.kps, reference_image, angleWidget)
          }
        });
        openDialogWidget.label = "draw kps";
        angleWidget.label = "angle: "
        angleWidget.value = "none"
        angleWidget.disabled = true
    }
    this.serialize = true;

      const onConnectionsChange = nodeType.prototype.onConnectionsChange;
      nodeType.prototype.onConnectionsChange = function (side, slot, connect, link_info, output) {
        const r = onConnectionsChange?.apply(this, arguments);
        if (output.name === "image_reference") {
          const widthWidget = this.widgets.find(w => w.name === "width");
          const heightWidget = this.widgets.find(w => w.name === "height");
          const angleWidget = this.widgets.find(w => w.name === "angle");

          this.imgs = []
          this.imageWidget.value = ""
          if (output.link) {
            widthWidget.disabled = true
            heightWidget.disabled = true
            if (this.getInputNode(0)?.imgs?.length) {
              const inputNode = this.getInputNode(0)
              const reference_image = inputNode.imgs[inputNode.imageIndex]
              widthWidget.value = reference_image.width
              heightWidget.value = reference_image.height
            }
          } else {
            widthWidget.disabled = false
            heightWidget.disabled = false
          }

          if (angleWidget) {
            angleWidget.value = "none"
          }
        }
        return r;
      }
    }
  }
})