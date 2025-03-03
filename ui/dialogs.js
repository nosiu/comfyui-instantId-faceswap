import { createShader, vertexShaderSrc, fragmentShaderSrc } from "./shaders.js"
import { getPointsCenter, drawKps, checkWebGlSupport, rotatePoints3D } from "./helpers.js"
import { createSlider, createButton, createRadiobox } from "./uiHelpers.js"

class KPSDialogBase {
    constructor(w, h, img = undefined) {
        this.isDragging = false
        this.draggedPointIndex = null
        this.mousedown_x = undefined
        this.mousedown_y = undefined
        this.mousedown_pan_x = undefined
        this.mousedown_pan_y = undefined
        this.pan_x = 0
        this.pan_y = 0
        this.cursorX = undefined
        this.cursorY = undefined
        this.zoom_ratio = 1
        this.min_zoom = undefined
        this.showOpacities = false
        if (img) {
          this.canvasWidth = img.width
          this.canvasHeight = img.height
        } else {
          this.canvasWidth = w
          this.canvasHeight = h
        }

        // ---------------------------------
        this.element = document.createElement("div")
        this.element.style.display = "none"
        this.element.style.width = "80vw"
        this.element.style.height = "80vh"
        this.element.style.zIndex = 8888
        this.element.classList.add('comfy-modal')
        this.element.classList.add('kps-sandbox')

        document.body.appendChild(this.element)

        this.canvas = document.createElement("canvas")
        this.canvas.style.position = "absolute"
        this.canvas.style.pointerEvents = "auto"
        this.canvas.style.zIndex = "-1"
        this.element.appendChild(this.canvas)

        this.canvas.width = this.canvasWidth
        this.canvas.height = this.canvasHeight
    }

    initializeCanvasPanZoom () {
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
  
      if (this.hasImage) {
        this.imageCanvas.style.width = `${raw_width}px`;
        this.imageCanvas.style.height = `${raw_height}px`;
        this.imageCanvas.style.left = `${this.pan_x}px`;
        this.imageCanvas.style.top = `${this.pan_y}px`;
      }
    }

    setBasicControls () {
      const buttonBar = document.createElement("div")
      buttonBar.id = "instantIdButtonBar"
      buttonBar.style.position = "absolute"
      buttonBar.style.bottom = "0"
      buttonBar.style.height = "50px"
      buttonBar.style.left = "20px"
      buttonBar.style.right = "20px"
      buttonBar.style.pointerEvents = "none"
      buttonBar.appendChild(createButton("Save", true, this.save.bind(this)))
      buttonBar.appendChild(createButton("Cancel", true, this.closeModal.bind(this)))
      buttonBar.appendChild(createButton("Reset pan & zoom", false, () => {
        this.initializeCanvasPanZoom();
      }));
      buttonBar.appendChild(createButton("Reset KPS", false, () => {
        this.kps = this.getDefaultKps();
        this.draw();
      }));

      buttonBar.appendChild(this.createZoomSlider())
  
      this.element.appendChild(buttonBar);
      // ----------------------
      const opacitiesButton = createButton(
        "Opacity options", false, () => {
          this.showOpacities = !this.showOpacities
          if (this.showOpacities) {
            opacitiesButton.innerText = "Hide options"
            const advancedDiv = document.createElement("div")
            advancedDiv.style.overflow = "auto"
            advancedDiv.id = "kpsDialog0"
            advancedDiv.style.padding = "20px"
            advancedDiv.style.paddingTop = "50px"
            advancedDiv.style.width = "200px"
            advancedDiv.style.height = "100%"
            advancedDiv.style.position = "absolute"
            advancedDiv.style.display = "flex"
            advancedDiv.style.left = "0"
            advancedDiv.style.top = "0"
            advancedDiv.style.backgroundColor = "black"
            advancedDiv.style.color = "white"
  
            const radioBar = document.createElement("div");
            radioBar.style.marginTop = "20px"
            radioBar.style.pointerEvents = "auto"
  
            radioBar.appendChild(createRadiobox("red", "red opacity", this.kpsOpacities, 0, this.draw.bind(this)))
            radioBar.appendChild(createRadiobox("green", "green opacity", this.kpsOpacities, 1, this.draw.bind(this)))
            radioBar.appendChild(createRadiobox("blue", "blue opacity", this.kpsOpacities, 2, this.draw.bind(this)))
            radioBar.appendChild(createRadiobox("yellow", "yellow opacity", this.kpsOpacities, 3, this.draw.bind(this)))
            radioBar.appendChild(createRadiobox("purple", "purple opacity", this.kpsOpacities, 4, this.draw.bind(this)))

            advancedDiv.appendChild(radioBar);
            this.element.appendChild(advancedDiv)
          } else {
            opacitiesButton.innerText = "Opacity options"
            const el = document.querySelector("#kpsDialog0")
            if (el) el.remove()
          }
        }
      )
      opacitiesButton.style.zIndex = 8889
      opacitiesButton.style.position = "absolute"
      this.element.appendChild(opacitiesButton);
    }

    createZoomSlider () {
      const el = createSlider("Zoom", "instantIdZoomSlider", this.min_zoom, "2", "0.1", this.zoom_ratio, (event) => {
        this.zoom_ratio = parseFloat(event.target.value)
        this.invalidatePanZoom()
      })
      return el
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
}

export class KPSDialog2d extends KPSDialogBase{
  constructor(w, h, referenceImage, angleWidget, kpsJsonWidget) {
    super(w, h, referenceImage)

    this.hasImage = referenceImage ? true  : false
    if (this.hasImage) {
      this.opacity = "0.6"
    } else {
      this.opacity = "1"
    }

    this.kpsOpacities = kpsJsonWidget.value.opacities.length ? JSON.parse(JSON.stringify(kpsJsonWidget.value.opacities)) : [1, 1, 1, 1, 1]
    this.kps = kpsJsonWidget.value.array.length ? JSON.parse(JSON.stringify(kpsJsonWidget.value.array)) : this.getDefaultKps()

    this.angleWidget = angleWidget
    this.kpsJsonWidget = kpsJsonWidget
    
    if (this.hasImage) {
      this.imageCanvas = document.createElement("canvas")
      this.imageCanvas.style.position = "absolute"
      this.imageCanvas.style.zIndex = "-2"
      this.imageCanvas.style.pointerEvents = "none"
      this.element.appendChild(this.imageCanvas)
      this.imageCanvas.width = this.canvasWidth
      this.imageCanvas.height = this.canvasHeight
    }

    this.setBasicControls()
    this.setControls()
    this.attachListeners()
    this.canvas.style.opacity = this.opacity
    this.element.style.display = "block"
    this.initializeCanvasPanZoom()

    this.draw()
    if (this.hasImage) this.drawImage(referenceImage)
  }

  getDefaultKps () {
    const halfWidth = this.canvasWidth / 2;
    const halfHeight = this.canvasHeight / 2;
    return [
      [halfWidth - halfWidth / 2, halfHeight - halfHeight / 2],
      [halfWidth + halfWidth / 2, halfHeight - halfHeight / 2],
      [halfWidth, halfHeight],
      [halfWidth - halfWidth / 2, halfHeight + halfHeight / 2],
      [halfWidth + halfWidth / 2, halfHeight + halfHeight / 2],
    ]
  }

  setControls () {
    const buttonBar = document.querySelector("#instantIdButtonBar")
    if (this.hasImage) {
      buttonBar.appendChild(this.createOpacitySlider())
    }
  }

  createOpacitySlider () {
    const el = createSlider("Opacity", "instantIdOpacitySlider", "0.1", "1", "0.1", this.opacity, (event) => {
      this.opacity = event.target.value
      this.canvas.style.opacity = event.target.value
    })
    return el
  }

  attachListeners () {
    this.canvas.addEventListener('mousedown', this.mouseDown.bind(this))
    this.canvas.addEventListener('mousemove', this.mouseMove.bind(this))
    this.canvas.addEventListener('mouseup', this.mouseUp.bind(this))
    this.element.addEventListener('wheel', this.wheel.bind(this))
    this.element.addEventListener('DOMMouseScroll', (e) => e.preventDefault()) // thanks firefox.
    this.element.addEventListener('keydown', (event) => {
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

  async save () {

    const minX = Math.min(...this.kps.map(e => e[0]))
    const maxX = Math.max(...this.kps.map(e => e[0]))

    const minY = Math.min(...this.kps.map(e => e[1]))
    const maxY = Math.max(...this.kps.map(e => e[1]))

    this.kpsJsonWidget.value = {
      array: this.kps,
      opacities: this.kpsOpacities,
      width: this.canvasWidth,
      height: this.canvasHeight,
      bbox: [
        [
          Math.max(Math.ceil(minX - ((maxX - minX) /3)), 0),
          Math.max(Math.ceil(minY - ((maxY - minY) /3)), 0)
        ],
        [
          Math.min(Math.ceil(maxX + ((maxX - minX) /3)), this.canvasWidth),
          Math.min(Math.ceil(maxY + ((maxY - minY) /3)), this.canvasHeight)
        ],       
      ]
    }

    this.kpsJsonWidget.callback()
    const a = this.kps[0]
    const b = this.kps[1]
    let angle = Math.atan2(b[1] - a[1], b[0] - a[0]) * 180 / Math.PI

    this.angleWidget.value = angle
    this.closeModal()
  }

  changePointsPosition(closer = false, step = 10) {
    step /= this.zoom_ratio;
    const center = this.kps[2]

    const points = this.kps

    const magnitudes = points.map((point, index) => {
        if (index === 2) return Infinity; // Skip the center point (no magnitude calculation)
        const direction = [center[0] - point[0], center[1] - point[1]]
        return Math.sqrt(direction[0] * direction[0] + direction[1] * direction[1])
    });

    const allGreaterThan50 = magnitudes.every(mag => mag > 10)

    return points.map((point, index) => {
        if (index === 2) return point
        const direction = [center[0] - point[0], center[1] - point[1]]
        const magnitude = magnitudes[index]

        const scaleFactor = closer ? (magnitude - step) / magnitude : (magnitude + step) / magnitude;
        const scaledMoveVector = [direction[0] * scaleFactor, direction[1] * scaleFactor];

        if (closer) {
            if (allGreaterThan50) {
                return [
                    center[0] - scaledMoveVector[0],
                    center[1] - scaledMoveVector[1]
                ];
            }
            return point
        } else {
            return [
                center[0] - scaledMoveVector[0],
                center[1] - scaledMoveVector[1]
            ];
        }
    });
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
    const { offsetX: mouseX, offsetY: mouseY } = event
    this.cursorX = event.pageX
    this.cursorY = event.pageY
    if (event.ctrlKey) {
      if (event.buttons == 1) {
        if (this.mousedown_x) {
          let deltaX = this.mousedown_x - event.clientX
          let deltaY = this.mousedown_y - event.clientY
          this.pan_x = this.mousedown_pan_x - deltaX
          this.pan_y = this.mousedown_pan_y - deltaY
          this.invalidatePanZoom()
        }
      }
    }
    if (this.isDragging) {
      const transformedX = (mouseX) / this.zoom_ratio
      const transformedY = (mouseY) / this.zoom_ratio
      if(this.draggedPointIndex !== null && this.draggedPointIndex > -1) {
          this.kps[this.draggedPointIndex] = [transformedX, transformedY]
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
      this.draw()
    }
  }

  mouseUp (event) {
    event.preventDefault()
    this.mousedown_x = null
    this.mousedown_y = null
    this.isDragging = false
    this.draggedPointIndex = null
  }

  wheel (event) {
    event.preventDefault()
    if (event.ctrlKey) {
      if (event.deltaY < 0) {
        this.zoom_ratio = Math.min(2, this.zoom_ratio + 0.2)
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
    this.drawKeyPoints()
    this.drawMoveAll()
  }


  drawKeyPoints (canvas = this.canvas) {
    drawKps(canvas, this.kps, this.kpsOpacities)
  }

  drawImage (ref_image) {
    /*
    webgl2 is used to render the reference background
    with masked images it is impossible to get pixel values
    */
    if (checkWebGlSupport()) {
        const gl = this.imageCanvas.getContext("webgl2");
        this.drawImageWebGL2(gl, ref_image)
        return
    }
    const ctx = this.imageCanvas.getContext("2d")
    ctx.drawImage(ref_image, 0, 0);
  }

  drawImageWebGL2 (gl, image) {
    const program = gl.createProgram()
    const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSrc)
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSrc)

    if (!vertexShader || !fragmentShader) {
      return;
    }

    gl.attachShader(program, vertexShader)
    gl.attachShader(program, fragmentShader)
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error(gl.getProgramInfoLog(program))
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
    )

    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 4 * 4, 0)
    gl.enableVertexAttribArray(0)
    gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 4 * 4, 2 * 4)
    gl.enableVertexAttribArray(1)

    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.activeTexture(gl.TEXTURE0)
    gl.uniform1i(gl.getUniformLocation(program, 'uSampler'), 0)

    const texture = gl.createTexture()
    gl.bindTexture(gl.TEXTURE_2D, texture)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, this.imageCanvas.width, this.imageCanvas.height, 0, gl.RGB, gl.UNSIGNED_BYTE, image)
    gl.generateMipmap(gl.TEXTURE_2D)

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
  }
}


export class KPSDialog3d extends KPSDialogBase{
  constructor(w, h, angleWidget, kpsJsonWidget) {
    super(w, h)

    this.showLandmarks = false

    this.defaultKpsData = kpsJsonWidget.value.defaultKpsData

    this.kpsOpacities = kpsJsonWidget.value.opacities.length ? JSON.parse(JSON.stringify(kpsJsonWidget.value.opacities)) : [1, 1, 1, 1, 1]
    this.kps = JSON.parse(JSON.stringify(kpsJsonWidget.value.array))

    this.angleWidget = angleWidget
    this.kpsJsonWidget = kpsJsonWidget
    
   
    const landmarks = [
      'jawline', 'eyebrow_left', 'eyebrow_right', 'nose_bridge', 'nose_lower',
      'eye_left', 'eye_right', 'mouth_outer', 'mouth_inner'
    ]
    landmarks.forEach(el => {
      this[el] = kpsJsonWidget.value[el].length ? JSON.parse(JSON.stringify(kpsJsonWidget.value[el])) : []
    })

    this.rotateX = kpsJsonWidget.value.rotateX || 0
    this.rotateY = kpsJsonWidget.value.rotateY || 0
    this.rotateZ = kpsJsonWidget.value.rotateZ || 0

    this.setBasicControls()
    this.setControls()
    this.attachListeners()

    this.element.style.display = "block"
    this.initializeCanvasPanZoom()

    this.draw()
    if (this.hasImage) this.drawImage(referenceImage)
  }

  getDefaultKps () {
    try {
      const data = JSON.parse(this.defaultKpsData)
      const landmarks = [
        'jawline', 'eyebrow_left', 'eyebrow_right', 'nose_bridge', 'nose_lower',
        'eye_left', 'eye_right', 'mouth_outer', 'mouth_inner'
      ]
      landmarks.forEach(el => {
        this[el] = data[el]
      })

      return data.array
    } catch (e) {
      console.error(e)
    }
  }

  setControls () {
    const buttonBar = document.querySelector("#instantIdButtonBar")
    buttonBar.appendChild(this.createRotationXSlider())
    buttonBar.appendChild(this.createRotationYSlider())
    buttonBar.appendChild(this.createRotationZSlider())

    this.element.appendChild(buttonBar);
  }

  createRotationXSlider () {
    const el = createSlider("rotate X", "instantIdRotateX", "0", "360", "1", this.rotateX, (event) => {
      this.rotateX = parseFloat(event.target.value)
      this.draw()
    })
    return el
  }

  createRotationYSlider () {
    const el = createSlider("rotate Y", "instantIdRotateY", "0", "360", "1", this.rotateY, (event) => {
      this.rotateY = parseFloat(event.target.value)
      this.draw()
    })
    return el
  }

  createRotationZSlider () {
    const el = createSlider("rotate Z", "instantIdRotateZ", "0", "360", "1", this.rotateZ, (event) => {
      this.rotateZ = parseFloat(event.target.value)
      this.draw()
    })
    return el
  }

  attachListeners () {
    this.canvas.addEventListener('mousedown', this.mouseDown.bind(this))
    this.canvas.addEventListener('mousemove', this.mouseMove.bind(this))
    this.canvas.addEventListener('mouseup', this.mouseUp.bind(this))
    this.element.addEventListener('wheel', this.wheel.bind(this))
    this.element.addEventListener('DOMMouseScroll', (e) => e.preventDefault()) // thanks firefox.
    this.element.addEventListener('keydown', (event) => {
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

  async save () {
    this.kpsJsonWidget.value = {
      array: this.kps,
      opacities: this.kpsOpacities.map(el => parseFloat(el)),
      width: this.canvasWidth,
      height: this.canvasHeight,
      jawline: this.jawline,
      eyebrow_left: this.eyebrow_left,
      eyebrow_right: this.eyebrow_right,
      nose_bridge: this.nose_bridge,
      nose_lower: this.nose_lower,
      eye_left: this.eye_left,
      eye_right: this.eye_right,
      mouth_outer: this.mouth_outer,
      mouth_inner: this.mouth_inner,
      rotateX: this.rotateX,
      rotateY: this.rotateY,
      rotateZ: this.rotateZ,
      defaultKpsData: this.defaultKpsData
    }
 
    this.kpsJsonWidget.callback()
    const a = this.kps[0]
    const b = this.kps[1]
    let angle = Math.atan2(b[1] - a[1], b[0] - a[0]) * 180 / Math.PI

    this.angleWidget.value = angle
    this.closeModal()
  }

  changePointsPosition(closer = false, step = 10) {
    step /= this.zoom_ratio

    const center = getPointsCenter([
      ...this.kps, ...this.jawline, ...this.eyebrow_left, ...this.eyebrow_right, ...this.nose_bridge,
      ...this.nose_lower,...this.eye_left, ...this.eye_right, ...this.mouth_outer, ...this.mouth_inner
    ])

    const points = [
      ...this.kps, ...this.jawline, ...this.eyebrow_left, ...this.eyebrow_right, ...this.nose_bridge,
      ...this.nose_lower,...this.eye_left, ...this.eye_right, ...this.mouth_outer, ...this.mouth_inner
    ]

    const magnitudes = points.map(point => {
        const direction = [center[0] - point[0], center[1] - point[1], center[2] - point[2]]
        return Math.sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2])
    })

    const allGreaterThan50 = magnitudes.every(mag => mag > 10)

    points.forEach((point, index) => {
        const direction = [center[0] - point[0], center[1] - point[1], center[2] - point[2]]
        const magnitude = magnitudes[index]
        const unitVector = [direction[0] / magnitude, direction[1] / magnitude, direction[2] / magnitude]

        const scaleFactor = magnitude / Math.max(...magnitudes)
        const adjustedStep = step * scaleFactor
        const moveVector = [unitVector[0] * adjustedStep, unitVector[1] * adjustedStep, unitVector[2] * adjustedStep]

        if (closer) {
            if (allGreaterThan50) {
                point[0] += moveVector[0]
                point[1] += moveVector[1]
                point[2] += moveVector[2]
            }
        } else {
            point[0] -= moveVector[0]
            point[1] -= moveVector[1]
            point[2] -= moveVector[2]
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

          this.kps.forEach((kp) => {
            let [x, y] = kp
            x *= this.zoom_ratio
            y *= this.zoom_ratio
            maxX = x > maxX ? x : maxX
            maxY = y > maxY ? y : maxY
          });
          maxX += 20 * this.zoom_ratio
          maxY += 20 * this.zoom_ratio
          if ((mouseX >= maxX) && (mouseY >= maxY) && (mouseX < maxX + 60 * this.zoom_ratio) && (mouseY < maxY + 60 * this.zoom_ratio)){
            this.mousedown_x = event.clientX
            this.mousedown_y = event.clientY
            this.isDragging = true;
            this.draggedPointIndex = -1
          }
        }
  }

  mouseMove (event) {
    event.preventDefault();
    this.cursorX = event.pageX
    this.cursorY = event.pageY
    if (event.ctrlKey) {
      if (event.buttons == 1) {
        if (this.mousedown_x) {
          let deltaX = this.mousedown_x - event.clientX
          let deltaY = this.mousedown_y - event.clientY
          this.pan_x = this.mousedown_pan_x - deltaX
          this.pan_y = this.mousedown_pan_y - deltaY
          this.invalidatePanZoom()
        }
      }
    }
    if (this.isDragging) {
      let deltaX = this.mousedown_x - event.clientX
      let deltaY = this.mousedown_y - event.clientY
      this.mousedown_x = event.clientX
      this.mousedown_y = event.clientY
      const points = [
        this.kps, this.jawline, this.eyebrow_left, this.eyebrow_right, this.nose_bridge,
        this.nose_lower, this.eye_left, this.eye_right, this.mouth_outer, this.mouth_inner
      ]
      points.forEach(p => {
        p.forEach(el => {
          el[0] -= deltaX / this.zoom_ratio
          el[1] -= deltaY / this.zoom_ratio
        })
      })
      this.draw()
    }
  }

  mouseUp (event) {
    event.preventDefault()
    this.mousedown_x = null
    this.mousedown_y = null
    this.isDragging = false
    this.draggedPointIndex = null
  }

  wheel (event) {
    event.preventDefault()
    if (event.ctrlKey) {
      if (event.deltaY < 0) {
        this.zoom_ratio = Math.min(2, this.zoom_ratio + 0.2)
      } else {
        this.zoom_ratio = Math.max(this.min_zoom, this.zoom_ratio - 0.2)
      }
      document.querySelector("#instantIdZoomSlider input").value = `${this.zoom_ratio}`
      this.invalidatePanZoom();
    }
    else if (event.altKey) {
      this.changePointsPosition(event.deltaY > 0)
      this.draw()
    }
  }

  draw () {
    this.drawKeyPoints()
    const landmarks = [
      this.jawline, this.eyebrow_left, this.eyebrow_right, this.nose_bridge,
      this.nose_lower, this.eye_left, this.eye_right, this.mouth_inner, this.mouth_outer
    ]
    landmarks.forEach(points => { this.drawLandmarks(points) })
    this.drawMoveAll()
  }

  drawLandmarks (p, canvas = this.canvas) {
    if (p.length === 0) return
    const ctx = canvas.getContext('2d')
    const points = rotatePoints3D(p.map(el => [...el]), this.kps, this.rotateX, this.rotateY, this.rotateZ)

    ctx.beginPath()
    ctx.strokeStyle  = "white"

    for (let i = 1; i < points.length; i++) {
        ctx.moveTo(points[i - 1][0], points[i - 1][1]);
        ctx.lineTo(points[i][0], points[i][1]);
    }
    ctx.stroke()
  }

  drawKeyPoints (canvas = this.canvas) {
    const kps = rotatePoints3D(this.kps.map(el => [...el]), this.kps, this.rotateX, this.rotateY, this.rotateZ)
    drawKps(canvas, kps, this.kpsOpacities)
  }
}