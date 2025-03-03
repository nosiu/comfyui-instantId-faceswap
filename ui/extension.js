import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { drawKps, normalizePoints, rotatePoints3D, getImgFromInput, getDefaultKpsData } from "./helpers.js"
import { KPSDialog2d, KPSDialog3d } from "./dialogs.js"


app.registerExtension({
  getCustomWidgets(app) {
    return {
      HIDDEN_STRING_JSON(node, inputName, inputData) {
        const widget = {
            type: inputData[0],
            name: inputName,
            async serializeValue() {
              return JSON.stringify(widget.value)
            }
        }
        node.addCustomWidget(widget)
        return  widget
      }
    }
  },
  name: "ComfyUI.instantid-faceswap",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeType.comfyClass == "KpsDraw" || nodeType.comfyClass == "Kps3dFromImage") {
      nodeType.prototype.showImage = function () {
        let w = this.widgets.find(w => w.name === "width").value
        let h = this.widgets.find(w => w.name === "height").value

        if (w > 0 && h > 0) {
          let kpsWidget = this.widgets.find(w => w.name === "kps")
          let kps = kpsWidget.value.array
          let kps_opacities = kpsWidget.value.opacities

          if (kps?.length === 0) {
            try {
              const parsed_kps = JSON.parse(kpsWidget.value)
              kps = parsed_kps.array
              if (parsed_kps.opacities && parsed_kps.opacities.length) {
                kps_opacities = parsed_kps.opacities
              }
            } catch(e) {
              console.log(e)
              return;
            }
          }
          if (kps) {
            const c = document.createElement("canvas")
            c.width = w
            c.height = h
            if (kpsWidget.value.rotateX || kpsWidget.value.rotateY || kpsWidget.value.rotateZ) {
              kps = rotatePoints3D(kps.map(el => [...el]), kps, kpsWidget.value.rotateX, kpsWidget.value.rotateY, kpsWidget.value.rotateZ)
            }
            drawKps(c, kps, kps_opacities)
            const image = new Image()
            image.src = c.toDataURL()
            this.imgs = [image]
            this.setSizeForImage()
            app.graph.setDirtyCanvas(true)
          }
        }
      }

      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function() {
        const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
        this.kpsJsonWidget = this.widgets.find(w => w.name === "kps")
        this.kpsJsonWidget.callback = this.showImage.bind(this)
        if (this.kpsJsonWidget.value == null) {
          this.kpsJsonWidget.value = getDefaultKpsData()
        }

        requestAnimationFrame(() => {
          if (this.kpsJsonWidget.value?.array?.length) {
            this.showImage();
          }
        })

        const angleWidget = this.addWidget("string", "angle", "", () => {})

        if (nodeType.comfyClass == "Kps3dFromImage") {

          const div = document.createElement("div")
          div.style.fontSize = "12px"
          div.style.backgroundColor = "#323334"
          div.style.padding = "8px"
          div.innerText = "";
          
          this.addDOMWidget("info_text2", "", div, {getMaxHeight: () => 50})
          
          const doMagic = this.addWidget("button", "getKPS", "", () => {
            const inputNode = getImgFromInput(this.getInputNode(0))

            const reference_image = inputNode.imgs[inputNode.imageIndex || 0].currentSrc
            div.style.color = "white"
            div.innerText = "Getting landmarks ..."

            doMagic.disabled = true
            openDialogWidget.disabled = true
            api.fetchApi("/get_keypoints_for_instantId", {
                method: "POST",
                headers: {
                  "Content-Type": "application/json",
                },
                body: JSON.stringify({image: reference_image})
              }).then(async (data) => {
                const json = await data.json()
                if (json.error) {
                  throw Error(json.error)
                }
                const normalizedPoints = normalizePoints(
                  [
                    ...json.data.jawline,
                    ...json.data.eyebrow_left,
                    ...json.data.eyebrow_right,
                    ...json.data.nose_bridge,
                    ...json.data.nose_lower,
                    ...json.data.eye_left,
                    ...json.data.eye_right,
                    ...json.data.mouth_outer,
                    ...json.data.mouth_inner,
                    ...json.data.kps
                  ],
                  this.widgets.find(w => w.name === "width").value, this.widgets.find(w => w.name === "height").value
                )

                this.kpsJsonWidget.value = getDefaultKpsData()

                this.kpsJsonWidget.value.jawline = normalizedPoints.slice(0, 17)
                this.kpsJsonWidget.value.eyebrow_left = normalizedPoints.slice(17, 22)
                this.kpsJsonWidget.value.eyebrow_right = normalizedPoints.slice(22, 27)
                this.kpsJsonWidget.value.nose_bridge = normalizedPoints.slice(27, 31)
                this.kpsJsonWidget.value.nose_lower = normalizedPoints.slice(31, 36)
                this.kpsJsonWidget.value.eye_left = normalizedPoints.slice(36, 42)
                this.kpsJsonWidget.value.eye_right = normalizedPoints.slice(42, 48)
                this.kpsJsonWidget.value.mouth_outer = normalizedPoints.slice(48, 60)
                this.kpsJsonWidget.value.mouth_inner = normalizedPoints.slice(60, 68)

                this.kpsJsonWidget.value.array = [
                  normalizedPoints[normalizedPoints.length - 5],
                  normalizedPoints[normalizedPoints.length - 4],
                  normalizedPoints[normalizedPoints.length - 3],
                  normalizedPoints[normalizedPoints.length - 2],
                  normalizedPoints[normalizedPoints.length - 1]
                ]
                this.kpsJsonWidget.value.width = this.widgets.find(w => w.name === "width").value
                this.kpsJsonWidget.value.height = this.widgets.find(w => w.name === "height").value

                this.kpsJsonWidget.value.defaultKpsData = JSON.stringify(this.kpsJsonWidget.value) 
                div.style.color = "#08a85a"
                div.innerText = "Success"

                this.showImage()

              }).catch(e => {
                div.style.color = "#C70039"
                div.innerText = "ERROR"
                div.innerText = e.message || "ERROR"
                console.log(e)
              }).finally(() => {
                doMagic.disabled = false
                openDialogWidget.disabled = false
              })
          })
          doMagic.label = "Get Kps From Image";
        }


        const openDialogWidget = this.addWidget("button", "drawbtn", "", () => {
          let w = this.widgets.find(w => w.name === "width").value
          let h = this.widgets.find(w => w.name === "height").value
          let reference_image
          const inputNode = getImgFromInput(this.getInputNode(0))
          if (inputNode?.imgs?.length && nodeType.comfyClass != "Kps3dFromImage") {
            reference_image = inputNode.imgs[inputNode.imageIndex || 0]
            w = reference_image.width
            h = reference_image.height
          }

          if (w > 0 && h > 0) {
            if (nodeType.comfyClass == "Kps3dFromImage") {
              new KPSDialog3d(w, h, angleWidget,  this.kpsJsonWidget)
            } else {
              new KPSDialog2d(
                w, h, reference_image, angleWidget, this.kpsJsonWidget
              )
            }
          }
        });

        const buttonText = nodeType.comfyClass === "KpsDraw" ? "draw kps" : "change kps"

        openDialogWidget.label = buttonText
        angleWidget.label = "angle: "
        angleWidget.value = "none"
        angleWidget.disabled = true
      }
      this.serialize = true

      const onConnectionsChange = nodeType.prototype.onConnectionsChange;
      nodeType.prototype.onConnectionsChange = function (side, slot, connect, link_info, output) {
        const r = onConnectionsChange?.apply(this, arguments);

        if (output.name === "image_reference" && nodeType.comfyClass == "KpsDraw") {
          const widthWidget = this.widgets.find(w => w.name === "width");
          const heightWidget = this.widgets.find(w => w.name === "height");
          const angleWidget = this.widgets.find(w => w.name === "angle");

          this.imgs = []
          if (output.link) {
            widthWidget.disabled = true
            heightWidget.disabled = true
            const inputNode = getImgFromInput(this.getInputNode(0))
            if (inputNode?.imgs?.length) {
              const reference_image = inputNode.imgs[inputNode.imageIndex || 0]
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

        if (output.name === "image" && nodeType.comfyClass == "Kps3dFromImage") {
          this.imgs = []
          const getKPSWidget = this.widgets.find(w => w.name === "getKPS")
          getKPSWidget.disabled = !!!output.link
        }
        return r;
      }
    }
  }
})