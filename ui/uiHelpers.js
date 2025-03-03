export const createSlider = (name, id, min, max, step, value, callback) => {
    const divElement = document.createElement("div");
    divElement.id = id;
    divElement.style.cssFloat = "left"
    divElement.style.fontFamily = "sans-serif"
    divElement.style.marginRight = "4px"
    divElement.style.color = "var(--input-text)"
    divElement.style.backgroundColor = "var(--comfy-input-bg)"
    divElement.style.borderRadius = "8px"
    divElement.style.borderColor = "var(--border-color)"
    divElement.style.borderStyle = "solid"
    divElement.style.fontSize = "15px"
    divElement.style.height = "21px"
    divElement.style.padding = "1px 6px"
    divElement.style.display = "flex"
    divElement.style.position = "relative"
    divElement.style.top = "2px"
    divElement.style.pointerEvents = "auto"

    const input = document.createElement("input")
    const labelElement = document.createElement("label")
    input.setAttribute("type", "range")
    input.setAttribute("min", `${min}`)
    input.setAttribute("max", `${max}`)
    input.setAttribute("step", `${step}`)
    input.setAttribute("value", `${value}`)
    labelElement.textContent = name;
    divElement.appendChild(labelElement)
    divElement.appendChild(input)
    input.addEventListener("input", callback)
    return divElement;
  }

  export const createButton = (name, isRight, callback) => {
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

  export const createRadiobox = (name, label, opacities, index, callback) => {
    const div = document.createElement("div");
    div.style.marginTop = "20px"
    const sliderInput = document.createElement("input")
    sliderInput.style.pointerEvents = "auto"
    sliderInput.id = `opacity_slider_${name}`
    sliderInput.type = "range"
    sliderInput.step = "0.1"
    sliderInput.min = "0"
    sliderInput.max = "1"
    sliderInput.tabIndex = "1"
    sliderInput.style.width = "100%"
    sliderInput.name = `s_${name}`
    sliderInput.value = opacities[index]
    sliderInput.addEventListener("change", (event) => {
      const input = document.querySelector(`#opacity_input_${name}`)
      if (input) input.value = event.target.value;
      opacities[index] = event.target.value;
      callback()
    })

    const valueInput = document.createElement("input")
    valueInput.style.pointerEvents = "auto"
    valueInput.id = `opacity_input_${name}`
    valueInput.type = "number";
    valueInput.min = "0";
    valueInput.max = "1";
    valueInput.tabIndex = "1"
    valueInput.style.width = "100%"
    valueInput.name = `i_${name}`
    valueInput.value = opacities[index];
    valueInput.addEventListener("change", (event) => {
      const input = document.querySelector(`#opacity_slider_${name}`)
      if (input) input.value = event.target.value
      opacities[index] = event.target.value
      callback()
    })

    div.style.marginRight = "4px";
    const labelDiv = document.createElement("div")
    labelDiv.innerText = label
    div.appendChild(labelDiv)
    div.appendChild(sliderInput)
    div.appendChild(valueInput)
    return div
  }