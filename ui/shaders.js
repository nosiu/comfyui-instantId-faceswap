export const vertexShaderSrc = `#version 300 es
#pragma vscode_glsllint_stage: vert
layout(location=0) in vec4 aPosition;
layout(location=1) in vec2 aTexCoord;
out vec2 vTexCoord;
void main()
{
  vTexCoord = aTexCoord;
    gl_Position = aPosition;
}`;

export const fragmentShaderSrc = `#version 300 es
#pragma vscode_glsllint_stage: frag
precision mediump float;
in vec2 vTexCoord;
uniform sampler2D uSampler;
out vec4 fragColor;
void main()
{
    fragColor = texture(uSampler, vTexCoord);
}`;


export const createShader = (gl, type, source) => {
  const shader = gl.createShader(type)
  gl.shaderSource(shader, source)
  gl.compileShader(shader)
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error(gl.getShaderInfoLog(shader))
    gl.deleteShader(shader)
    return null
  }
  return shader
}