"use strict";
function getGlActivation(type) {
  switch (type) {
    case "LOGISTIC":
      return glslSigmoid(type)
    case "RELU":
      return glslRelu(type);
    case "LEAKY":
      return glslLeakyRelu(type)
    case "MISH":
      return glslMish(type)
    case "SWISH":
      return glslSwish(type)
    default: throw new Error("not recognized activation");
  }
}
function glslElu(type) {
  const alpha = Math.exp(0.1)
  return {function:`float ${type}(float v) {
    return v >= 0.0 ? v: (exp(v) - 1.0) * ${alpha}; /* float number format */
  }`,call:`${type}(v);`}
}
function glslLeakyRelu(type) {
  const alpha = 0.1
  return {func:`
  float ${type}(float v) {
    return v < 0.0 ? v * float(${alpha}) : v;
  }
  `,call:`${type}(value);`};
}
function glslRelu(type) {
  return {func:`
  float ${type}(float v) {
    return v < 0.0 ? 0.0 : v;
  }
  `,call:`${type}(value);`}
}
function glslSigmoid(type) {
  return `
  float ${type}(float v) {
    return 1.0 / (1.0 + exp(-v));
  }
  `;
}
function glslSwish(type) {
  return `
  float ${type}(float v) {
    return v*(1.0 / (1.0 + exp(-v));
  }
  `;
}
function glslTanh(type) {
  return `
  float ${type}(float v){
    v = clamp(v, -10., 10.);
    v = exp(2.*v);
    return (v - 1.) / (v + 1.);
  }
  `;
}
function glslSoftplus(type) {
  const threshold = 20
  return `
  float ${type}() {
    float threshold = float(${threshold});
    if (v > threshold) ${glsl.output} = vec4(v);           
    else if (v < -threshold) ${glsl.output} = vec4(exp(v)); 
    else ${glsl.output} = vec4(log(exp(v) + 1.0));
  }`
}

function glslMish(type) {
  const threshold = 20
  return `
  float softplus(float v){
    float threshold = float(${threshold});
    if (v > threshold) v = v;           
    else if (v < -threshold) v = exp(v); 
    else v = log(exp(v) + 1.0);
    return v;
  }
  float tang(float v){
    v = clamp(v, -10., 10.);
    v = exp(2.*v);
    return (v - 1.) / (v + 1.);
  }
  float ${type}(float v) {
    return v * tang(softplus(v));
  }
  `
}