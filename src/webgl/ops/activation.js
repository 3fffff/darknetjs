export function getGlActivation(type) {
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
  return {
    funcActivation: `float ${type}(float v) {
    return v >= 0.0 ? v: (exp(v) - 1.0) * ${alpha}; /* float number format */
  }`, nameActivation: `value = ${type}(value);`
  }
}
function glslLeakyRelu(type) {
  const alpha = 0.1
  return {
    funcActivation: `
  float ${type}(float v) {
    return v < 0.0 ? v * float(${alpha}) : v;
  }
  `, nameActivation: `value = ${type}(value);`
  };
}
function glslRelu(type) {
  return {
    funcActivation: `
  float ${type}(float v) {
    return v < 0.0 ? 0.0 : v;
  }
  `, nameActivation: `value = ${type}(value);`
  }
}
function glslSigmoid(type) {
  return {
    funcActivation: `
  float ${type}(float v) {
    return 1.0 / (1.0 + exp(-v));
  }
  `, nameActivation: `value = ${type}(value);`
  };
}
function glslSwish(type) {
  return {
    funcActivation: `
  float ${type}(float v) {
    return v*(1.0 / (1.0 + exp(-v));
  }
  `, nameActivation: `value = ${type}(value);`
  };
}
function glslTanh(type) {
  return {
    funcActivation: `
  float ${type}(float v){
    v = clamp(v, -10., 10.);
    v = exp(2.*v);
    return (v - 1.) / (v + 1.);
  }
  `, nameActivation: `value = ${type}(value);`
  };
}
function glslSoftplus(type) {
  const threshold = 20
  return {
    funcActivation: `
  float ${type}() {
    float threshold = float(${threshold});
    if (v > threshold) return v;           
    else if (v < -threshold) return exp(v); 
    else return log(exp(v) + 1.0);
  }`, nameActivation: `value = ${type}(value);`
  }
}
function glslMish(type) {
  const threshold = 20
  return {
    funcActivation: `
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
  `, nameActivation: `value = ${type}(value);`
  }
}