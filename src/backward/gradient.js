export function gradient(x, a, delta) {
    switch (a) {
      case "LOGISTIC":
        for (let i = 0; i < x.length; i++)delta[i] *= logistic(x[i])
        return delta;
      case "RELU":
        for (let i = 0; i < x.length; i++)delta[i] *= relu(x[i])
        return delta;
      case "ELU":
        for (let i = 0; i < x.length; i++)delta[i] *= elu(x[i])
        return delta;
      case "LEAKY":
        for (let i = 0; i < x.length; i++)delta[i] *= leaky(x[i])
        return delta;
    }
  }
  function relu(x) { return (x > 0); }
  function elu(x) { return (x >= 0) + (x < 0) * (x + 1); }
  function leaky(x) { return (x > 0) ? 1 : 0.1; }
  function logistic(x) { return (1 - x) * x; }
  function loggy(x) {
    const y = (x + 1.0) / 2.0;
    return 2 * (1 - y) * y;
  }