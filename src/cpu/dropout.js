export function dropout(layers) {
    const l = this
    l.output = layers[l.index - 1].output
    if (!l.train) return;
    for (i = 0; i < l.batch * l.inputs; ++i) {
        const r = Math.random();
        l.rand[i] = r;
        if (r < l.probability) layers[l.index - 1].delta[i] = 0;
        else layers[l.index - 1].delta[i] *= l.scale;
    }
}