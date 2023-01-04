export function upsample(layers) {
	const l = this
	const input = layers[l.index - 1].output
	upsampleNearest(l, input)
}
function upsampleNearest(l, input, forward = true) {
	for (let b = 0; b < l.batch; ++b)
		for (let k = 0; k < l.c; ++k)
			for (let j = 0; j < l.h * l.stride; ++j)
				for (let i = 0; i < l.w * l.stride; ++i)
					if (forward) l.output[b * l.w * l.h * l.c * l.stride * l.stride + k * l.w * l.h * l.stride * l.stride + j * l.w * l.stride + i] =
						l.scale * input[b * l.w * l.h * l.c + k * l.w * l.h + ~~(j / l.stride) * l.w + ~~(i / l.stride)];
					else input[b * l.w * l.h * l.c + k * l.w * l.h + ~~(j / l.stride) * l.w + ~~(i / l.stride)] =
						l.scale * l.output[b * l.w * l.h * l.c * l.stride * l.stride + k * l.w * l.h * l.stride * l.stride + j * l.w * l.stride + i];
}