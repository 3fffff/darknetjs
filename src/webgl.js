import { pool } from './webgl/ops/pool.js'
import { connected } from './webgl/ops/matmul.js'
import { convolutional } from './webgl/ops/conv.js'
import { route } from './webgl/ops/route.js'
import { scale_channels } from './webgl/ops/scale_channels.js'
import { softmax } from './webgl/ops/softmax.js'
import { upsample } from './webgl/ops/upsample.js'

export const layersWebGL = {
  'CONVOLUTIONAL': convolutional,
  'CONNECTED': connected,
  'SOFTMAX': softmax,
  'MAXPOOL': pool,
  'AVGPOOL': pool,
  'ROUTE': route,
  'UPSAMPLE': upsample,
  'SCALE_CHANNELS': scale_channels,
  'YOLO': (webgl, l) => { }
}
