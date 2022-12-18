import { cost } from './cpu/cost.js'
import { pool } from './cpu/pool.js'
import { connected } from './cpu/connected.js'
import { convolutional } from './cpu/conv.js'
import { crop } from './cpu/crop.js'
import { deconvolutional } from './cpu/deconv.js'
import { dropout } from './cpu/dropout.js'
import { route } from './cpu/route.js'
import { sam } from './cpu/sam.js'
import { scale_channels } from './cpu/scale_channels.js'
import { shortcut } from './cpu/shortcut.js'
import { softmax } from './cpu/softmax.js'
import { upsample } from './cpu/upsample.js'

export const layersForward = {
  'CONVOLUTIONAL': convolutional,
  'DECONVOLUTIONAL': deconvolutional,
  'CONNECTED': connected,
  'SOFTMAX': softmax,
  'MAXPOOL': pool,
  'AVGPOOL': pool,
  'ROUTE': route,
  'YOLO': dropout,
  'SHORTCUT': shortcut,
  'DROPOUT': dropout,
  'UPSAMPLE': upsample,
  'SCALE_CHANNELS': scale_channels,
  'SAM': sam,
  'CROP': crop,
  'COST': cost
}
