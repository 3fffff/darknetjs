import { cost } from './layers/cost.js'
import { avgpool } from './layers/avgpool.js'
import { connected } from './layers/connected.js'
import { convolutional } from './layers/conv.js'
import { crop } from './layers/crop.js'
import { deconvolutional } from './layers/deconv.js'
import { dropout } from './layers/dropout.js'
import { maxpool } from './layers/maxpool.js'
import { route } from './layers/route.js'
import { sam } from './layers/sam.js'
import { scale_channels } from './layers/scale_channels.js'
import { shortcut } from './layers/shortcut.js'
import { softmax } from './layers/softmax.js'
import { upsample } from './layers/upsample.js'
import { yolo } from './layers/yolo.js'

export const layersDefinition = {
  'CONVOLUTIONAL': convolutional,
  'DECONVOLUTIONAL': deconvolutional,
  'CONNECTED': connected,
  'SOFTMAX': softmax,
  'MAXPOOL': maxpool,
  'AVGPOOL': avgpool,
  'ROUTE': route,
  'YOLO': yolo,
  'SHORTCUT': shortcut,
  'DROPOUT': dropout,
  'UPSAMPLE': upsample,
  'SCALE_CHANNELS': scale_channels,
  'SAM': sam,
  'CROP': crop,
  'COST': cost
}
