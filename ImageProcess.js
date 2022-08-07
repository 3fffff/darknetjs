class ImageLoader {
  constructor(imageWidth, imageHeight, quant) {
    this.canvas = document.createElement('canvas');
    this.canvas.width = imageWidth;
    this.canvas.height = imageHeight;
    this.ctx = this.canvas.getContext('2d');
    this.quant = quant
  }
  loadImage(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.addEventListener('load', () => resolve(img));
      img.addEventListener('error', reject);
      img.src = url;
    });
  }
  async getImageData(url) {
    const img = await this.loadImage(url);
    this.ctx.drawImage(img, 0, 0)
    const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
    this.pixels = ImageProcess.preprocess(imageData.data, 768, 576, this.quant ? [127, 127, 127] : [0, 0, 0], [0, 1, 2], this.quant ? 1 : 1 / 255, false, this.quant);
  }
  get context() {
    return this.ctx
  }
}

class ImageProcess {
  static preprocess(data, width, height, mean = [0, 0, 0], c = [2, 1, 0], scale = 1, inv = false, quant) {
    const vol = quant ? new Int8Array(width * height * 3) : new Float32Array(width * height * 3);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const pp = (y * width + x) * 4;
        const pd = (y * width + x) * 3
        for (let i = 0; i < 3; i++)vol[pd + i] = (((inv ? 255 - data[pp + c[i]] : data[pp + c[i]]) - mean[c[i]]) * scale);
      }
    }
    const channels = 3, res = quant ? new Int8Array(width * height * 3) : new Float32Array(width * height * channels);
    for (let k = 0; k < channels; ++k)
      for (let j = 0; j < height; ++j)
        for (let i = 0; i < width; ++i)
          res[i + width * j + width * height * k] = vol[k + channels * i + channels * width * j];
    return res;
  }

  get_yolo_detections(l, w, h, netw, neth, thresh, relative, dets, letter) {
    let predictions = l.output;
    let count = 0;
    for (let i = 0; i < l.w * l.h; ++i) {
      const row = i / l.w;
      const col = i % l.w;
      for (let n = 0; n < l.filters; ++n) {
        const obj_index = this.entry_index(l, 0, n * l.w * l.h + i, 4);
        const objectness = predictions[obj_index];
        if (objectness > thresh) {
          const box_index = this.entry_index(l, 0, n * l.w * l.h + i, 0);
          dets[count].bbox = this.get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w * l.h);
          dets[count].objectness = objectness;
          dets[count].classes = l.classes;
          for (let j = 0; j < l.classes; ++j) {
            const class_index = this.entry_index(l, 0, n * l.w * l.h + i, 4 + 1 + j);
            const prob = objectness * predictions[class_index];
            dets[count].prob[j] = (prob > thresh) ? prob : 0;
          }
          ++count;
        }
      }
    }
    this.correct_yolo_boxes(dets, count, w, h, netw, neth, relative, letter);
    return count;
  }
  get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride) {
    let b = {};
    b.x = (i + x[index + 0 * stride]) / lw;
    b.y = (j + x[index + 1 * stride]) / lh;
    b.w = (Math.exp(x[index + 2 * stride]) * biases[2 * n] / w);
    b.h = (Math.exp(x[index + 3 * stride]) * biases[2 * n + 1] / h);
    return b;
  }
  fill_network_boxes(layers, w, h, thresh, relative, dets, letter) {
    for (let j = 1; j < layers.length; ++j)
      if (layers[j].type == "YOLO")
        this.get_yolo_detections(layers[j], w, h, layers[0].w, layers[0].h, thresh, relative, dets, letter);
  }

  get_network_boxes(layers, w, h, thresh, relative, letter) {
    let dets = this.make_network_boxes(layers, thresh);
    this.fill_network_boxes(layers, w, h, thresh, relative, dets, letter);
    return dets;
  }

  correct_yolo_boxes(dets, n, w, h, netw, neth, relative, letter) {
    let new_w = 0;
    let new_h = 0;
    if (letter == 0) {
      if ((netw / w) < (neth / h)) {
        new_w = netw;
        new_h = (h * netw) / w;
      }
      else {
        new_h = neth;
        new_w = (w * neth) / h;
      }
    }
    else {
      new_w = netw;
      new_h = neth;
    }
    for (let i = 0; i < n; ++i) {
      let b = dets[i].bbox;
      b.x = (b.x - (netw - new_w) / 2.0 / netw) / (new_w / netw);
      b.y = (b.y - (neth - new_h) / 2.0 / neth) / (new_h / neth);
      b.w *= netw / new_w;
      b.h *= neth / new_h;
      if (relative != 0) {
        b.x *= w;
        b.w *= w;
        b.y *= h;
        b.h *= h;
      }
      dets[i].bbox = b;
    }
  }

  static get_pixel(m, x, y, c, w, h) {
    return m[c * h * w + y * w + x];
  }

  // image.c
  static set_pixel(m, x, y, c, val, w, h, ci) {
    if (x < 0 || y < 0 || c < 0 || x >= w || y >= h || c >= ci) return;
    m[c * h * w + y * w + x] = val;
  }

  // image.c
  static add_pixel(m, x, y, c, val, w, h) {
    m[c * h * w + y * w + x] += val;
  }
  static resize_image(im, w, h, wi, hi, ci, quant) {
    let resized = quant ? new Int8Array(Math.ceil(w * h * ci)) : new Float32Array(Math.ceil(w * h * ci));
    let part = new Array(w * hi * ci);
    let w_scale = (wi - 1) / (w - 1);
    let h_scale = (hi - 1) / (h - 1);
    for (let k = 0; k < ci; ++k) {
      for (let r = 0; r < hi; ++r) {
        for (let c = 0; c < w; ++c) {
          let val = 0;
          if (c == w - 1 || wi == 1)
            val = ImageProcess.get_pixel(im, wi - 1, r, k, wi, hi);
          else {
            let sx = c * w_scale;
            let ix = Math.ceil(sx);
            let dx = sx - ix;
            val = ((1 - dx) * ImageProcess.get_pixel(im, ix, r, k, wi, hi) + dx * ImageProcess.get_pixel(im, ix + 1, r, k, wi, hi));
          }
          ImageProcess.set_pixel(part, c, r, k, val, w, hi, ci);
        }
      }
    }
    for (let k = 0; k < ci; ++k) {
      for (let r = 0; r < h; ++r) {
        const sy = r * h_scale;
        const iy = Math.ceil(sy);
        const dy = sy - iy;
        for (let c = 0; c < w; ++c) {
          const val = ((1 - dy) * ImageProcess.get_pixel(part, c, iy, k, w, hi));
          ImageProcess.set_pixel(resized, c, r, k, val, w, h, ci);
        }
        if (r == h - 1 || hi == 1) continue;
        for (let c = 0; c < w; ++c) {
          const val = (dy * ImageProcess.get_pixel(part, c, iy + 1, k, w, hi));
          ImageProcess.add_pixel(resized, c, r, k, val, w, h);
        }
      }
    }
    return resized;
  }

  make_network_boxes(layers, thresh) {
    const l = layers[layers.length - 1];
    const nboxes = this.num_detections(layers, thresh);
    const dets = new Array(nboxes);
    for (let i = 0; i < nboxes; ++i) {
      dets[i] = {};
      dets[i].prob = new Float32Array(l.classes);
      if (l.coords > 4)
        dets[i].mask = new Float32Array(l.coords - 4);
    }
    return dets;
  }
  yolo_num_detections(l, thresh) {
    let count = 0;
    for (let i = 0; i < l.w * l.h; ++i) {
      for (let n = 0; n < l.filters; ++n) {
        let obj_index = this.entry_index(l, 0, n * l.w * l.h + i, 4);
        if (l.output[obj_index] > thresh)
          ++count;
      }
    }
    return count;
  }
  entry_index(l, batch, location, entry) {
    return batch * l.outputs + Math.floor(location / (l.w * l.h)) * l.w * l.h * (4 + l.classes + 1) + entry * l.w * l.h + Math.floor(location % (l.w * l.h));
  }

  num_detections(layers, thresh) {
    let s = 0;
    for (let i = 0; i < layers.length; ++i)
      if (layers[i].type == "YOLO")
        s += this.yolo_num_detections(layers[i], thresh);
    return s;
  }
}
