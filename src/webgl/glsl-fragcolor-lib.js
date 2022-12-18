
/**
 * This GLSL library handles routines around reading a texlet and writing to it
 * Reading and writing could be more than just dealing with one channel
 * It may require encoding/decoding to/from 4 channels into one
 */
export class FragColorGlslLib extends GlslLib {
  constructor(context) {
    super(context);
  }
  getFunctions() {
    return Object.assign({}, this.setFragColor(), this.getColorAsFloat());
  }
  setFragColor() {
    const glsl = getGlsl(this.context.glContext.version);
    return {
      setFragColor: new GlslLibRoutine(`
        void setFragColor(float value) {
            ${glsl.output} = encode(value);
        }
        `, ['encoding.encode'])
    };
  }
  getColorAsFloat() {
    return {
      getColorAsFloat: new GlslLibRoutine(`
        float getColorAsFloat(vec4 color) {
            return decode(color);
        }
        `, ['encoding.decode'])
    };
  }
}