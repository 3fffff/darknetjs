
/**
 * Preprocessor for the additions to the GLSL language
 * It deals with:
 *  @include directives
 *  @inline
 *  Loop unrolling (not implemented)
 *  Macro resolution (not implemented)
 */
export class GlslPreprocessor {
  constructor(glContext, programInfo) {
    this.libs = {};
    this.glslLibRoutineDependencyGraph = {};
    this.context = new GlslContext(glContext, programInfo);
    const glslRegistry = {
      'encoding':EncodingGlslLib,
      'fragcolor': FragColorGlslLib,
      'vec': VecGlslLib,
      'shapeUtils': ShapeUtilsGlslLib,
      'coordinates': CoordsGlslLib,
    };
    // construct GlslLibs
    Object.keys(glslRegistry).forEach((name) => {
      const lib = new glslRegistry[name](this.context);
      this.libs[name] = lib;
    });
    // construct GlslRoutineDependencyGraph
    const map = this.glslLibRoutineDependencyGraph;
    for (const libName in this.libs) {
      const lib = this.libs[libName];
      const routinesInLib = lib.getFunctions();
      for (const routine in routinesInLib) {
        const key = libName + '.' + routine;
        let currentNode;
        if (map[key]) {
          currentNode = map[key];
          currentNode.routineBody = routinesInLib[routine].routineBody;
        }
        else {
          currentNode = new GlslLibRoutineNode(key, routinesInLib[routine].routineBody);
          map[key] = currentNode;
        }
        const dependencies = routinesInLib[routine].dependencies;
        if (dependencies) {
          for (let i = 0; i < dependencies.length; ++i) {
            if (!map[dependencies[i]]) {
              const node = new GlslLibRoutineNode(dependencies[i]);
              map[dependencies[i]] = node;
              currentNode.addDependency(node);
            }
            else {
              currentNode.addDependency(map[dependencies[i]]);
            }
          }
        }
      }
    }
  }
  preprocess() {
    const programInfo = this.context.programInfo;
    let source = programInfo.shaderSource;
    // append main() function
    if (!this.context.programInfo.hasMain) {
      source = `${source}
      ${getDefaultFragShaderMain(this.context.glContext.version, programInfo.outputLayout.shape.length)}`;
    }
    // replace inlines
    source = replaceInlines(source);
    // concat final source string
    return `${getFragShaderPreamble(this.context.glContext.version)}
    ${this.getUniforms(programInfo.samplers, programInfo.variables)}
    ${this.getImports(source)}
    ${source}`;
  }
  getImports(script) {
    const routinesIncluded = this.selectGlslLibRoutinesToBeIncluded(script);
    if (routinesIncluded.length === 0) {
      return '';
    }
    let routines = ``;
    for (let i = 0; i < routinesIncluded.length; ++i) {
      if (routinesIncluded[i].routineBody) {
        routines += routinesIncluded[i].routineBody + `\n`;
      }
      else {
        throw new Error(`Missing body for the Glsl Library routine: ${routinesIncluded[i].name}`);
      }
    }
    return routines;
  }
  selectGlslLibRoutinesToBeIncluded(script) {
    const nodes = [];
    Object.keys(this.glslLibRoutineDependencyGraph).forEach(classAndRoutine => {
      const routine = classAndRoutine.split('.')[1];
      if (script.indexOf(routine) !== -1) {
        nodes.push(this.glslLibRoutineDependencyGraph[classAndRoutine]);
      }
    });
    return TopologicalSortGlslRoutines.returnOrderedNodes(nodes);
  }
  getUniforms(samplers, variables) {
    const uniformLines = [];
    if (samplers) {
      for (const sampler of samplers) {
        uniformLines.push(`uniform sampler2D ${sampler};`);
      }
    }
    if (variables) {
      for (const variable of variables) {
        uniformLines.push(`uniform ${variable.type} ${variable.name}${variable.arrayLength ? `[${variable.arrayLength}]` : ''};`);
      }
    }
    return uniformLines.join('\n');
  }
}
/**
 * GLSL preprocessor responsible for resolving @inline directives
 */
 const INLINE_FUNC_DEF_REGEX = /@inline[\s\n\r]+(\w+)[\s\n\r]+([0-9a-zA-Z_]+)\s*\(([^)]*)\)\s*{(([^}]|[\n\r])*)}/gm;
 const FUNC_CALL_REGEX = '(\\w+)?\\s+([_0-9a-zA-Z]+)\\s+=\\s+__FUNC__\\((.*)\\)\\s*;';
 function replaceInlines(script) {
   const inlineDefs = {};
   let match;
   while ((match = INLINE_FUNC_DEF_REGEX.exec(script)) !== null) {
     const params = match[3]
       .split(',')
       .map(s => {
         const tokens = s.trim().split(' ');
         if (tokens && tokens.length === 2) {
           return { type: tokens[0], name: tokens[1] };
         }
         return null;
       })
       .filter(v => v !== null);
     inlineDefs[match[2]] = { params, body: match[4] };
   }
   for (const name in inlineDefs) {
     const regexString = FUNC_CALL_REGEX.replace('__FUNC__', name);
     const regex = new RegExp(regexString, 'gm');
     while ((match = regex.exec(script)) !== null) {
       const type = match[1];
       const variable = match[2];
       const params = match[3].split(',');
       const declLine = (type) ? `${type} ${variable};` : '';
       let newBody = inlineDefs[name].body;
       let paramRedecLine = '';
       inlineDefs[name].params.forEach((v, i) => {
         if (v) {
           paramRedecLine += `${v.type} ${v.name} = ${params[i]};\n`;
         }
       });
       newBody = `${paramRedecLine}\n ${newBody}`;
       newBody = newBody.replace('return', `${variable} = `);
       const replacement = `
       ${declLine}
       {
         ${newBody}
       }
       `;
       script = script.replace(match[0], replacement);
     }
   }
   script = script.replace(INLINE_FUNC_DEF_REGEX, '');
   return script;
 }