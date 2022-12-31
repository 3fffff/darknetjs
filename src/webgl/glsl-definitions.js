
class GlslContext {
  constructor(glContext, programInfo) {
    this.glContext = glContext;
    this.programInfo = programInfo;
  }
}
class GlslLib {
  constructor(context) {
    this.context = context;
  }
}
// abstraction to represent a GLSL library routine and it's dependencies
class GlslLibRoutine {
  constructor(routineBody, dependencies) {
    this.routineBody = routineBody;
    this.dependencies = dependencies;
  }
}
// abstraction to represent a GLSL library routine and it's dependencies AS GRAPH Nodes
// this level of abstraction is used to topologically sort routines before fragment shade inclusion
export class GlslLibRoutineNode {
  constructor(name, routineBody, dependencies) {
    this.name = name;
    if (dependencies) {
      this.dependencies = dependencies;
    }
    else {
      this.dependencies = [];
    }
    if (routineBody) {
      this.routineBody = routineBody;
    }
  }
  addDependency(node) {
    if (node) {
      this.dependencies.push(node);
    }
  }
}
// topologically sort GLSL library routines (graph nodes abstraction) before shader script inclusion
class TopologicalSortGlslRoutines {
  static returnOrderedNodes(nodes) {
    if (!nodes || nodes.length === 0) {
      return [];
    }
    if (nodes.length === 1) {
      return nodes;
    }
    const cycleCheck = new Set();
    const alreadyTraversed = new Set();
    const result = new Array();
    this.createOrderedNodes(nodes, cycleCheck, alreadyTraversed, result);
    return result;
  }
  static createOrderedNodes(graphNodes, cycleCheck, alreadyTraversed, result) {
    for (let i = 0; i < graphNodes.length; ++i) {
      this.dfsTraverse(graphNodes[i], cycleCheck, alreadyTraversed, result);
    }
  }
  static dfsTraverse(root, cycleCheck, alreadyTraversed, result) {
    // if this root has already been traversed return
    if (!root || alreadyTraversed.has(root.name)) {
      return;
    }
    // cyclic dependency has been detected
    if (cycleCheck.has(root.name)) {
      throw new Error(`Cyclic dependency detected. Can't topologically sort routines needed for shader.`);
    }
    // hold this node to detect cycles if any
    cycleCheck.add(root.name);
    // traverse children in a dfs fashion
    const dependencies = root.dependencies;
    if (dependencies && dependencies.length > 0) {
      for (let i = 0; i < dependencies.length; ++i) {
        this.dfsTraverse(dependencies[i], cycleCheck, alreadyTraversed, result);
      }
    }
    // add to result holder
    result.push(root);
    // mark this node as traversed so that we don't traverse from this again
    alreadyTraversed.add(root.name);
    // release the hold
    cycleCheck.delete(root.name);
  }
}