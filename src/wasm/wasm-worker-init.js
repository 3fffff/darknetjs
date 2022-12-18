export function initWorkers(numWorkers = 4, initTimeout = 5000) {
  let workers, numWebWorkers, completeCallbacks, initializing, initialized;
  return function () {
    if (initialized) {
      return Promise.resolve();
    }
    if (initializing) {
      throw new Error(`multiple calls to 'init()' detected.`);
    }
    initializing = true;
    return new Promise((resolve, reject) => {
      // the timeout ID that used as a guard for rejecting binding init.
      // we set the type of this variable to unknown because the return type of function 'setTimeout' is different
      // in node.js (type Timeout) and browser (number)
      let waitForBindingInitTimeoutId;
      const clearWaitForBindingInit = () => {
        if (waitForBindingInitTimeoutId !== undefined) {
          // tslint:disable-next-line:no-any
          clearTimeout(waitForBindingInitTimeoutId);
          waitForBindingInitTimeoutId = undefined;
        }
      };
      const onFulfilled = () => {
        clearWaitForBindingInit();
        resolve();
        initializing = false;
        initialized = true;
      };
      const onRejected = (err) => {
        clearWaitForBindingInit();
        reject(err);
        initializing = false;
      };
      //  const bindingInitTask = bindingCore.init();
      // a promise that gets rejected after 5s to work around the fact that
      // there is an unrejected promise in the wasm glue logic file when
      // it has some problem instantiating the wasm file
      const rejectAfterTimeOutPromise = new Promise((resolve, reject) => {
        waitForBindingInitTimeoutId = setTimeout(() => {
          reject('Wasm init promise failed to be resolved within set timeout');
        }, initTimeout);
      });
      // user requests positive number of workers
      if (numWorkers > 0) {
        console.log('WebAssembly-Workers', `User has requested ${numWorkers} Workers.`);
        // check if environment supports usage of workers
        if (areWebWorkersSupported()) {
          console.log('WebAssembly-Workers', `Environment supports usage of Workers. Will spawn ${numWorkers} Workers`);
          WORKER_NUMBER = numWorkers;
        }
        else {
          console.error('WebAssembly-Workers', 'Environment does not support usage of Workers. Will not spawn workers.');
          WORKER_NUMBER = 0;
        }
      }
      // user explicitly disables workers
      else {
        console.log('WebAssembly-Workers', 'User has disabled usage of Workers. Will not spawn workers.');
        WORKER_NUMBER = 0;
      }
      const workerInitTasks = new Array(WORKER_NUMBER);
      workers = new Array(WORKER_NUMBER);
      completeCallbacks = new Array(WORKER_NUMBER);
      numWebWorkers = WORKER_NUMBER
      for (let workerId = 0; workerId < WORKER_NUMBER; workerId++) {
        const workerInitTask = new Promise((resolveWorkerInit, rejectWorkerInit) => {
          // tslint:disable-next-line
          const worker = new Worker("wasm/wasm-worker.js");
          workers[workerId] = worker;
          completeCallbacks[workerId] = [];
          worker.onerror = e => {
            console.error('WebAssembly-Workers', `worker-${workerId} ERR: ${e}`);
            /*if (initialized) {
                // TODO: we need error-handling logic
            }
            else {*/
            rejectWorkerInit();
            // }
          };
          worker.onmessage = e => {
            if (e && e.data && e.data.type) {
              if (e.data.type === 'init-success') {
                resolveWorkerInit();
              }
              else if (e.data.type === 'ccall') {
                const perfData = e.data.perfData;
                completeCallbacks[workerId].shift()(e.data.buffer, perfData);
              }
              else {
                throw new Error(`unknown message type from worker: ${e.data.type}`);
              }
            }
            else {
              throw new Error(`missing message type from worker`);
            }
          };
        });
        workerInitTasks[workerId] = workerInitTask;
      }
      // TODO: Fix this hack to work-around the fact that the Wasm binding instantiate promise
      // is unrejected incase there is a fatal exception (missing wasm file for example)
      // we impose a healthy timeout (should not affect core framework performance)
      Promise.race([Module, rejectAfterTimeOutPromise])
        .then(() => {
          // Wasm init promise resolved
          Promise.all(workerInitTasks)
            .then(
              // Wasm AND Web-worker init promises resolved. SUCCESS!!
              onFulfilled,
              // Wasm init promise resolved. Some (or all) web-worker init promises failed to be resolved.
              // PARTIAL SUCCESS. Use Wasm backend with no web-workers (best-effort).
              (e) => {
                console.warning('WebAssembly-Workers', `Unable to get all requested workers initialized. Will use Wasm backend with 0 workers. ERR: ${e}`);
                // TODO: need house-keeping logic to cull exisitng successfully initialized workers
                // WORKER_NUMBER = 0;
                onFulfilled();
              });
        },
          // Wasm init promise failed to be resolved. COMPLETE FAILURE. Reject this init promise.
          onRejected);
    });
  }
}

function areWebWorkersSupported() {
  if (typeof window !== 'undefined' && typeof window.Worker !== 'undefined') return true;
  return false;
}