importScripts("wforward.js");
importScripts("wasm.js");
onmessage = (e) => {
    if (e && e.data && e.data.type) {
        if (e.data.type === 'ccall') {
            const func = e.data.func;
            const buffer = e.data.buffer;
            const perfData = WasmBinding.getInstance().ccallRaw(func, new Uint8Array(buffer));
            
            postMessage({ type: 'ccall', buffer, perfData }, [buffer]);
        } else {
            throw new Error(`unknown message type from main thread: ${e.data.type}`);
        }
    } else {
        throw new Error(`missing message type from main thread`);
    }
};