async function runInference() {
    const status = document.getElementById('status');
    status.innerText = "Loading Model...";
    
    try {
        // Create inference session
        // Note: You must run export_onnx.py first to generate the .onnx file
        const session = await ort.InferenceSession.create('./vit_tiny.onnx', { executionProviders: ['wasm'] });
        
        status.innerText = "Model Loaded. Running Inference...";
        
        // Create dummy input (Batch: 1, RGB: 3, Size: 32x32)
        const data = Float32Array.from({length: 1 * 3 * 32 * 32}, () => Math.random());
        const tensor = new ort.Tensor('float32', data, [1, 3, 32, 32]);
        
        // Run
        const feeds = { input: tensor };
        const results = await session.run(feeds);
        const output = results.output.data;
        
        status.innerText = `Success! Output class logits: [${output.slice(0,5)}...]`;
        
    } catch (e) {
        status.innerText = `Error: ${e.message}`;
        console.error(e);
    }
}