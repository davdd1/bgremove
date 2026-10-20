# Model evaluation

Test set: 4 flat logos, a CC0 portrait with hair, a CC0 product photo on a busy table, the dog sample (sources in kassako-media/bgremove-tests/SOURCES.md).

| Model | License | Size | Result |
|---|---|---|---|
| U2-Netp | Apache-2.0 | 4 MB | eats letters in logos, soft edges. Kept as the offline "Quick" mode |
| MODNet | Apache-2.0 | 6 MB | good on people, empty or broken on everything else. "Person" mode |
| ISNet general use (imgly ONNX) | Apache-2.0 (DIS), MIT (conversion) | 88 / 176 MB | clean on photos and logos, runs on WASM and WebGPU. Default for photos |
| BiRefNet lite (onnx-community) | MIT | 109 / 214 MB | best masks, but cannot run in the browser: WASM runs out of memory (std::bad_alloc at 1024 px, 4 GB wasm limit), WebGPU fails (fp16 needs shader-f16, fp32 needs 17 storage buffers per shader, limit is 16 on the tested RTX 5070) with onnxruntime-web 1.22 and 1.30 |
| BiRefNet (full) | MIT | 467 MB fp16 | marginally better than lite, same limits, too big |
| onnx-community/ISNet-ONNX | AGPL-3.0 | | not used (license) |
| RMBG-1.4 / 2.0 | non-commercial | | not used (license) |

Flat-background cut (color key) beats every model on logos, so Auto uses it when the border is flat.
