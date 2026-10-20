# bgremove

Free background remover that runs fully in the browser (static site, no server).

Run: `python3 -m http.server 8000 --bind 127.0.0.1` in this directory, open http://127.0.0.1:8000.
Test: `node --test tests/lib.test.mjs`

Models (self-hosted in `models/`, both Apache-2.0, commercial use allowed):
- `u2netp.onnx` 4.4 MB: U2-Net-small, https://github.com/xuebinqin/U-2-Net (ONNX from rembg release v0.0.0)
- `modnet.onnx` 6.3 MB: MODNet quantized, https://github.com/ZHKKKe/MODNet (ONNX from huggingface Xenova/modnet)

Runtime in `vendor/`: onnxruntime-web 1.22.0 (MIT). Sample `samples/dog.jpg`: CC0, Wikimedia Commons "Boxer (dog) , Iran 08.jpg" (Mostafa Meraji).
RMBG-1.4/2.0 are avoided (non-commercial licenses).
