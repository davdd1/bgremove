# bgremove

Remove the background from a photo without uploading it. Everything runs in the browser, so the image never leaves your device, and it keeps working offline after the first visit.

Live: https://davdd1.github.io/bgremove/

![bgremove with the sample photo processed](docs/screenshot.png)

## Run it

    python3 -m http.server 8000 --bind 127.0.0.1

Open http://127.0.0.1:8000. Tests: `node --test tests/lib.test.mjs`.

## What is in here

- `models/`: U2-Netp (objects) and MODNet (people), both Apache-2.0, so commercial use is fine. RMBG is left out on purpose, its license is non-commercial.
- `vendor/`: ONNX Runtime Web 1.22.0 (MIT) and Motion 12.23.12 (MIT, `motion.js`, used for the small animations; skipped when reduced motion is on).
- `samples/dog.jpg`: CC0, "Boxer (dog), Iran 08.jpg" by Mostafa Meraji, Wikimedia Commons.
- `sw.js`: caches everything for offline use. Bump `CACHE` when you change a file.
