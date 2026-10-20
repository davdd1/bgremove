// Cache-first så att verktyget fungerar offline efter första laddningen.
const CACHE = 'bgremove-v2';
const FILES = ['./', 'index.html', 'style.css', 'app.js', 'lib.mjs', 'samples/dog.jpg',
  'vendor/ort.wasm.min.js', 'vendor/motion.js', 'vendor/ort-wasm-simd-threaded.mjs', 'vendor/ort-wasm-simd-threaded.wasm',
  'models/u2netp.onnx', 'models/modnet.onnx'];
self.addEventListener('install', (e) => { e.waitUntil(caches.open(CACHE).then((c) => c.addAll(FILES)).then(() => self.skipWaiting())); });
self.addEventListener('activate', (e) => { e.waitUntil(caches.keys().then((k) => Promise.all(k.filter((x) => x !== CACHE).map((x) => caches.delete(x)))).then(() => self.clients.claim())); });
self.addEventListener('fetch', (e) => { e.respondWith(caches.match(e.request).then((r) => r || fetch(e.request))); });
