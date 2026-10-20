import { MODELS, toTensorData, toMask, sharpen, fitSize, applyAlpha, outName, hexToRgb, fitBackground, isFlat, colorKey, coverage, refineMask, decontaminate } from './lib.mjs';

const $ = (id) => document.getElementById(id);
const ort = window.ort;
ort.env.wasm.wasmPaths = new URL('vendor/', location.href).href;
ort.env.wasm.numThreads = 1; // SHORTCUT: single thread avoids COOP/COEP headers; a service-worker header shim would allow threads (4-8x faster on CPU)
ort.env.wasm.proxy = true; // inference in a worker so the page keeps animating

// Motion is optional: without it (or with reduced motion) everything still works, just without easing.
const M = window.Motion;
const calm = matchMedia('(prefers-reduced-motion: reduce)');
const fancy = () => !!M && !calm.matches;
const spring = { type: 'spring', stiffness: 320, damping: 26 };
const ease = [0.22, 1, 0.36, 1];
const anim = (el, kf, opt) => (fancy() ? M.animate(el, kf, opt) : null);
const tick = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));

const sessions = {};
const items = [];
let queue = Promise.resolve();
let sink = null; // progress callback of the item that is currently being processed

// ---------- models ----------

let gpuOK = null;
async function hasGpu() {
  if (gpuOK !== null) return gpuOK;
  try { gpuOK = !!(navigator.gpu && (await navigator.gpu.requestAdapter())); } catch { gpuOK = false; }
  return gpuOK;
}

// Remote weights are streamed with progress and kept in the Cache API, so the download happens once.
async function fetchModel(url, bytes) {
  let cache = null;
  try { cache = await caches.open('bgremove-models'); const hit = await cache.match(url); if (hit) return new Uint8Array(await hit.arrayBuffer()); } catch { /* no cache available */ }
  const res = await fetch(url);
  if (!res.ok) throw new Error('Model download failed (' + res.status + ')');
  const total = +res.headers.get('content-length') || bytes;
  const rd = res.body.getReader(); const parts = []; let got = 0;
  for (;;) {
    const { done, value } = await rd.read(); if (done) break;
    parts.push(value); got += value.length;
    if (sink) sink({ loaded: got, total });
  }
  const buf = new Uint8Array(got); let o = 0; for (const p of parts) { buf.set(p, o); o += p.length; }
  try { if (cache) await cache.put(url, new Response(buf)); } catch { /* quota: fine, just re-download next time */ }
  return buf;
}

async function makeSession(key) {
  const cfg = MODELS[key];
  if (cfg.file) {
    const buf = await (await fetch(cfg.file)).arrayBuffer();
    return { s: await ort.InferenceSession.create(buf, { executionProviders: ['wasm'] }), ep: 'wasm' };
  }
  if (await hasGpu()) {
    try {
      const buf = await fetchModel(cfg.remote.gpu, cfg.remote.gpuBytes);
      return { s: await ort.InferenceSession.create(buf, { executionProviders: ['webgpu'] }), ep: 'webgpu' };
    } catch (e) { console.warn('WebGPU failed, using CPU', e); }
  }
  const buf = await fetchModel(cfg.remote.cpu, cfg.remote.cpuBytes);
  return { s: await ort.InferenceSession.create(buf, { executionProviders: ['wasm'] }), ep: 'wasm' };
}

function getSession(key) {
  if (!sessions[key]) sessions[key] = makeSession(key).catch((e) => { delete sessions[key]; throw e; });
  return sessions[key];
}

function canvasOf(w, h) { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; }

async function modelMask(bitmap, key, w, h, stage) {
  const cfg = MODELS[key];
  stage('Loading the model', 0);
  let { s: session, ep } = await getSession(key);
  stage('Finding the subject');
  await tick();
  const small = canvasOf(cfg.size, cfg.size);
  const sctx = small.getContext('2d', { willReadFrequently: true });
  sctx.drawImage(bitmap, 0, 0, cfg.size, cfg.size);
  const input = toTensorData(sctx.getImageData(0, 0, cfg.size, cfg.size).data, cfg.size, cfg);
  const feed = () => ({ [session.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, cfg.size, cfg.size]) });
  let res;
  try { res = await session.run(feed()); }
  catch (e) {
    if (ep !== 'webgpu') throw e;
    // some GPUs/drivers cannot run the graph: drop WebGPU for good and redo it on the CPU
    console.warn('WebGPU run failed, using CPU', e); gpuOK = false; delete sessions[key];
    stage('Loading the model', 0); ({ s: session, ep } = await getSession(key)); stage('Finding the subject'); await tick();
    res = await session.run(feed());
  }
  const t = res[session.outputNames[0]];
  const out = t.data ?? (await t.getData());
  stage('Cleaning the edges');
  await tick();
  const m = toMask(out, cfg.minmax, cfg.sigmoid);
  const mc = canvasOf(cfg.size, cfg.size);
  const mimg = mc.getContext('2d').createImageData(cfg.size, cfg.size);
  for (let i = 0; i < m.length; i++) { mimg.data[i * 4] = mimg.data[i * 4 + 1] = mimg.data[i * 4 + 2] = m[i]; mimg.data[i * 4 + 3] = 255; }
  mc.getContext('2d').putImageData(mimg, 0, 0);
  const up = canvasOf(w, h);
  const uctx = up.getContext('2d', { willReadFrequently: true });
  uctx.imageSmoothingQuality = 'high';
  uctx.drawImage(mc, 0, 0, w, h);
  const upPx = uctx.getImageData(0, 0, w, h).data;
  const mask = new Uint8ClampedArray(w * h);
  for (let i = 0; i < mask.length; i++) mask[i] = upPx[i * 4];
  return { mask, label: MODELS[key].label + (ep === 'webgpu' ? ' · WebGPU' : '') };
}

// Edge refinement (guided filter, halo cleanup, island removal). Very large images are refined at reduced size.
function refineBig(mask, rgba, w, h) {
  const cap = 10e6;
  if (w * h <= cap) return refineMask(mask, rgba, w, h);
  const k = Math.sqrt(cap / (w * h)), rw = Math.round(w * k), rh = Math.round(h * k);
  const src = canvasOf(w, h); src.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(rgba), w, h), 0, 0);
  const sm = canvasOf(rw, rh); sm.getContext('2d').drawImage(src, 0, 0, rw, rh);
  const mc = canvasOf(w, h), mi = mc.getContext('2d').createImageData(w, h);
  for (let i = 0; i < mask.length; i++) { mi.data[i * 4] = mi.data[i * 4 + 1] = mi.data[i * 4 + 2] = mask[i]; mi.data[i * 4 + 3] = 255; }
  mc.getContext('2d').putImageData(mi, 0, 0);
  const ms = canvasOf(rw, rh); ms.getContext('2d').drawImage(mc, 0, 0, rw, rh);
  const sd = sm.getContext('2d').getImageData(0, 0, rw, rh).data, md = ms.getContext('2d').getImageData(0, 0, rw, rh).data;
  const small = new Uint8ClampedArray(rw * rh); for (let i = 0; i < small.length; i++) small[i] = md[i * 4];
  const r = refineMask(small, sd, rw, rh);
  const o = canvasOf(rw, rh), oi = o.getContext('2d').createImageData(rw, rh);
  for (let i = 0; i < r.length; i++) { oi.data[i * 4] = oi.data[i * 4 + 1] = oi.data[i * 4 + 2] = r[i]; oi.data[i * 4 + 3] = 255; }
  o.getContext('2d').putImageData(oi, 0, 0);
  const big = canvasOf(w, h); const bx = big.getContext('2d', { willReadFrequently: true }); bx.imageSmoothingQuality = 'high'; bx.drawImage(o, 0, 0, w, h);
  const bd = bx.getImageData(0, 0, w, h).data, out = new Uint8ClampedArray(w * h);
  for (let i = 0; i < out.length; i++) out[i] = bd[i * 4];
  return out;
}

// Returns { cut, orig, mask, method }. mode: auto | portrait | flat | fast
async function segment(bitmap, mode, crisp, stage) {
  const [w, h] = fitSize(bitmap.width, bitmap.height);
  const full = canvasOf(w, h);
  const fctx = full.getContext('2d', { willReadFrequently: true });
  fctx.drawImage(bitmap, 0, 0, w, h);
  let px = fctx.getImageData(0, 0, w, h);
  let mask = null, label = '';

  if (mode === 'auto' || mode === 'flat') {
    stage('Checking the background');
    await tick();
    const fit = fitBackground(px.data, w, h);
    if (mode === 'flat' || isFlat(fit)) {
      const keyed = new Uint8ClampedArray(px.data);
      const km = colorKey(keyed, w, h, fit);
      const cov = coverage(km);
      if (mode === 'flat' || (cov > 0.002 && cov < 0.9)) { px.data.set(keyed); mask = km; label = 'Color key'; }
    }
    if (mask) { stage('Cutting by color'); await tick(); }
  }
  let note = '';
  if (!mask) {
    const key = mode === 'portrait' ? 'portrait' : mode === 'fast' ? 'fast' : 'quality';
    let r;
    try { r = await modelMask(bitmap, key, w, h, stage); }
    catch (e) {
      if (key !== 'quality') throw e;
      console.warn('Quality model unavailable, using Quick', e);
      note = 'offline, used Quick';
      r = await modelMask(bitmap, 'fast', w, h, stage);
    }
    mask = r.mask; label = r.label;
    px = fctx.getImageData(0, 0, w, h);
    stage('Cleaning the edges');
    await tick();
    // SHORTCUT: refinement runs on the main thread at full size up to ~10 MP; bigger images are refined at reduced size (softer edges), move it to a worker if that matters
    mask = refineBig(mask, px.data, w, h);
    if (crisp) mask = sharpen(mask);
    if (w * h <= 10e6) decontaminate(px.data, mask, w, h);
    applyAlpha(px.data, mask);
  } else if (crisp) {
    // colour-keyed edges are already antialiased; nothing to sharpen
  }
  const cut = canvasOf(w, h);
  cut.getContext('2d').putImageData(px, 0, 0);
  // light subjects (white text on a logo) vanish on a pale transparency grid, so the card switches to a darker one
  let sum = 0, n = 0;
  for (let i = 0; i < mask.length; i += 7) if (mask[i] > 200) { sum += (0.2126 * px.data[i * 4] + 0.7152 * px.data[i * 4 + 1] + 0.0722 * px.data[i * 4 + 2]) / 255; n++; }
  return { cut, orig: full, mask, light: n > 20 && sum / n > 0.72, method: label + (note ? ' (' + note + ')' : '') };
}

function composite(it) {
  const { cut, orig } = it;
  const mode = $('bg').value;
  const c = canvasOf(cut.width, cut.height);
  const ctx = c.getContext('2d');
  if (mode === 'color') { const [r, g, b] = hexToRgb($('color').value); ctx.fillStyle = `rgb(${r},${g},${b})`; ctx.fillRect(0, 0, c.width, c.height); }
  else if (mode === 'blur') {
    const px = +$('blur').value * Math.max(c.width, c.height) / 1000;
    ctx.filter = `blur(${px}px)`; ctx.drawImage(orig, -px * 2, -px * 2, c.width + px * 4, c.height + px * 4); ctx.filter = 'none';
  }
  ctx.drawImage(cut, 0, 0);
  return c;
}

function render(it) {
  if (!it.cut) return;
  const out = composite(it);
  out.className = 'after';
  const old = it.el.querySelector('canvas.after');
  if (old) old.replaceWith(out); else it.el.querySelector('.cmp').prepend(out);
  it.out = out;
}

// ---------- cards ----------

function makeCard(it) {
  const el = document.createElement('div');
  el.className = 'card';
  el.innerHTML = `<div class="cmp busy queued">
      <canvas class="pre"></canvas>
      <div class="scan" aria-hidden="true"></div><div class="shimmer" aria-hidden="true"></div>
      <div class="bar" aria-hidden="true"><i></i></div>
      <div class="stage" role="status"><span class="dot" aria-hidden="true"></span><span class="msg">Waiting</span><span class="pct"></span></div>
    </div>
    <div class="row"><span></span><button class="btn" disabled>Download PNG</button></div>`;
  el.querySelector('.row span').textContent = it.name;
  it.el = el;
  el.querySelector('button').onclick = () => download(it);
  $('results').append(el);
  anim(el, { opacity: [0, 1], y: [26, 0], scale: [0.96, 1] }, { ...spring, delay: 0.02 });
}

function drawPreview(it) {
  const c = it.el.querySelector('canvas.pre');
  const [w, h] = fitSize(it.bmp.width, it.bmp.height, 1200);
  c.width = w; c.height = h; c.getContext('2d').drawImage(it.bmp, 0, 0, w, h);
}

function makeStage(it) {
  const cmp = it.el.querySelector('.cmp');
  const msg = cmp.querySelector('.msg'), pct = cmp.querySelector('.pct'), bar = cmp.querySelector('.bar i');
  cmp.classList.remove('queued');
  return (text, frac) => {
    msg.textContent = text;
    if (frac === undefined) { pct.textContent = ''; bar.style.transform = ''; cmp.classList.remove('det'); return; }
    cmp.classList.add('det'); bar.style.transform = `scaleX(${frac})`; pct.textContent = frac ? Math.round(frac * 100) + '%' : '';
  };
}

function attachSlider(it) {
  const cmp = it.el.querySelector('.cmp');
  const before = it.orig; before.className = 'before';
  const handle = document.createElement('div'); handle.className = 'handle';
  handle.innerHTML = '<div class="knob"></div>';
  const knob = handle.firstChild;
  const tl = document.createElement('span'); tl.className = 'tag l'; tl.textContent = 'Before';
  const tr = document.createElement('span'); tr.className = 'tag r'; tr.textContent = 'After';
  cmp.append(before, handle, tl, tr);
  cmp.tabIndex = 0; cmp.setAttribute('role', 'slider'); cmp.setAttribute('aria-label', 'Before and after');
  cmp.setAttribute('aria-valuemin', '0'); cmp.setAttribute('aria-valuemax', '100');
  let pos = 0, sweep = null;
  const put = (p) => {
    pos = Math.min(100, Math.max(0, p));
    before.style.clipPath = `inset(0 ${100 - pos}% 0 0)`; handle.style.left = pos + '%';
    tl.style.opacity = pos < 14 ? 0 : 1; tr.style.opacity = pos > 86 ? 0 : 1;
    cmp.setAttribute('aria-valuenow', Math.round(pos));
  };
  const fromX = (x) => { const r = cmp.getBoundingClientRect(); return (x - r.left) / r.width * 100; };
  const glide = (to) => {
    if (sweep) sweep.stop();
    if (!fancy()) return put(to);
    sweep = M.animate(pos, to, { ...spring, onUpdate: put });
  };
  cmp.addEventListener('pointerdown', (e) => {
    if (sweep) sweep.stop();
    cmp.setPointerCapture(e.pointerId); glide(fromX(e.clientX));
    anim(knob, { scale: 1.18 }, spring);
    cmp.onpointermove = (ev) => { if (sweep) sweep.stop(); put(fromX(ev.clientX)); };
  });
  const up = () => { cmp.onpointermove = null; anim(knob, { scale: 1 }, spring); };
  cmp.addEventListener('pointerup', up); cmp.addEventListener('pointercancel', up);
  cmp.addEventListener('keydown', (e) => {
    const k = { ArrowLeft: -5, ArrowRight: 5, Home: -100, End: 100 }[e.key];
    if (k === undefined) return; e.preventDefault(); glide(pos + k);
  });
  put(fancy() ? 0 : 50);
  return { nudge() { if (!fancy()) return; handle.style.opacity = 0; anim(handle, { opacity: [0, 1] }, { duration: 0.4 }); setTimeout(() => glide(16), 350); setTimeout(() => glide(0), 1000); } };
}

// The signature moment: the background breaks into tiles that shrink and drift away from left to right,
// leaving the subject standing on the transparency grid, with a light edge running ahead of the wave.
function playReveal(it, slider) {
  const cmp = it.el.querySelector('.cmp');
  const after = it.out;
  const long = Math.min(1100, Math.max(after.width, after.height));
  const k = long / Math.max(after.width, after.height);
  const W = Math.max(2, Math.round(after.width * k)), H = Math.max(2, Math.round(after.height * k));
  // peel source: the original with the subject cut out (alpha = 1 - mask)
  const full = canvasOf(it.orig.width, it.orig.height);
  const fx = full.getContext('2d');
  fx.drawImage(it.orig, 0, 0);
  const id = fx.getImageData(0, 0, full.width, full.height);
  for (let i = 0; i < it.mask.length; i++) id.data[i * 4 + 3] = 255 - it.mask[i];
  fx.putImageData(id, 0, 0);
  const src = canvasOf(W, H); src.getContext('2d').drawImage(full, 0, 0, W, H);
  const peel = canvasOf(W, H); peel.className = 'peel';
  const edge = document.createElement('div'); edge.className = 'edge';
  cmp.append(peel, edge);
  const pctx = peel.getContext('2d');
  const cols = 30, ts = W / cols, rows = Math.ceil(H / ts);
  const jit = new Float32Array(cols * rows); for (let i = 0; i < jit.length; i++) jit[i] = Math.random();
  const frame = (p) => {
    pctx.clearRect(0, 0, W, H);
    if (p <= 0) { pctx.drawImage(src, 0, 0); return; }
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const u = (c + 0.5) / cols, v = (r + 0.5) / rows;
      const s = u * 0.62 + jit[r * cols + c] * 0.14 + Math.abs(v - 0.5) * 0.06;
      const t = Math.min(1, Math.max(0, (p - s) / 0.24));
      if (t >= 1) continue;
      const x = c * ts, y = r * ts, sw = Math.min(ts, W - x), sh = Math.min(ts, H - y);
      const e = t * t * (3 - 2 * t), sc = 1 - 0.6 * e;
      pctx.globalAlpha = 1 - e;
      const dw = sw * sc + (t ? 0 : 0.6), dh = sh * sc + (t ? 0 : 0.6);
      pctx.drawImage(src, x, y, sw, sh, x + (sw - dw) / 2 + e * ts * 0.35, y + (sh - dh) / 2 - e * ts * 0.9, dw, dh);
    }
    pctx.globalAlpha = 1;
  };
  frame(0);
  cmp.classList.add('revealing');
  const finish = () => { peel.remove(); edge.remove(); cmp.classList.remove('revealing'); slider.nudge(); };
  return new Promise((res) => {
    M.animate(0, 1, {
      duration: 1.7, ease: [0.5, 0, 0.25, 1],
      onUpdate: (p) => {
        frame(p);
        const front = Math.min(1.12, p / 0.7);
        edge.style.left = front * 100 + '%'; edge.style.opacity = Math.max(0, Math.min(1, (1.08 - front) * 6));
      },
      onComplete: () => { finish(); res(); },
    });
    anim(after, { scale: [1, 1.03, 1], filter: ['drop-shadow(0 0 0 rgba(25,24,21,0))', 'drop-shadow(0 10px 18px rgba(25,24,21,.32))', 'drop-shadow(0 0 0 rgba(25,24,21,0))'] }, { duration: 1.9, ease: 'easeInOut' });
  });
}

function download(it) {
  const a = document.createElement('a');
  it.out.toBlob((b) => {
    a.href = URL.createObjectURL(b);
    a.download = outName(it.name, $('bg').value === 'transparent' ? 'utan-bakgrund' : 'ny-bakgrund');
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  }, 'image/png');
}

function updateStatus() {
  const done = items.filter((i) => i.cut).length; const settled = items.filter((i) => i.cut || i.failed).length;
  $('status').textContent = items.length ? `${done} of ${items.length} done` : '';
  $('dlall').disabled = done === 0;
  window.__done = settled === items.length && items.length > 0;
}

function addFile(file) {
  drop.classList.add('compact'); document.body.classList.add('used');
  const it = { name: file.name || 'image', file };
  items.push(it); makeCard(it); updateStatus();
  const mode = $('model').value, crisp = $('crisp').checked;
  const ready = createImageBitmap(file).then((b) => { it.bmp = b; drawPreview(it); return b; });
  ready.catch(() => {});
  queue = queue.then(async () => {
    const t0 = performance.now();
    const cmp = it.el.querySelector('.cmp');
    const stage = makeStage(it);
    sink = ({ loaded, total }) => stage('Downloading the model, once', Math.min(1, loaded / total));
    try {
      stage('Reading the image');
      const bmp = await ready;
      await tick();
      const r = await segment(bmp, mode, crisp, stage);
      sink = null;
      Object.assign(it, r, { ms: Math.round(performance.now() - t0) });
      const dwell = 1000 - (performance.now() - t0); // let the processing state be seen even when the cut is instant
      if (fancy() && dwell > 0) await new Promise((r) => setTimeout(r, dwell));
      render(it);
      cmp.classList.toggle('dark-grid', !!it.light);
      cmp.classList.remove('busy', 'det');
      cmp.querySelectorAll('.scan,.shimmer,.bar,.stage,canvas.pre').forEach((n) => n.remove());
      const slider = attachSlider(it);
      it.el.querySelector('button').disabled = false;
      it.el.querySelector('.row span').textContent = `${it.name} · ${(it.ms / 1000).toFixed(1)} s · ${it.method}`;
      if (fancy()) await playReveal(it, slider); else slider.nudge();
    } catch (e) {
      sink = null;
      it.failed = true; cmp.className = 'cmp'; cmp.innerHTML = '<div class="ph">Could not process this file.<br><small></small></div>'; cmp.querySelector('small').textContent = String(e.message || e);
      console.error(e);
    }
    updateStatus();
  });
}

function addFiles(list) { for (const f of list) if (f.type.startsWith('image/')) addFile(f); }

// ---------- drop zone ----------

const drop = $('drop');
const frame = drop.querySelector('.frame');
const corners = [...frame.querySelectorAll('i')];
let depth = 0;
const lock = (on) => {
  if (on) {
    anim(frame, { scale: 0.94 }, { type: 'spring', stiffness: 260, damping: 14 });
    anim(drop, { scale: 1.015 }, { type: 'spring', stiffness: 300, damping: 18 });
    anim(corners, { scale: [1.9, 1], rotate: [0, 0] }, { type: 'spring', stiffness: 420, damping: 16, delay: M ? M.stagger(0.04) : 0 });
  } else {
    anim(frame, { scale: 1 }, spring); anim(drop, { scale: 1 }, spring);
  }
};
const setOver = (on) => {
  if (on === drop.classList.contains('over')) return;
  drop.classList.toggle('over', on); document.body.classList.toggle('dragging', on); lock(on);
};
// the whole window accepts a drop, the zone just reacts the loudest
window.addEventListener('dragenter', (e) => { e.preventDefault(); depth++; setOver(true); });
window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('dragleave', () => { depth = Math.max(0, depth - 1); if (!depth) setOver(false); });
window.addEventListener('drop', (e) => {
  e.preventDefault(); depth = 0; setOver(false);
  anim(drop, { scale: [1.03, 1] }, { type: 'spring', stiffness: 500, damping: 14 });
  addFiles(e.dataTransfer.files);
});
drop.addEventListener('pointerenter', () => { if (!drop.classList.contains('over')) anim(frame, { scale: 1.03 }, spring); });
drop.addEventListener('pointerleave', () => { if (!drop.classList.contains('over')) anim(frame, { scale: 1 }, spring); });
document.addEventListener('paste', (e) => addFiles(e.clipboardData?.files || []));
$('file').onchange = (e) => { addFiles(e.target.files); e.target.value = ''; };
const sample = (url, name, type) => async () => { const b = await (await fetch(url)).blob(); addFiles([new File([b], name, { type: b.type || type })]); };
$('sample').onclick = sample('samples/dog.jpg', 'sample.jpg', 'image/jpeg');
$('sample2').onclick = sample('samples/logo.png', 'logo.png', 'image/png');
$('dlall').onclick = () => items.filter((i) => i.out).forEach((i, n) => setTimeout(() => download(i), n * 400)); // SHORTCUT: sequential downloads, no zip; add a zip writer for big batches

// ---------- controls ----------

function reveal(el, show) {
  if (!show) { el.hidden = true; return; }
  const was = el.hidden; el.hidden = false;
  if (was) anim(el, { opacity: [0, 1], x: [-8, 0] }, spring);
}

// a sliding thumb behind the selected segment instead of a hard swap
function placeThumb(seg, instant) {
  const thumb = seg.querySelector('.thumb'); const b = seg.querySelector('[aria-checked=true]');
  if (!thumb || !b || !b.offsetWidth) return;
  const to = { x: b.offsetLeft, width: b.offsetWidth };
  if (instant || !fancy()) { thumb.style.transform = `translateX(${to.x}px)`; thumb.style.width = to.width + 'px'; }
  else M.animate(thumb, { transform: `translateX(${to.x}px)`, width: to.width + 'px' }, spring);
}

document.querySelectorAll('.seg').forEach((seg) => {
  const sel = $(seg.dataset.for);
  const thumb = document.createElement('i'); thumb.className = 'thumb'; seg.prepend(thumb); seg.classList.add('has-thumb');
  placeThumb(seg, true);
  new ResizeObserver(() => placeThumb(seg, true)).observe(seg);
  seg.addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    seg.querySelectorAll('button').forEach((x) => x.setAttribute('aria-checked', String(x === b)));
    sel.value = b.dataset.v; sel.dispatchEvent(new Event('input')); placeThumb(seg);
    anim(b, { scale: [0.94, 1] }, spring);
  });
  seg.addEventListener('keydown', (e) => {
    const d = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key]; if (!d) return;
    const bs = [...seg.querySelectorAll('button')]; const i = bs.findIndex((x) => x.getAttribute('aria-checked') === 'true');
    const n = bs[(i + d + bs.length) % bs.length]; e.preventDefault(); n.click(); n.focus();
  });
});
document.fonts?.ready.then(() => document.querySelectorAll('.seg').forEach((s) => placeThumb(s, true)));

function syncBg() {
  const m = $('bg').value;
  reveal($('colorLabel'), m === 'color'); reveal($('blurLabel'), m === 'blur');
  items.forEach(render);
}
['bg', 'color', 'blur'].forEach((id) => $(id).addEventListener('input', syncBg));

// ---------- entrance ----------

function intro() {
  const root = document.documentElement;
  const h1 = document.querySelector('.hero h1');
  if (h1 && !h1.dataset.split) {
    h1.dataset.split = '1';
    h1.setAttribute('aria-label', h1.textContent);
    h1.innerHTML = h1.textContent.split(' ').map((w) => `<span class="w" aria-hidden="true"><span>${w}</span></span>`).join(' ');
  }
  if (!fancy()) { root.classList.remove('pre'); return; }
  const words = h1.querySelectorAll('.w > span');
  const done = () => root.classList.remove('pre');
  M.animate('.top', { opacity: [0, 1], x: [-10, 0] }, { duration: 0.6, ease });
  M.animate(document.querySelector('.mark'), { rotate: [-180, 0], scale: [0.4, 1] }, { type: 'spring', stiffness: 160, damping: 14 });
  M.animate(words, { y: ['110%', '0%'], rotate: [4, 0] }, { duration: 0.8, delay: M.stagger(0.07, { startDelay: 0.12 }), ease });
  M.animate('.lede', { opacity: [0, 1], y: [14, 0] }, { duration: 0.7, delay: 0.55, ease });
  M.animate('#drop', { opacity: [0, 1], y: [34, 0], scale: [0.94, 1] }, { type: 'spring', stiffness: 150, damping: 18, delay: 0.5 });
  M.animate(corners, { opacity: [0, 1], x: (i) => [i % 2 ? 22 : -22, 0], y: (i) => [i < 2 ? -22 : 22, 0] }, { duration: 0.8, delay: M.stagger(0.07, { startDelay: 0.85 }), ease });
  M.animate('#drop .inner > *', { opacity: [0, 1], y: [14, 0] }, { duration: 0.6, delay: M.stagger(0.09, { startDelay: 0.8 }), ease });
  M.animate('#controls .field, #controls .chk, #dlall', { opacity: [0, 1], y: [14, 0] }, { duration: 0.6, delay: M.stagger(0.07, { startDelay: 1.05 }), ease });
  M.inView('.notes', (el) => { M.animate(el, { opacity: [0, 1], y: [18, 0] }, { duration: 0.7, ease }); M.animate(el.querySelectorAll('.steps li'), { opacity: [0, 1], y: [16, 0] }, { duration: 0.6, delay: M.stagger(0.1, { startDelay: 0.15 }), ease }); });
  // content is shown by the animations above; this only releases the pre-hide class once they are running
  requestAnimationFrame(() => requestAnimationFrame(done));
}
intro();

const qs = new URLSearchParams(location.search);
if (qs.has('mode')) document.querySelector(`.seg[data-for=model] [data-v=${qs.get('mode')}]`)?.click();
if (qs.get('sample') === 'logo') $('sample2').click(); else if (qs.has('sample')) $('sample').click();
if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});
