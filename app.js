import { MODELS, toTensorData, toMask, sharpen, fitSize, applyAlpha, outName, hexToRgb } from './lib.mjs';

const $ = (id) => document.getElementById(id);
const ort = window.ort;
ort.env.wasm.wasmPaths = new URL('vendor/', location.href).href;
ort.env.wasm.numThreads = 1; // SHORTCUT: single thread avoids COOP/COEP headers; enable threads if hosting sets them

// Motion is optional: without it (or with reduced motion) everything still works, just without easing.
const M = window.Motion;
const calm = matchMedia('(prefers-reduced-motion: reduce)');
const spring = { type: 'spring', stiffness: 320, damping: 26 };
const anim = (el, kf, opt) => (M && !calm.matches ? M.animate(el, kf, opt) : null);

const sessions = {};
const items = [];
let queue = Promise.resolve();

async function getSession(key) {
  if (!sessions[key]) {
    sessions[key] = (async () => {
      const buf = await (await fetch(MODELS[key].file)).arrayBuffer();
      return ort.InferenceSession.create(buf, { executionProviders: ['wasm'] });
    })();
  }
  return sessions[key];
}

function canvasOf(w, h) { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; }

async function segment(bitmap, key, crisp) {
  const cfg = MODELS[key];
  const [w, h] = fitSize(bitmap.width, bitmap.height);
  const full = canvasOf(w, h);
  const fctx = full.getContext('2d', { willReadFrequently: true });
  fctx.drawImage(bitmap, 0, 0, w, h);
  const small = canvasOf(cfg.size, cfg.size);
  const sctx = small.getContext('2d', { willReadFrequently: true });
  sctx.drawImage(bitmap, 0, 0, cfg.size, cfg.size);
  const input = toTensorData(sctx.getImageData(0, 0, cfg.size, cfg.size).data, cfg.size, cfg);
  const session = await getSession(key);
  const res = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, cfg.size, cfg.size]) });
  const out = res[session.outputNames[0]].data;
  const m = toMask(out, key === 'general');
  const mc = canvasOf(cfg.size, cfg.size);
  const mimg = mc.getContext('2d').createImageData(cfg.size, cfg.size);
  for (let i = 0; i < m.length; i++) { mimg.data[i * 4] = mimg.data[i * 4 + 1] = mimg.data[i * 4 + 2] = m[i]; mimg.data[i * 4 + 3] = 255; }
  mc.getContext('2d').putImageData(mimg, 0, 0);
  const up = canvasOf(w, h);
  const uctx = up.getContext('2d', { willReadFrequently: true });
  uctx.imageSmoothingQuality = 'high';
  uctx.drawImage(mc, 0, 0, w, h);
  const upPx = uctx.getImageData(0, 0, w, h).data;
  let mask = new Uint8ClampedArray(w * h);
  for (let i = 0; i < mask.length; i++) mask[i] = upPx[i * 4];
  if (crisp) mask = sharpen(mask);
  const px = fctx.getImageData(0, 0, w, h);
  applyAlpha(px.data, mask);
  const cut = canvasOf(w, h);
  cut.getContext('2d').putImageData(px, 0, 0);
  return { cut, orig: full };
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

function makeCard(it) {
  const el = document.createElement('div');
  el.className = 'card';
  el.innerHTML = `<div class="cmp"><div class="ph busy">Working on it</div></div>
    <div class="row"><span></span><button class="btn" disabled>Download PNG</button></div>`;
  el.querySelector('.row span').textContent = it.name;
  it.el = el;
  el.querySelector('button').onclick = () => download(it);
  $('results').append(el);
  anim(el, { opacity: [0, 1], y: [18, 0], scale: [0.98, 1] }, spring);
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
  let pos = 50, sweep = null;
  const put = (p) => {
    pos = Math.min(100, Math.max(0, p));
    before.style.clipPath = `inset(0 ${100 - pos}% 0 0)`; handle.style.left = pos + '%';
    tl.style.opacity = pos < 14 ? 0 : 1; tr.style.opacity = pos > 86 ? 0 : 1;
    cmp.setAttribute('aria-valuenow', Math.round(pos));
  };
  const fromX = (x) => { const r = cmp.getBoundingClientRect(); return (x - r.left) / r.width * 100; };
  const glide = (to) => {
    if (sweep) sweep.stop();
    if (!M || calm.matches) return put(to);
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
  // first reveal: start on the original and sweep to the cut-out so the comparison explains itself
  if (M && !calm.matches) { put(100); setTimeout(() => glide(50), 250); } else put(50);
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
  queue = queue.then(async () => {
    const t0 = performance.now();
    try {
      const bmp = await createImageBitmap(file);
      const r = await segment(bmp, $('model').value, $('crisp').checked);
      Object.assign(it, r, { bmp, ms: Math.round(performance.now() - t0) });
      it.el.querySelector('.cmp').innerHTML = '';
      render(it); attachSlider(it);
      anim(it.el.querySelector('.cmp canvas.after'), { opacity: [0, 1] }, { duration: 0.4 });
      it.el.querySelector('button').disabled = false;
      it.el.querySelector('.row span').textContent = `${it.name} · ${(it.ms / 1000).toFixed(1)} s`;
    } catch (e) {
      it.failed = true; it.el.querySelector('.cmp').innerHTML = '<div class="ph">Could not process this file.<br><small></small></div>'; it.el.querySelector('small').textContent = String(e.message || e);
      console.error(e);
    }
    updateStatus();
  });
}

function addFiles(list) { for (const f of list) if (f.type.startsWith('image/')) addFile(f); }

const drop = $('drop');
const frame = drop.querySelector('.frame');
const lock = (on) => anim(frame, { scale: on ? 1.035 : 1 }, spring);
['dragenter', 'dragover'].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); if (!drop.classList.contains('over')) { drop.classList.add('over'); lock(true); } }));
['dragleave', 'drop'].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); if (drop.classList.contains('over')) { drop.classList.remove('over'); lock(false); } }));
drop.addEventListener('pointerenter', () => { if (!drop.classList.contains('over')) anim(frame, { scale: 1.015 }, spring); });
drop.addEventListener('pointerleave', () => { if (!drop.classList.contains('over')) anim(frame, { scale: 1 }, spring); });
drop.addEventListener('drop', (e) => addFiles(e.dataTransfer.files));
document.addEventListener('paste', (e) => addFiles(e.clipboardData?.files || []));
$('file').onchange = (e) => { addFiles(e.target.files); e.target.value = ''; };
$('sample').onclick = async () => { const b = await (await fetch('samples/dog.jpg')).blob(); addFiles([new File([b], 'sample.jpg', { type: b.type })]); };
$('dlall').onclick = () => items.filter((i) => i.out).forEach((i, n) => setTimeout(() => download(i), n * 400)); // SHORTCUT: sequential downloads, no zip; add a zip writer for big batches

function reveal(el, show) {
  if (!show) { el.hidden = true; return; }
  const was = el.hidden; el.hidden = false;
  if (was) anim(el, { opacity: [0, 1], x: [-8, 0] }, spring);
}

document.querySelectorAll('.seg').forEach((seg) => {
  const sel = $(seg.dataset.for);
  seg.addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    seg.querySelectorAll('button').forEach((x) => x.setAttribute('aria-checked', String(x === b)));
    sel.value = b.dataset.v; sel.dispatchEvent(new Event('input'));
  });
  seg.addEventListener('keydown', (e) => {
    const d = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key]; if (!d) return;
    const bs = [...seg.querySelectorAll('button')]; const i = bs.findIndex((x) => x.getAttribute('aria-checked') === 'true');
    const n = bs[(i + d + bs.length) % bs.length]; e.preventDefault(); n.click(); n.focus();
  });
});

function syncBg() {
  const m = $('bg').value;
  reveal($('colorLabel'), m === 'color'); reveal($('blurLabel'), m === 'blur');
  items.forEach(render);
}
['bg', 'color', 'blur'].forEach((id) => $(id).addEventListener('input', syncBg));

if (M && !calm.matches) {
  M.animate('.top, .hero h1, .lede, #drop', { opacity: [0, 1], y: [10, 0] }, { duration: 0.6, delay: M.stagger(0.07), ease: [0.22, 1, 0.36, 1] });
  M.inView('.notes', (el) => { M.animate(el, { opacity: [0, 1], y: [14, 0] }, { duration: 0.6, ease: [0.22, 1, 0.36, 1] }); });
}

if (new URLSearchParams(location.search).has('sample')) $('sample').click();
if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});
