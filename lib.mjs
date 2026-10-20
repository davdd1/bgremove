// Rena hjälpfunktioner (inga DOM-beroenden) så de kan testas med node --test.

// Alla modeller kör med samma förbehandling; skillnader ligger i konfigurationen.
// local: ligger i repot (laddas från sajten). remote: laddas från huggingface.co vid första användning och cachas i webbläsaren.
const HF = 'https://huggingface.co/imgly/isnet-general-onnx/resolve/main/onnx/';
export const MODELS = {
  // ISNet general use (DIS, Apache-2.0; imgly ONNX conversion, MIT). WebGPU runs the fp32 file, WASM the half-size fp16 file.
  quality: { remote: { gpu: HF + 'model.onnx', gpuBytes: 176149806, cpu: HF + 'model_fp16.onnx', cpuBytes: 88152708 }, label: 'ISNet', size: 1024, mean: [0.5, 0.5, 0.5], std: [1, 1, 1], scaleMax: false, sigmoid: false, minmax: true },
  portrait: { file: 'models/modnet.onnx', label: 'MODNet', size: 512, mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5], scaleMax: false, sigmoid: false, minmax: false },
  fast: { file: 'models/u2netp.onnx', label: 'U2-Netp', size: 320, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], scaleMax: true, sigmoid: false, minmax: true },
};

// RGBA (Uint8ClampedArray, size*size*4) -> Float32 CHW, normaliserad enligt modellen.
export function toTensorData(rgba, size, cfg) {
  const n = size * size;
  const out = new Float32Array(3 * n);
  let max = 1;
  if (cfg.scaleMax) {
    max = 0;
    for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) if (rgba[i * 4 + c] > max) max = rgba[i * 4 + c];
    if (max === 0) max = 1;
    max /= 255; // x/max_255 -> värden 0..1 med maxvärde 1
  }
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) {
      const v = rgba[i * 4 + c] / 255 / max;
      out[c * n + i] = (v - cfg.mean[c]) / cfg.std[c];
    }
  }
  return out;
}

// Modellutdata -> 0..255 mask. minmax=true för U2-Net (utdata är ej 0..1-kalibrerad), sigmoid=true för BiRefNet (logits).
export function toMask(data, minmax, sigmoid = false) {
  const m = new Uint8ClampedArray(data.length);
  if (sigmoid) { for (let i = 0; i < data.length; i++) m[i] = Math.round(255 / (1 + Math.exp(-data[i]))); return m; }
  let lo = 0, hi = 1;
  if (minmax) {
    lo = Infinity; hi = -Infinity;
    for (const v of data) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (hi - lo < 1e-9) hi = lo + 1;
  }
  for (let i = 0; i < data.length; i++) m[i] = Math.round(255 * Math.min(1, Math.max(0, (data[i] - lo) / (hi - lo))));
  return m;
}

// Kontrastjustering av mask för skarpare kanter (smoothstep mellan lo och hi, 0..255).
export function sharpen(mask, lo = 40, hi = 215) {
  const out = new Uint8ClampedArray(mask.length);
  for (let i = 0; i < mask.length; i++) {
    const t = Math.min(1, Math.max(0, (mask[i] - lo) / (hi - lo)));
    out[i] = Math.round(255 * t * t * (3 - 2 * t));
  }
  return out;
}

// Maxstorlek för utdata så att mobiler inte får slut på minne.
export function fitSize(w, h, maxSide = 4096) {
  const s = Math.min(1, maxSide / Math.max(w, h));
  return [Math.max(1, Math.round(w * s)), Math.max(1, Math.round(h * s))];
}

// Sätt alfakanalen i rgba (full storlek) från mask (samma storlek, 1 byte/px).
export function applyAlpha(rgba, mask) {
  for (let i = 0; i < mask.length; i++) rgba[i * 4 + 3] = mask[i];
  return rgba;
}

export function outName(name, suffix = 'utan-bakgrund') {
  const base = name.replace(/\.[^.]+$/, '') || 'bild';
  return `${base}-${suffix}.png`;
}

export function hexToRgb(hex) {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex);
  if (!m) return [255, 255, 255];
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

// ---- Platt bakgrund (loggor, skärmdumpar): färgnyckel mot en anpassad bakgrundsyta ----

// Passar ett plan c = a + b*x + c*y (x,y i 0..1) per kanal mot kantpixlarna. Fångar enfärgade och linjära toningar.
export function fitBackground(rgba, w, h) {
  const r = Math.max(2, Math.round(Math.min(w, h) * 0.01));
  const pts = [];
  const step = Math.max(1, Math.floor((2 * (w + h) * r) / 6000));
  let k = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (x >= r && x < w - r && y >= r && y < h - r) { x = w - r - 1; continue; }
    if (k++ % step === 0) pts.push(x, y);
  }
  const n = pts.length / 2;
  // normalekvationer 3x3 för A = [1, x/w, y/h]
  const M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const B = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]; // B[kanal][term]
  for (let i = 0; i < n; i++) {
    const x = pts[2 * i], y = pts[2 * i + 1], v = [1, x / w, y / h], o = (y * w + x) * 4;
    for (let a = 0; a < 3; a++) {
      for (let b = 0; b < 3; b++) M[a][b] += v[a] * v[b];
      for (let c = 0; c < 3; c++) B[c][a] += v[a] * rgba[o + c];
    }
  }
  const coef = [0, 1, 2].map((c) => solve3(M, B[c]));
  const resid = new Float32Array(n);
  let noise = 0, nn = 0, prev = null;
  for (let i = 0; i < n; i++) {
    const x = pts[2 * i], y = pts[2 * i + 1], o = (y * w + x) * 4;
    let m = 0;
    for (let c = 0; c < 3; c++) { const p = coef[c][0] + coef[c][1] * x / w + coef[c][2] * y / h; m = Math.max(m, Math.abs(rgba[o + c] - p)); }
    resid[i] = m;
    if (prev !== null && Math.abs(y - prev[1]) <= 1 && Math.abs(x - prev[0]) <= step + 1) {
      let d = 0; for (let c = 0; c < 3; c++) d = Math.max(d, Math.abs(rgba[o + c] - rgba[prev[2] + c]));
      noise += d; nn++;
    }
    prev = [x, y, o];
  }
  const sorted = Float32Array.from(resid).sort();
  return { coef, p95: sorted[Math.floor(0.95 * (n - 1))], noise: nn ? noise / nn : 0 };
}

function solve3(M, b) {
  const A = M.map((row, i) => [...row, b[i]]);
  for (let i = 0; i < 3; i++) {
    let p = i; for (let r = i + 1; r < 3; r++) if (Math.abs(A[r][i]) > Math.abs(A[p][i])) p = r;
    [A[i], A[p]] = [A[p], A[i]];
    if (Math.abs(A[i][i]) < 1e-12) return [b.length ? b[0] / (M[0][0] || 1) : 0, 0, 0];
    for (let r = i + 1; r < 3; r++) { const f = A[r][i] / A[i][i]; for (let c = i; c < 4; c++) A[r][c] -= f * A[i][c]; }
  }
  const x = [0, 0, 0];
  for (let i = 2; i >= 0; i--) { let s = A[i][3]; for (let c = i + 1; c < 3; c++) s -= A[i][c] * x[c]; x[i] = s / A[i][i]; }
  return x;
}

// Platt bakgrund = kanten är jämn (liten rest mot planet) och brusfri. Foton (brus, motiv i kanten) faller igenom till AI-modellen.
export function isFlat(fit, maxResid = 10, maxNoise = 2) {
  return fit.p95 <= maxResid && fit.noise <= maxNoise;
}

const smooth = (t) => (t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t));

// Nyckel ut bakgrunden: alfa växer mjukt med avståndet från bakgrundsytan (lo..hi) och färgen
// "avblandas" (fg = bg + (p-bg)/alfa) så att kantpixlar inte får en mörk eller ljus kant mot nya bakgrunder.
// Ändrar rgba på plats och returnerar masken (0..255).
export function colorKey(rgba, w, h, fit, lo = 6, hi = 28) {
  const mask = new Uint8ClampedArray(w * h);
  const [cr, cg, cb] = fit.coef;
  for (let y = 0; y < h; y++) {
    const fy = y / h;
    for (let x = 0; x < w; x++) {
      const fx = x / w, i = y * w + x, o = i * 4;
      const br = cr[0] + cr[1] * fx + cr[2] * fy, bgc = cg[0] + cg[1] * fx + cg[2] * fy, bb = cb[0] + cb[1] * fx + cb[2] * fy;
      const dr = rgba[o] - br, dg = rgba[o + 1] - bgc, db = rgba[o + 2] - bb;
      const d = Math.max(Math.abs(dr), Math.abs(dg), Math.abs(db));
      const a = smooth((d - lo) / (hi - lo));
      mask[i] = Math.round(255 * a);
      if (a > 0 && a < 1) {
        rgba[o] = Math.min(255, Math.max(0, br + dr / a));
        rgba[o + 1] = Math.min(255, Math.max(0, bgc + dg / a));
        rgba[o + 2] = Math.min(255, Math.max(0, bb + db / a));
      }
      rgba[o + 3] = mask[i];
    }
  }
  return mask;
}

// Andel förgrund; används för att avvisa "platt" när nästan allt eller inget nycklades bort.
export function coverage(mask) {
  let s = 0; for (let i = 0; i < mask.length; i++) s += mask[i] > 127 ? 1 : 0;
  return s / mask.length;
}

// ---- Maskförfining: kantsnap mot bilden, halo-städning, tunna delar och färgrening ----

// Boxsuddighet (medelvärde i (2r+1)^2-fönster, kanter klampas) i O(n) med glidande summa.
export function boxBlur(src, w, h, r) {
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h), k = 2 * r + 1;
  for (let y = 0; y < h; y++) {
    const o = y * w; let s = 0;
    for (let x = -r; x <= r; x++) s += src[o + Math.min(w - 1, Math.max(0, x))];
    for (let x = 0; x < w; x++) {
      tmp[o + x] = s / k;
      s += src[o + Math.min(w - 1, x + r + 1)] - src[o + Math.max(0, x - r)];
    }
  }
  for (let x = 0; x < w; x++) {
    let s = 0;
    for (let y = -r; y <= r; y++) s += tmp[Math.min(h - 1, Math.max(0, y)) * w + x];
    for (let y = 0; y < h; y++) {
      out[y * w + x] = s / k;
      s += tmp[Math.min(h - 1, y + r + 1) * w + x] - tmp[Math.max(0, y - r) * w + x];
    }
  }
  return out;
}

// Guided filter (He et al.) med gråskaleguide: gör att maskens kant följer kanten i bilden i stället för modellens mjuka gissning.
export function guidedFilter(guide, src, w, h, r, eps) {
  const n = w * h;
  const mI = boxBlur(guide, w, h, r), mP = boxBlur(src, w, h, r);
  const ip = new Float32Array(n), ii = new Float32Array(n);
  for (let i = 0; i < n; i++) { ip[i] = guide[i] * src[i]; ii[i] = guide[i] * guide[i]; }
  const mIP = boxBlur(ip, w, h, r), mII = boxBlur(ii, w, h, r);
  const a = new Float32Array(n), b = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const varI = mII[i] - mI[i] * mI[i], cov = mIP[i] - mI[i] * mP[i];
    a[i] = cov / (varI + eps); b[i] = mP[i] - a[i] * mI[i];
  }
  const mA = boxBlur(a, w, h, r), mB = boxBlur(b, w, h, r), out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = mA[i] * guide[i] + mB[i];
  return out;
}

// Tar bort små lösa öar i masken (alfa > 127) som är mindre än minFrac av den största delen. Ändrar masken på plats.
export function dropIslands(mask, w, h, minFrac = 0.02) {
  const lab = new Int32Array(w * h), sizes = [0], stack = [];
  let n = 0;
  for (let s = 0; s < mask.length; s++) {
    if (mask[s] <= 127 || lab[s]) continue;
    n++; let cnt = 0; stack.push(s); lab[s] = n;
    while (stack.length) {
      const p = stack.pop(); cnt++;
      const x = p % w, y = (p - x) / w;
      if (x > 0 && mask[p - 1] > 127 && !lab[p - 1]) { lab[p - 1] = n; stack.push(p - 1); }
      if (x < w - 1 && mask[p + 1] > 127 && !lab[p + 1]) { lab[p + 1] = n; stack.push(p + 1); }
      if (y > 0 && mask[p - w] > 127 && !lab[p - w]) { lab[p - w] = n; stack.push(p - w); }
      if (y < h - 1 && mask[p + w] > 127 && !lab[p + w]) { lab[p + w] = n; stack.push(p + w); }
    }
    sizes.push(cnt);
  }
  const big = Math.max(0, ...sizes);
  if (!big) return mask;
  const keep = sizes.map((c) => c >= big * minFrac);
  // en kvarvarande del behåller sin mjuka kant: allt inom avstånd 3 px från en kvarvarande del får vara kvar
  const grow = new Uint8Array(w * h);
  for (let i = 0; i < mask.length; i++) if (lab[i] && keep[lab[i]]) grow[i] = 1;
  const wide = boxBlur(Float32Array.from(grow), w, h, 3);
  for (let i = 0; i < mask.length; i++) if (wide[i] === 0 && mask[i] > 0) mask[i] = 0;
  return mask;
}

// Förfina en 0..255-mask mot bilden (rgba, w*h). Steg: guided filter -> kontrastkurva som nollar svag "dis"
// och lyfter halvsvaga tunna delar -> släng lösa öar. opts.lo/hi styr kurvan (0..1).
export function refineMask(mask, rgba, w, h, opts = {}) {
  const { lo = 0.13, hi = 0.6, radius = Math.max(3, Math.round(Math.min(w, h) * 0.004)), eps = 2e-3 } = opts;
  const n = w * h, guide = new Float32Array(n), a = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    guide[i] = (0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2]) / 255;
    a[i] = mask[i] / 255;
  }
  const g = guidedFilter(guide, a, w, h, radius, eps);
  const out = new Uint8ClampedArray(n);
  for (let i = 0; i < n; i++) {
    // filtret får lägga till och skärpa kanter men inte äta in i det modellen är säker på
    const v = Math.max(g[i], a[i] * 0.95);
    out[i] = Math.round(255 * smooth((v - lo) / (hi - lo)));
  }
  return dropIslands(out, w, h);
}

// Färgrening: pixlar med alfa < 1 får färg från närmaste säkra förgrund (normaliserad suddighet),
// så att kanten inte bär med sig bakgrundens grå/färg. Ändrar rgba på plats.
export function decontaminate(rgba, mask, w, h, r = Math.max(3, Math.round(Math.min(w, h) * 0.006))) {
  const n = w * h, cr = new Float32Array(n), cg = new Float32Array(n), cb = new Float32Array(n), cw = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const c = mask[i] > 242 ? 1 : 0;
    cw[i] = c; cr[i] = c * rgba[i * 4]; cg[i] = c * rgba[i * 4 + 1]; cb[i] = c * rgba[i * 4 + 2];
  }
  const R = boxBlur(cr, w, h, r), G = boxBlur(cg, w, h, r), B = boxBlur(cb, w, h, r), W = boxBlur(cw, w, h, r);
  for (let i = 0; i < n; i++) {
    if (mask[i] > 242 || mask[i] === 0 || W[i] < 0.02) continue;
    rgba[i * 4] = R[i] / W[i]; rgba[i * 4 + 1] = G[i] / W[i]; rgba[i * 4 + 2] = B[i] / W[i];
  }
  return rgba;
}
