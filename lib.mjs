// Rena hjälpfunktioner (inga DOM-beroenden) så de kan testas med node --test.

export const MODELS = {
  general: { file: 'models/u2netp.onnx', size: 320, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], scaleMax: true },
  portrait: { file: 'models/modnet.onnx', size: 512, mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5], scaleMax: false },
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

// Modellutdata -> 0..255 mask. minmax=true för U2-Net (utdata är ej 0..1-kalibrerad).
export function toMask(data, minmax) {
  const m = new Uint8ClampedArray(data.length);
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
