/**
 * Server-side AI face transformation engine.
 *
 * Pipeline:
 *   SCRFD (det_10g.onnx)    — face detection + 5-point keypoints
 *   ArcFace (w600k_r50.onnx) — 512-dim identity embedding from reference photo
 *   InSwapper (inswapper_128.onnx) — neural face synthesis using identity latent
 *   sharp                   — JPEG encode/decode, resize (no native canvas needed)
 *
 * All ONNX inference via onnxruntime-node (prebuilt binaries, no C++ build).
 * All image I/O via sharp (prebuilt binaries, no C++ build).
 */

import sharp from 'sharp';
import { InferenceSession, Tensor } from 'onnxruntime-node';
import { existsSync, mkdirSync, statSync, createWriteStream } from 'fs';
import { readFile } from 'fs/promises';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import https from 'https';
import http from 'http';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const MODELS_DIR = join(__dirname, 'models');
if (!existsSync(MODELS_DIR)) mkdirSync(MODELS_DIR, { recursive: true });

const MODEL_URLS = {
  det:     'https://huggingface.co/Aitrepreneur/insightface/resolve/main/models/buffalo_l/det_10g.onnx',
  arcface: 'https://huggingface.co/Aitrepreneur/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx',
  swapper: 'https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx',
};

const MODEL_PATHS = {
  det:     join(MODELS_DIR, 'det_10g.onnx'),
  arcface: join(MODELS_DIR, 'w600k_r50.onnx'),
  swapper: join(MODELS_DIR, 'inswapper_128.onnx'),
};

// Minimum valid sizes (90% of real size as guard)
const MODEL_MIN_SIZES = {
  det:     15_000_000,
  arcface: 149_000_000,
  swapper: 498_000_000,
};

let sessions   = { det: null, arcface: null, swapper: null };
let swapperEmap = null;
let initState  = 'idle';
let initError  = null;
let downloadProgress = {};

export function getInitState() {
  return { state: initState, error: initError, progress: { ...downloadProgress } };
}

// ─── HTTP download: redirect-following, resume, retry, progress ──────────────

const STALL_TIMEOUT_MS  = 60_000;  // abort if no data received for 60s
const MAX_RETRIES       = 5;
const RETRY_DELAY_MS    = 3_000;

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

/**
 * Download url → dest with:
 *  - automatic redirect following
 *  - Range resume if partial file exists
 *  - stall detection (60 s without data → retry)
 *  - up to MAX_RETRIES retries with back-off
 *  - MB/total progress logging
 */
async function downloadFile(url, dest, label) {
  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    const alreadyBytes = existsSync(dest) ? statSync(dest).size : 0;
    const success = await attemptDownload(url, dest, label, alreadyBytes);
    if (success) return;
    if (attempt < MAX_RETRIES) {
      const wait = RETRY_DELAY_MS * attempt;
      console.warn(`[FaceSwap] ${label} download failed (attempt ${attempt}/${MAX_RETRIES}), retrying in ${wait/1000}s…`);
      await sleep(wait);
    }
  }
  throw new Error(`Failed to download ${label} after ${MAX_RETRIES} attempts`);
}

function attemptDownload(url, dest, label, resumeFrom) {
  return new Promise((resolve) => {
    const followRedirects = (u, rangeStart) => {
      const mod = u.startsWith('https') ? https : http;
      const headers = { 'User-Agent': 'Mozilla/5.0' };
      if (rangeStart > 0) headers['Range'] = `bytes=${rangeStart}-`;

      const req = mod.get(u, { headers }, (res) => {
        // Follow redirects (carry range header through)
        if ([301, 302, 307, 308].includes(res.statusCode)) {
          res.resume(); // drain
          return followRedirects(res.headers.location, rangeStart);
        }

        // 416 = range not satisfiable — server may not support resume, restart
        if (res.statusCode === 416) {
          res.resume();
          return followRedirects(u, 0);
        }

        const isPartial = res.statusCode === 206;
        if (res.statusCode !== 200 && !isPartial) {
          res.resume();
          console.warn(`[FaceSwap] ${label} HTTP ${res.statusCode}`);
          return resolve(false);
        }

        const contentLength = parseInt(res.headers['content-length'] || '0', 10);
        const total   = contentLength > 0 ? contentLength + (isPartial ? rangeStart : 0) : 0;
        let received  = isPartial ? rangeStart : 0;
        let lastData  = Date.now();
        let settled   = false;

        const finish = (ok) => { if (!settled) { settled = true; resolve(ok); } };

        // Stall watchdog — fires if no data for STALL_TIMEOUT_MS
        const watchdog = setInterval(() => {
          if (Date.now() - lastData > STALL_TIMEOUT_MS) {
            clearInterval(watchdog);
            req.destroy();
            console.warn(`[FaceSwap] ${label} stalled — no data for ${STALL_TIMEOUT_MS/1000}s`);
            finish(false);
          }
        }, 5_000);

        const ws = createWriteStream(dest, isPartial ? { flags: 'a' } : { flags: 'w' });

        res.on('data', (chunk) => {
          received += chunk.length;
          lastData = Date.now();
          if (total > 0) {
            const pct = Math.round((received / total) * 100);
            const mb  = (received / 1_000_000).toFixed(1);
            const tot = (total    / 1_000_000).toFixed(1);
            downloadProgress[label] = pct;
            // Log every 5%
            if (pct % 5 === 0 && downloadProgress[`${label}_lastLog`] !== pct) {
              downloadProgress[`${label}_lastLog`] = pct;
              console.log(`[FaceSwap] ${label}: ${mb} MB / ${tot} MB (${pct}%)`);
            }
          }
        });

        res.pipe(ws);

        ws.on('finish', () => {
          clearInterval(watchdog);
          downloadProgress[label] = 100;
          console.log(`[FaceSwap] ${label} download complete (${(received/1_000_000).toFixed(1)} MB)`);
          finish(true);
        });

        ws.on('error', (err) => {
          clearInterval(watchdog);
          console.warn(`[FaceSwap] ${label} write error:`, err.message);
          finish(false);
        });

        res.on('error', (err) => {
          clearInterval(watchdog);
          ws.destroy();
          console.warn(`[FaceSwap] ${label} response error:`, err.message);
          finish(false);
        });
      });

      req.on('error', (err) => {
        console.warn(`[FaceSwap] ${label} request error:`, err.message);
        resolve(false);
      });
      // No req.setTimeout — stall watchdog handles it per-data-event
    };

    followRedirects(url, resumeFrom);
  });
}

async function ensureModel(key) {
  const path = MODEL_PATHS[key];
  if (existsSync(path) && statSync(path).size >= MODEL_MIN_SIZES[key]) {
    downloadProgress[key] = 100;
    console.log(`[FaceSwap] ${key} already cached`);
    return;
  }
  // Partial file exists — will resume
  const partial = existsSync(path) ? statSync(path).size : 0;
  if (partial > 0) {
    console.log(`[FaceSwap] Resuming ${key} from ${(partial/1_000_000).toFixed(1)} MB…`);
  } else {
    console.log(`[FaceSwap] Downloading ${key}…`);
  }
  downloadProgress[key] = 0;
  await downloadFile(MODEL_URLS[key], path, key);
  console.log(`[FaceSwap] ${key} ready`);
}

// ─── Image helpers via sharp (no canvas, no native build required) ────────────

/**
 * Decode any image buffer (JPEG/PNG/etc.) to raw RGBA pixels.
 * Returns { data: Uint8Array, width, height }
 */
async function decodeToRGBA(buffer) {
  const { data, info } = await sharp(buffer)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return { data: new Uint8Array(data.buffer, data.byteOffset, data.byteLength), width: info.width, height: info.height };
}

/**
 * Resize RGBA pixel array to newW×newH using sharp.
 * Input and output are flat Uint8Array RGBA.
 */
async function resizeRGBA(pixels, srcW, srcH, newW, newH) {
  const buf = await sharp(Buffer.from(pixels), { raw: { width: srcW, height: srcH, channels: 4 } })
    .resize(newW, newH, { fit: 'fill', kernel: 'lanczos3' })
    .raw()
    .toBuffer();
  return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
}

/**
 * Encode RGBA pixel array to JPEG buffer.
 */
async function encodeToJPEG(pixels, width, height, quality = 85) {
  return sharp(Buffer.from(pixels), { raw: { width, height, channels: 4 } })
    .jpeg({ quality })
    .toBuffer();
}

// ─── Geometry helpers ─────────────────────────────────────────────────────────

// ArcFace canonical 5-point landmarks for 112×112 crop
const ARCFACE_DST = [
  38.2946, 51.6963,
  73.5318, 51.5014,
  56.0252, 71.7366,
  41.5493, 92.3655,
  70.7299, 92.2041,
];

/**
 * Estimate 2×3 similarity transform M mapping srcKps (flat [x0,y0,x1,y1,...]) → ARCFACE_DST.
 * Returns [a, b, tx, c, d, ty].
 */
function estimateNorm(srcKps, imageSize = 112) {
  const ratio = imageSize / 112.0;
  const dx    = imageSize % 128 === 0 ? 8.0 * ratio : 0;
  const dstX  = ARCFACE_DST.filter((_, i) => i % 2 === 0).map(v => v * ratio + dx);
  const dstY  = ARCFACE_DST.filter((_, i) => i % 2 !== 0).map(v => v * ratio);

  const srcX = srcKps.filter((_, i) => i % 2 === 0);
  const srcY = srcKps.filter((_, i) => i % 2 !== 0);
  const N = 5;

  const smx = srcX.reduce((a, v) => a + v, 0) / N;
  const smy = srcY.reduce((a, v) => a + v, 0) / N;
  const dmx = dstX.reduce((a, v) => a + v, 0) / N;
  const dmy = dstY.reduce((a, v) => a + v, 0) / N;

  let ss = 0, sxy = 0, syx = 0;
  for (let i = 0; i < N; i++) {
    const sx = srcX[i] - smx, sy = srcY[i] - smy;
    const dx2 = dstX[i] - dmx, dy2 = dstY[i] - dmy;
    ss  += sx * sx + sy * sy;
    sxy += sx * dx2 + sy * dy2;
    syx += sx * dy2 - sy * dx2;
  }

  const a  =  sxy / (ss + 1e-10);
  const b  = -syx / (ss + 1e-10);
  const tx =  dmx - a * smx - b * smy;
  const c  =  syx / (ss + 1e-10);
  const d  =  sxy / (ss + 1e-10);
  const ty =  dmy - c * smx - d * smy;

  return [a, b, tx, c, d, ty];
}

/** Invert 2×3 affine [a,b,tx,c,d,ty]. */
function invertAffine([a, b, tx, c, d, ty]) {
  const det = a * d - b * c;
  if (Math.abs(det) < 1e-10) return [1, 0, 0, 0, 1, 0];
  const ia = d / det, ib = -b / det;
  const ic = -c / det, id = a / det;
  return [ia, ib, (b * ty - d * tx) / det, ic, id, (c * tx - a * ty) / det];
}

/**
 * Warp RGBA pixel array by 2×3 affine (inverse mapping).
 * Output pixel at (x,y) is sampled from source at M_inv * [x,y,1].
 */
function warpAffine(pixels, srcW, srcH, M, outSize) {
  const [a, b, tx, c, d, ty] = M;
  const det = a * d - b * c;
  const out = new Uint8ClampedArray(outSize * outSize * 4);

  // Inverse of M
  let ia, ib, itx, ic, id2, ity;
  if (Math.abs(det) < 1e-10) {
    ia = 1; ib = 0; itx = 0; ic = 0; id2 = 1; ity = 0;
  } else {
    ia = d / det; ib = -b / det; itx = (b * ty - d * tx) / det;
    ic = -c / det; id2 = a / det; ity = (c * tx - a * ty) / det;
  }

  for (let y = 0; y < outSize; y++) {
    for (let x = 0; x < outSize; x++) {
      const sx = Math.round(ia * x + ib * y + itx);
      const sy = Math.round(ic * x + id2 * y + ity);
      const di = (y * outSize + x) * 4;
      if (sx >= 0 && sx < srcW && sy >= 0 && sy < srcH) {
        const si = (sy * srcW + sx) * 4;
        out[di]     = pixels[si];
        out[di + 1] = pixels[si + 1];
        out[di + 2] = pixels[si + 2];
        out[di + 3] = 255;
      }
    }
  }
  return out;
}

/**
 * Warp outW×outH target image using forward M (paint fake face back).
 * Equivalent to cv2.warpAffine with M_inv.
 */
function warpAffineBack(pixels, srcW, srcH, M_inv, outW, outH) {
  const [a, b, tx, c, d, ty] = M_inv;
  const out = new Uint8ClampedArray(outW * outH * 4);
  for (let y = 0; y < srcH; y++) {
    for (let x = 0; x < srcW; x++) {
      const dx = Math.round(a * x + b * y + tx);
      const dy = Math.round(c * x + d * y + ty);
      if (dx >= 0 && dx < outW && dy >= 0 && dy < outH) {
        const si = (y * srcW + x) * 4;
        const di = (dy * outW + dx) * 4;
        out[di]     = pixels[si];
        out[di + 1] = pixels[si + 1];
        out[di + 2] = pixels[si + 2];
        out[di + 3] = 255;
      }
    }
  }
  return out;
}

// ─── ONNX helpers ─────────────────────────────────────────────────────────────

/**
 * Convert RGBA pixel array to CHW float32 blob.
 * norm='arcface': (v-127.5)/128  norm='swapper': v/255
 */
function pixelsToBlob(pixels, w, h, norm) {
  const n = w * h;
  const blob = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    const r = pixels[i * 4], g = pixels[i * 4 + 1], b = pixels[i * 4 + 2];
    if (norm === 'arcface') {
      blob[i]         = (r - 127.5) / 128;
      blob[n + i]     = (g - 127.5) / 128;
      blob[2 * n + i] = (b - 127.5) / 128;
    } else {
      blob[i]         = r / 255;
      blob[n + i]     = g / 255;
      blob[2 * n + i] = b / 255;
    }
  }
  return blob;
}

/** Convert CHW float32 prediction [0,1] → RGBA Uint8ClampedArray. */
function predToPixels(pred, size) {
  const n = size * size;
  const out = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    out[i * 4]     = Math.max(0, Math.min(255, Math.round(pred[i]         * 255)));
    out[i * 4 + 1] = Math.max(0, Math.min(255, Math.round(pred[n + i]     * 255)));
    out[i * 4 + 2] = Math.max(0, Math.min(255, Math.round(pred[2 * n + i] * 255)));
    out[i * 4 + 3] = 255;
  }
  return out;
}

// ─── SCRFD face detection ─────────────────────────────────────────────────────

function iou(a, b) {
  const ix1 = Math.max(a.x1, b.x1), iy1 = Math.max(a.y1, b.y1);
  const ix2 = Math.min(a.x2, b.x2), iy2 = Math.min(a.y2, b.y2);
  if (ix2 <= ix1 || iy2 <= iy1) return 0;
  const inter = (ix2 - ix1) * (iy2 - iy1);
  return inter / ((a.x2-a.x1)*(a.y2-a.y1) + (b.x2-b.x1)*(b.y2-b.y1) - inter);
}

function decodeSCRFD(outputs, inputW, inputH, threshold = 0.45) {
  const strides = [8, 16, 32];
  const numAnchors = 2;
  const faces = [];

  let outIdx = 0;
  for (let si = 0; si < strides.length; si++) {
    const stride = strides[si];
    const fh = Math.ceil(inputH / stride);
    const fw = Math.ceil(inputW / stride);
    const n = fh * fw * numAnchors;

    const scores  = outputs[outIdx].data;
    const bboxD   = outputs[outIdx + 1].data;
    const kpsD    = outputs[outIdx + 2].data;
    outIdx += 3;

    for (let i = 0; i < n; i++) {
      if (scores[i] < threshold) continue;
      const ay = Math.floor(i / (fw * numAnchors));
      const ax = Math.floor((i % (fw * numAnchors)) / numAnchors);
      const cx = (ax + 0.5) * stride, cy = (ay + 0.5) * stride;

      faces.push({
        score: scores[i],
        x1: cx - bboxD[i*4]   * stride, y1: cy - bboxD[i*4+1] * stride,
        x2: cx + bboxD[i*4+2] * stride, y2: cy + bboxD[i*4+3] * stride,
        kps: Array.from({ length: 10 }, (_, k) => {
          const isX = k % 2 === 0;
          return (isX ? cx : cy) + kpsD[i*10 + k] * stride;
        }),
      });
    }
  }

  // NMS
  faces.sort((a, b) => b.score - a.score);
  const keep = [], used = new Set();
  for (let i = 0; i < faces.length; i++) {
    if (used.has(i)) continue;
    keep.push(faces[i]);
    for (let j = i + 1; j < faces.length; j++) {
      if (!used.has(j) && iou(faces[i], faces[j]) > 0.4) used.add(j);
    }
  }
  return keep;
}

async function detectFaces(pixels, W, H) {
  const inputSize = 640;
  const scale = Math.min(inputSize / W, inputSize / H);
  const newW = Math.round(W * scale), newH = Math.round(H * scale);
  const padX = Math.floor((inputSize - newW) / 2);
  const padY = Math.floor((inputSize - newH) / 2);

  // Resize + letterbox using sharp
  const resizedBuf = await sharp(Buffer.from(pixels), { raw: { width: W, height: H, channels: 4 } })
    .resize(newW, newH, { fit: 'fill', kernel: 'lanczos3' })
    .raw()
    .toBuffer();

  // Create 640×640 gray canvas, paste resized image
  const padded = new Uint8Array(inputSize * inputSize * 4);
  padded.fill(128); // gray letterbox
  for (let y = 0; y < newH; y++) {
    for (let x = 0; x < newW; x++) {
      const si = (y * newW + x) * 4;
      const di = ((y + padY) * inputSize + (x + padX)) * 4;
      padded[di]     = resizedBuf[si];
      padded[di + 1] = resizedBuf[si + 1];
      padded[di + 2] = resizedBuf[si + 2];
      padded[di + 3] = 255;
    }
  }

  // BGR CHW, mean 127.5 std 128
  const blob = new Float32Array(3 * inputSize * inputSize);
  const n = inputSize * inputSize;
  for (let i = 0; i < n; i++) {
    blob[i]         = (padded[i*4]     - 127.5) / 128; // R
    blob[n + i]     = (padded[i*4 + 1] - 127.5) / 128; // G
    blob[2*n + i]   = (padded[i*4 + 2] - 127.5) / 128; // B
  }

  const result = await sessions.det.run({
    input: new Tensor('float32', blob, [1, 3, inputSize, inputSize]),
  });
  const faces = decodeSCRFD(Object.values(result), inputSize, inputSize);

  // Rescale to original coords
  return faces.map(f => ({
    ...f,
    x1:  (f.x1 - padX) / scale, y1:  (f.y1 - padY) / scale,
    x2:  (f.x2 - padX) / scale, y2:  (f.y2 - padY) / scale,
    kps: f.kps.map((v, i) => i % 2 === 0 ? (v - padX) / scale : (v - padY) / scale),
  }));
}

function getBestFace(faces) {
  if (!faces.length) return null;
  return faces.reduce((b, f) =>
    (f.x2-f.x1)*(f.y2-f.y1) > (b.x2-b.x1)*(b.y2-b.y1) ? f : b
  );
}

// ─── ArcFace embedding ────────────────────────────────────────────────────────

async function getEmbedding(pixels, W, H, kps) {
  const sz = 112;
  const M  = estimateNorm(kps, sz);
  const aligned = warpAffine(pixels, W, H, M, sz);
  const blob = pixelsToBlob(aligned, sz, sz, 'arcface');

  const inName  = sessions.arcface.inputNames[0];
  const outName = sessions.arcface.outputNames[0];
  const result  = await sessions.arcface.run({
    [inName]: new Tensor('float32', blob, [1, 3, sz, sz]),
  });
  const emb = result[outName].data;

  let norm = 0;
  for (const v of emb) norm += v * v;
  norm = Math.sqrt(norm);
  const normed = new Float32Array(emb.length);
  for (let i = 0; i < emb.length; i++) normed[i] = emb[i] / norm;
  return normed;
}

// ─── InSwapper emap extraction ────────────────────────────────────────────────

async function loadSwapperEmap() {
  try {
    const buf = await readFile(MODEL_PATHS.swapper);
    const { Root } = await import('protobufjs');

    const root = Root.fromJSON({
      nested: {
        TensorProto: {
          fields: {
            dims:       { id: 1, rule: 'repeated', type: 'int64' },
            data_type:  { id: 2, rule: 'optional', type: 'int32' },
            float_data: { id: 4, rule: 'repeated', type: 'float', options: { packed: true } },
            name:       { id: 8, rule: 'optional', type: 'string' },
            raw_data:   { id: 9, rule: 'optional', type: 'bytes' },
          },
        },
        GraphProto: {
          fields: {
            node:        { id: 1, rule: 'repeated', type: 'NodeProto' },
            initializer: { id: 6, rule: 'repeated', type: 'TensorProto' },
          },
        },
        NodeProto: { fields: { output: { id: 2, rule: 'repeated', type: 'string' } } },
        ModelProto: { fields: { graph: { id: 7, rule: 'optional', type: 'GraphProto' } } },
      },
    });

    const model = root.lookupType('ModelProto').decode(buf);
    const inits = model.graph?.initializer ?? [];
    for (let i = inits.length - 1; i >= 0; i--) {
      const t = inits[i];
      const dims = Array.from(t.dims ?? []).map(Number);
      if (dims.length === 2 && dims[0] === 512 && dims[1] === 512) {
        if (t.raw_data?.length >= 512 * 512 * 4) {
          const arr = new Float32Array(t.raw_data.buffer, t.raw_data.byteOffset, 512 * 512);
          console.log('[FaceSwap] emap extracted from model initializer', i);
          return Float32Array.from(arr);
        }
        if (t.float_data?.length === 512 * 512) {
          console.log('[FaceSwap] emap (float_data) extracted from initializer', i);
          return Float32Array.from(t.float_data);
        }
      }
    }
    console.warn('[FaceSwap] emap not found — identity pass-through');
    return null;
  } catch (e) {
    console.warn('[FaceSwap] emap load error:', e.message);
    return null;
  }
}

// ─── InSwapper inference ──────────────────────────────────────────────────────

async function runSwapper(pixels, W, H, kps, srcEmbedding) {
  const sz = 128;
  const M  = estimateNorm(kps, sz);
  const aligned = warpAffine(pixels, W, H, M, sz);
  const blob = pixelsToBlob(aligned, sz, sz, 'swapper');

  // Compute latent: normed_emb @ emap, re-normalize
  const dim = 512;
  let latent;
  if (swapperEmap) {
    latent = new Float32Array(dim);
    for (let j = 0; j < dim; j++) {
      let s = 0;
      for (let k = 0; k < dim; k++) s += srcEmbedding[k] * swapperEmap[k * dim + j];
      latent[j] = s;
    }
    let norm = 0;
    for (const v of latent) norm += v * v;
    norm = Math.sqrt(norm);
    for (let i = 0; i < dim; i++) latent[i] /= norm;
  } else {
    latent = srcEmbedding;
  }

  const [inN0, inN1] = sessions.swapper.inputNames;
  const outN = sessions.swapper.outputNames[0];
  const result = await sessions.swapper.run({
    [inN0]: new Tensor('float32', blob,   [1, 3, sz, sz]),
    [inN1]: new Tensor('float32', latent, [1, dim]),
  });
  return { pred: result[outN].data, M, aligned };
}

// ─── Paste-back with blurred mask ─────────────────────────────────────────────

function gaussianBlur1ch(src, w, h, r) {
  const k = r * 2 + 1, sigma = r / 3;
  const kern = new Float32Array(k);
  let s = 0;
  for (let i = 0; i < k; i++) { const x = i - r; kern[i] = Math.exp(-(x*x)/(2*sigma*sigma)); s += kern[i]; }
  for (let i = 0; i < k; i++) kern[i] /= s;

  const tmp = new Float32Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let v = 0;
      for (let ki = 0; ki < k; ki++) { const xi = Math.min(w-1, Math.max(0, x+ki-r)); v += src[y*w+xi]*kern[ki]; }
      tmp[y*w+x] = v;
    }
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let v = 0;
      for (let ki = 0; ki < k; ki++) { const yi = Math.min(h-1, Math.max(0, y+ki-r)); v += tmp[yi*w+x]*kern[ki]; }
      out[y*w+x] = v;
    }
  return out;
}

function pasteBack(targetPixels, W, H, pred, M, swapSize) {
  const fakePx  = predToPixels(pred, swapSize);
  const M_inv   = invertAffine(M);
  const warped  = warpAffineBack(fakePx, swapSize, swapSize, M_inv, W, H);

  // Build raw mask from warped alpha
  const rawMask = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) rawMask[i] = warped[i*4+3] > 0 ? 255 : 0;

  // Erode 5px
  const erR = 5;
  const eroded = new Float32Array(W * H);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      let mn = 255;
      for (let dy = -erR; dy <= erR; dy++)
        for (let dx = -erR; dx <= erR; dx++) {
          const v = rawMask[Math.min(H-1,Math.max(0,y+dy))*W + Math.min(W-1,Math.max(0,x+dx))];
          if (v < mn) mn = v;
        }
      eroded[y*W+x] = mn;
    }

  const blurR = Math.max(8, Math.floor(Math.sqrt(W*H/(640*480))*16));
  const mask  = gaussianBlur1ch(eroded, W, H, blurR);

  const out = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    const a = mask[i] / 255;
    out[i*4]   = Math.round(a * warped[i*4]   + (1-a) * targetPixels[i*4]);
    out[i*4+1] = Math.round(a * warped[i*4+1] + (1-a) * targetPixels[i*4+1]);
    out[i*4+2] = Math.round(a * warped[i*4+2] + (1-a) * targetPixels[i*4+2]);
    out[i*4+3] = 255;
  }
  return out;
}

// ─── Public API ───────────────────────────────────────────────────────────────

let referenceEmbedding = null;

export async function initialize(onProgress) {
  if (initState === 'ready' || initState === 'loading' || initState === 'downloading') return;
  try {
    initState = 'downloading';
    onProgress?.({ state: 'downloading', message: 'Downloading det_10g.onnx (16 MB)...' });
    await ensureModel('det');

    onProgress?.({ state: 'downloading', message: 'Downloading w600k_r50.onnx (166 MB)...' });
    await ensureModel('arcface');

    onProgress?.({ state: 'downloading', message: 'Downloading inswapper_128.onnx (554 MB)...' });
    await ensureModel('swapper');

    initState = 'loading';
    onProgress?.({ state: 'loading', message: 'Loading ONNX sessions...' });

    const opts = { executionProviders: ['cpu'] };
    sessions.det     = await InferenceSession.create(MODEL_PATHS.det,     opts);
    sessions.arcface = await InferenceSession.create(MODEL_PATHS.arcface,  opts);
    sessions.swapper = await InferenceSession.create(MODEL_PATHS.swapper,  opts);

    console.log('[FaceSwap] Swapper inputs:', sessions.swapper.inputNames);
    swapperEmap = await loadSwapperEmap();

    initState = 'ready';
    onProgress?.({ state: 'ready', message: 'AI transformation engine ready' });
    console.log('[FaceSwap] All models loaded');
  } catch (err) {
    initState = 'error';
    initError = err.message;
    onProgress?.({ state: 'error', message: err.message });
    console.error('[FaceSwap] Init error:', err);
    throw err;
  }
}

export async function registerReferenceFace(imageBuffer) {
  if (initState !== 'ready') throw new Error(`Engine not ready (${initState})`);

  const { data, width, height } = await decodeToRGBA(imageBuffer);
  const faces = await detectFaces(data, width, height);
  const best  = getBestFace(faces);
  if (!best) throw new Error('No face detected in reference image');

  referenceEmbedding = await getEmbedding(data, width, height, best.kps);
  console.log('[FaceSwap] Reference embedded, dim:', referenceEmbedding.length);
  return { success: true, bbox: { x1: best.x1, y1: best.y1, x2: best.x2, y2: best.y2 } };
}

export async function transformFrame(jpegBuffer) {
  if (initState !== 'ready' || !referenceEmbedding) return null;

  const { data, width, height } = await decodeToRGBA(jpegBuffer);
  const faces = await detectFaces(data, width, height);
  const best  = getBestFace(faces);
  if (!best) return null;

  const { pred, M } = await runSwapper(data, width, height, best.kps, referenceEmbedding);
  const resultPixels = pasteBack(data, width, height, pred, M, 128);

  return encodeToJPEG(resultPixels, width, height, 85);
}

export function clearReference() {
  referenceEmbedding = null;
}
