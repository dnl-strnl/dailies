'use strict';
// Dailies: scans a folder of stills and clips in the browser, hashes each still and each
// clip's first/last frame, and surfaces boundary matches and same-start variant groups.
// Every clip is split into shots (runs of frames between cuts), each shown as one frame in the shots view.
// The harvest view scrubs every clip as a filmstrip and cuts frames into key-frame collections on disk.

const IMG = /\.(png|jpe?g|webp)$/i, VID = /\.(mp4|webm|mov|m4v)$/i;
const THUMB_W = 320;
const HN = 32, HK = 16, BITS = HK * HK - 1;   // pHash: 32x32 luma -> low 16x16 DCT block, DC dropped
const SIG = 8 * 8 * 3;                         // bytes in a frame's colour signature
const STRIP = [0.25, 0.5, 0.75];              // extra sample points for the variant comparison strip
const SAMPLES = 12, SIMILAR_MIN = 0.7;         // whole-clip similarity: frames sampled per clip, floor to list
const TIMEOUT = 20000;
const PLAY_RATE = 4;                           // playback speed of the frame-index pass
const BLANK = 6;                              // luma spread below which a frame counts as blank (black, one flat colour)
const SEG_W = 320, SEG_MAX = 32, SEG_VER = 2;  // shots: sprite cell width, most shots kept per clip, cache version
const CUT_HASH = 90, CUT_COL = 16;             // a cut: neighbouring frames differ by this many hash bits and this much mean colour (0-255)

const S = {
  items: [], groups: [], list: [], meta: new Map(),
  type: 'all', folder: '*', q: '', starred: false, failed: false, sort: 'path', view: 'library',
  thr: 0.96, hover: false, sel: null, source: null, dir: null, detail: null, detailUrl: null, scanId: 0,
};
const C = { open: false };                    // compare view state

const $ = (s, r = document) => r.querySelector(s);
function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const k in attrs || {}) {
    const v = attrs[k];
    if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (k === 'class') e.className = v;
    else if (k in e) e[k] = v;
    else if (v != null && v !== false) e.setAttribute(k, v);
  }
  e.append(...kids.flat(Infinity).filter(x => x != null && x !== false && x !== 0));
  return e;
}
const pct = s => Math.round(s * 100) + '%';
const secs = t => (t < 10 ? t.toFixed(1) : Math.round(t)) + 's';

// ---------- IndexedDB ----------
const DB = (() => {
  let p;
  const open = () => p ||= new Promise((res, rej) => {
    const r = indexedDB.open('dailies', 1);
    r.onupgradeneeded = () => { for (const s of ['cache', 'meta', 'kv']) r.result.createObjectStore(s); };
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
  const run = async (store, mode, fn) => {
    const db = await open();
    return new Promise((res, rej) => {
      const t = db.transaction(store, mode), rq = fn(t.objectStore(store));
      t.oncomplete = () => res(rq.result);
      t.onerror = t.onabort = () => rej(t.error);
    });
  };
  return {
    get: (s, k) => run(s, 'readonly', o => o.get(k)),
    set: (s, k, v) => run(s, 'readwrite', o => o.put(v, k)).catch(e => console.warn('cache write failed', e)),
    del: (s, k) => run(s, 'readwrite', o => o.delete(k)),
    // whole store as a Map, so a rescan is one read instead of one per file
    map: async s => {
      const [ks, vs] = await Promise.all([run(s, 'readonly', o => o.getAllKeys()), run(s, 'readonly', o => o.getAll())]);
      return new Map(ks.map((k, i) => [k, vs[i]]));
    },
  };
})();

// ---------- perceptual hash ----------
const COS = new Float32Array(HK * HN);
for (let k = 0; k < HK; k++) for (let n = 0; n < HN; n++) COS[k * HN + n] = Math.cos(Math.PI * (2 * n + 1) * k / (2 * HN));
const hcx = new OffscreenCanvas(HN, HN).getContext('2d', { willReadFrequently: true });
hcx.imageSmoothingQuality = 'high';

function phash(src) {
  hcx.drawImage(src, 0, 0, HN, HN);
  const d = hcx.getImageData(0, 0, HN, HN).data;
  const g = new Float32Array(HN * HN);
  let m = 0, q = 0;
  for (let i = 0; i < g.length; i++) { g[i] = 0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2]; m += g[i]; q += g[i] * g[i]; }
  phash.sd = Math.sqrt(Math.max(0, q / g.length - (m / g.length) ** 2));   // contrast of the frame just hashed
  const sig = phash.sig = new Uint8Array(SIG);   // and its colour layout: mean RGB of an 8x8 grid
  for (let by = 0; by < 8; by++) for (let bx = 0; bx < 8; bx++) for (let c = 0; c < 3; c++) {
    let a = 0;
    for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++) a += d[((by * 4 + y) * HN + bx * 4 + x) * 4 + c];
    sig[(by * 8 + bx) * 3 + c] = a >> 4;
  }
  const tmp = new Float32Array(HN * HK), c = new Float32Array(HK * HK);
  for (let y = 0; y < HN; y++) for (let k = 0; k < HK; k++) {
    let s = 0;
    for (let x = 0; x < HN; x++) s += g[y * HN + x] * COS[k * HN + x];
    tmp[y * HK + k] = s;
  }
  for (let j = 0; j < HK; j++) for (let k = 0; k < HK; k++) {
    let s = 0;
    for (let y = 0; y < HN; y++) s += tmp[y * HK + k] * COS[j * HN + y];
    c[j * HK + k] = s;
  }
  const med = Float32Array.from(c.subarray(1)).sort()[BITS >> 1];
  const out = new Uint32Array(HK * HK / 32);
  for (let i = 1; i < c.length; i++) if (c[i] > med) out[i >> 5] |= 1 << (i & 31);
  return out;
}
function ham(a, b) {
  let n = 0;
  for (let i = 0; i < a.length; i++) {
    let x = a[i] ^ b[i];
    x -= (x >>> 1) & 0x55555555;
    x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
    n += Math.imul((x + (x >>> 4)) & 0x0f0f0f0f, 0x01010101) >>> 24;
  }
  return n;
}

// ---------- frame capture ----------
function once(v, name) {
  return new Promise((res, rej) => {
    const fin = (f, x) => { clearTimeout(t); v.removeEventListener(name, ok); v.removeEventListener('error', bad); f(x); };
    const ok = () => fin(res), bad = () => fin(rej, new Error(v.error?.message || 'cannot decode'));
    const t = setTimeout(() => fin(rej, new Error('timed out')), TIMEOUT);
    v.addEventListener(name, ok);
    v.addEventListener('error', bad);
  });
}
async function openVideo(file) {
  const v = document.createElement('video'), url = URL.createObjectURL(file);
  v.muted = true; v.playsInline = true; v.preload = 'auto';
  const done = () => { v.removeAttribute('src'); v.load(); URL.revokeObjectURL(url); };
  try {
    const p = once(v, 'loadeddata');
    v.src = url;
    await p;
    if (!v.videoWidth) throw new Error('no video track');
  } catch (e) { done(); throw e; }
  return { v, done };
}
// Seeking past the end clamps to the last frame; 1e9 also covers files that report no duration.
const endTime = v => isFinite(v.duration) ? v.duration : 1e9;
async function seek(v, t) {
  const p = once(v, 'seeked');
  v.currentTime = Math.min(Math.max(t, 0), endTime(v));
  await p;
}
async function capture(src, sw, sh) {
  const cv = new OffscreenCanvas(THUMB_W, Math.max(1, Math.round(THUMB_W * sh / sw))), cx = cv.getContext('2d');
  cx.imageSmoothingQuality = 'high';
  cx.drawImage(src, 0, 0, cv.width, cv.height);
  const hash = phash(cv), sd = phash.sd;
  return { hash, sd, blob: await cv.convertToBlob({ type: 'image/jpeg', quality: 0.82 }) };
}
async function analyze(file, kind) {
  if (kind === 'image') {
    const bmp = await createImageBitmap(file);
    try {
      const f = await capture(bmp, bmp.width, bmp.height);
      return { w: bmp.width, h: bmp.height, first: f.blob, hf: f.hash, sd: f.sd };
    } finally { bmp.close(); }
  }
  const { v, done } = await openVideo(file);
  try {
    const f = await capture(v, v.videoWidth, v.videoHeight);
    await seek(v, Infinity);
    const l = await capture(v, v.videoWidth, v.videoHeight);
    return { w: v.videoWidth, h: v.videoHeight, dur: v.duration, first: f.blob, hf: f.hash, last: l.blob, hl: l.hash };
  } finally { done(); }
}
async function ensureStrip(it) {
  if (it.rec.strip) return;
  const { v, done } = await openVideo(it.file);
  try {
    const out = [];
    for (const f of STRIP) {
      await seek(v, f * v.duration);
      out.push((await capture(v, v.videoWidth, v.videoHeight)).blob);
    }
    it.rec.strip = out;
    DB.set('cache', it.key, it.rec);
  } finally { done(); }
}

// ---------- per-frame index (runs after every scan; hashes, a filmstrip and the first/last non-blank frame are kept) ----------
const fcv = new OffscreenCanvas(THUMB_W, 180), fcx = fcv.getContext('2d');
function frameHash(v) {   // same downscale path as capture(), so these compare cleanly with the boundary hashes
  const hh = Math.max(1, Math.round(THUMB_W * v.videoHeight / v.videoWidth));
  if (fcv.height !== hh) fcv.height = hh;
  fcx.imageSmoothingQuality = 'high';
  fcx.drawImage(v, 0, 0, THUMB_W, hh);
  return phash(fcv);
}
// Seconds per frame, measured from a short burst of playback. Falls back to a fine step; repeats are dropped later.
async function frameStep(v) {
  if (!v.requestVideoFrameCallback) return 1 / 60;
  const ts = [];
  await new Promise(res => {
    const on = (_, m) => { ts.push(m.mediaTime); ts.length < 6 ? v.requestVideoFrameCallback(on) : res(); };
    v.requestVideoFrameCallback(on);
    v.play().catch(res);
    setTimeout(res, 1500);
  });
  v.pause();
  const ds = ts.slice(1).map((t, i) => t - ts[i]).filter(d => d > 1e-4);
  if (!ds.length) return 1 / 60;
  // Smallest gap is one frame; average the gaps near it, since WebM timestamps are rounded to the millisecond.
  const one = ds.filter(d => d < Math.min(...ds) * 1.25);
  return one.reduce((a, b) => a + b) / one.length;
}
// Play frames `from`..`to` of a clip at speed, calling take(k) for each frame as it is presented. Frames the player skips are left out.
async function playThrough(v, d, n, take, from = 0, to = n - 1, rate = PLAY_RATE) {
  if (!v.requestVideoFrameCallback) return;
  await seek(v, from ? (from + 0.5) * d : 0);
  v.playbackRate = rate;
  await new Promise(res => {
    let live = true;
    const stop = () => { live = false; res(); };
    const on = (_, m) => {
      const k = Math.min(n - 1, Math.round(m.mediaTime / d));
      if (!live || k > to) return stop();
      if (k >= from) take(k);
      v.requestVideoFrameCallback(on);
    };
    v.requestVideoFrameCallback(on);
    v.addEventListener('ended', stop, { once: true });
    v.play().catch(stop);
    setTimeout(stop, ((to - from + 1) * d / rate + 5) * 1000);
  });
  v.pause();
  v.playbackRate = 1;
}
const indexed = it => it.rec?.fr?.c;   // an index from before colour signatures is redone
// Clips often open or close on black. Matching and thumbnails use the first and last frame that show something.
const hf = r => r.e1?.h || r.hf, hl = r => r.e2?.h || r.hl;
function blankBelow(it) {
  if (it.bt == null) { const s = Array.from(it.rec.fr.s).sort((a, b) => a - b); it.bt = Math.max(BLANK, 0.3 * s[s.length >> 1]); }
  return it.bt;
}
function setUrls(it) {
  const r = it.rec, set = (k, blob) => { if (it[k]) URL.revokeObjectURL(it[k]); it[k] = blob ? URL.createObjectURL(blob) : null; };
  set('u1', r.e1?.blob || r.first);
  set('u2', r.last && (r.e2?.blob || r.last));
  if (r.film && !it.uf) it.uf = URL.createObjectURL(r.film);
  if (r.sg?.film !== it.ugb) { if (it.ug) URL.revokeObjectURL(it.ug); it.ugb = r.sg?.film; it.ug = it.ugb ? URL.createObjectURL(it.ugb) : null; }
}
// fill = false is the quick pass: only the frames the player presents at speed. fill = true seeks to every frame it skipped.
async function indexFrames(it, fill) {
  const { v, done } = await openVideo(it.file);
  try {
    if (!isFinite(v.duration)) throw new Error('no duration');
    const r = it.rec, d = await frameStep(v), n = Math.max(1, Math.round(v.duration / d)), t = [], hs = [], sd = [], cs = [];
    const film = r.film ? null : filmCanvas(v);   // the harvest filmstrip is drawn on the same pass
    const got = new Array(n), cellAt = i => Math.min(n - 1, Math.floor((i + 0.5) / FILM_N * n));
    let cell = 0, count = 0;
    const take = k => {
      if (got[k]) return;
      got[k] = { h: frameHash(v), s: phash.sd, c: phash.sig };
      count++;
      while (film && cell < FILM_N && cellAt(cell) <= k) film.draw(v, cell++);   // the first frame seen at or after the cell's own
    };
    await playThrough(v, d, n, take);   // hash frames as they play; seeking to each one is several times slower
    fill ||= count < n / 6;             // the player presented almost nothing (hidden tab, no frame callbacks)
    if (fill) for (let k = 0; k < n; k++) if (!got[k]) { await seek(v, (k + 0.5) * d); take(k); }
    for (; film && cell < FILM_N; cell++) { await seek(v, (cellAt(cell) + 0.5) * d); film.draw(v, cell); }
    for (let k = 0; k < n; k++) if (got[k] && (!hs.length || ham(got[k].h, hs.at(-1)))) { t.push(k * d); hs.push(got[k].h); sd.push(Math.min(255, Math.round(got[k].s))); cs.push(got[k].c); }
    const flat = new Uint32Array(hs.length * hs[0].length), col = new Uint8Array(cs.length * SIG);
    hs.forEach((x, k) => flat.set(x, k * x.length));
    cs.forEach((x, k) => col.set(x, k * SIG));
    r.fr = { t: Float32Array.from(t), h: flat, s: Uint8Array.from(sd), c: col, part: count < n };
    r.fd = d;
    if (film) r.film = await film.blob();
    it.fh = it.samp = it.bt = it.cands = null;
    const lo = blankBelow(it), a = sd.findIndex(x => x >= lo), b = sd.findLastIndex(x => x >= lo);
    const grab = async k => { await seek(v, t[k] + d / 2); return { t: t[k], h: hs[k], blob: (await capture(v, v.videoWidth, v.videoHeight)).blob }; };
    delete r.e1; delete r.e2;
    if (a > 0) r.e1 = await grab(a);
    if (b >= 0 && b < sd.length - 1) r.e2 = await grab(b);
    await shoot(it, v);
    DB.set('cache', it.key, r);
    setUrls(it);
  } finally { done(); }
}

// ---------- shots: a clip is split at its hard cuts; each shot keeps one typical frame ----------
const segs = it => it.rec?.sg?.ver === SEG_VER ? it.rec.sg.list : [];
function segment(it) {
  const f = frames(it), { t, s } = it.rec.fr, n = f.length, lo = blankBelow(it), fd = fdur(it);
  // Fast motion scrambles the hash but barely moves the colour layout; a flash of light does the reverse. A cut does both.
  const c = it.rec.fr.c, col = new Float32Array(n);
  for (let k = 1; k < n; k++) { let x = 0; for (let i = 0; i < SIG; i++) x += Math.abs(c[k * SIG + i] - c[(k - 1) * SIG + i]); col[k] = x / SIG; }
  const cutAt = k => {
    const d = ham(f[k - 1], f[k]);
    if (d < CUT_HASH) return false;
    // the quick pass leaves gaps between frames, and motion across a gap looks more like a cut
    const gap = it.rec.fr.part ? Math.min(4, Math.round((t[k] - t[k - 1]) / fd)) : 1;
    if (col[k] >= CUT_COL + 6 * (gap - 1)) return true;
    if (gap > 1 || d < CUT_HASH + 10 || col[k] < 5) return false;
    const w = [];   // between two dark scenes the colour moves little, but still far more than around the cut
    for (let j = Math.max(1, k - 8); j <= Math.min(n - 1, k + 8); j++) if (j !== k) w.push(col[j]);
    w.sort((x, y) => x - y);
    return col[k] >= 4 * (w[w.length >> 1] || 0) + 1;
  };
  const runs = [];
  let a = -1;
  const close = b => { if (a >= 0) runs.push([a, b]); a = -1; };
  for (let k = 0; k < n; k++) {
    if (s[k] < lo) { close(k - 1); continue; }   // blank frames separate shots and belong to none
    if (a >= 0 && cutAt(k)) close(k - 1);
    if (a < 0) a = k;
  }
  close(n - 1);
  const all = runs.map(([a, b]) => {
    // The typical frame: the one closest to the rest of its shot.
    const step = Math.max(1, Math.floor((b - a + 1) / 40));
    let bk = a, bs = Infinity;
    for (let k = a; k <= b; k++) {
      let sum = 0;
      for (let j = a; j <= b && sum < bs; j += step) sum += ham(f[k], f[j]);
      if (sum < bs) { bs = sum; bk = k; }
    }
    return { t0: t[a], t1: b + 1 < n ? t[b + 1] : it.rec.dur, tk: t[bk] };
  });
  const len = x => x.t1 - x.t0, byLen = (x, y) => len(y) - len(x);
  let out = all.filter(x => len(x) > 2.5 * fd);   // a frame or two between cuts is a transition, not a shot
  if (!out.length) out = all.sort(byLen).slice(0, 1);
  return out.sort(byLen).slice(0, SEG_MAX).sort((x, y) => x.t0 - y.t0);
}
// Draw each shot's typical frame into one sprite. `v` is a player showing the clip.
async function shoot(it, v) {
  const list = segment(it), fd = fdur(it), ch = Math.max(1, Math.round(SEG_W * v.videoHeight / v.videoWidth));
  const cv = new OffscreenCanvas(Math.max(1, list.length) * SEG_W, ch), cx = cv.getContext('2d');
  cx.imageSmoothingQuality = 'high';
  for (const [i, s] of list.entries()) { await seek(v, s.tk + fd / 2); cx.drawImage(v, i * SEG_W, 0, SEG_W, ch); }
  it.rec.sg = { ver: SEG_VER, list, film: await cv.convertToBlob({ type: 'image/jpeg', quality: 0.82 }) };
}
async function shootOnly(it) {   // a clip indexed before shots existed: the hashes are there, only the sprite is missing
  const { v, done } = await openVideo(it.file);
  try { await shoot(it, v); DB.set('cache', it.key, it.rec); setUrls(it); } finally { done(); }
}
function shotOf(id) {   // a shot tile's id is s<shot>|<clip id>
  const m = /^s(\d+)\|([\s\S]*)$/.exec(id || ''), it = m && S.items.find(x => x.id === m[2]), s = it && segs(it)[+m[1]];
  return s ? { it, s } : null;
}

const unindexed = () => S.items.filter(it => it.kind === 'video' && it.rec && (!indexed(it) || it.rec.fr.part));
const unshot = () => S.items.filter(it => it.kind === 'video' && indexed(it) && it.rec.sg?.ver !== SEG_VER);
const pending = () => new Set([...unindexed(), ...unshot()]).size;
// A quick pass over every clip so thumbnails, shots and joins appear, then every skipped frame is filled in.
async function indexAll(reopen) {
  await S.indexP;   // a pass over the previous folder winds down first
  const id = S.scanId, live = () => S.indexing && id === S.scanId;
  if (!pending() || S.indexing) return;
  S.indexing = true;
  refresh();
  const pass = async (list, fn, label, every) => {
    let done = 0;
    const timer = setInterval(() => { if (live()) refresh(); }, every);
    await pool(list, 3, async it => {
      if (!live()) return;
      try { await fn(it); } catch (e) { console.warn('frame index failed', it.path, e); }
      if (live()) status(`${label} ${++done} / ${list.length}`, done, list.length);
    });
    clearInterval(timer);
    if (live()) refresh();
  };
  S.indexP = (async () => {
    await pass(S.items.filter(it => it.kind === 'video' && it.rec && !indexed(it)), it => indexFrames(it, false), 'Indexing frames', 5000);
    await pass(unshot(), shootOnly, 'Finding shots', 5000);
    if (live() && reopen && S.detail && !C.open) openDetail(S.detail);
    await pass(unindexed(), it => indexFrames(it, true), 'Thumbnails, shots and joins are ready. Filling in every frame', 20000);
    S.indexing = false;
  })();
  await S.indexP;
  if (id === S.scanId) refresh();
}
function toggleIndex() {
  if (!S.indexing) return indexAll();
  S.indexing = false;
  refresh();
}
const frames = it => it.fh ||= Array.from({ length: it.rec.fr.t.length }, (_, k) => it.rec.fr.h.subarray(k * HK * HK / 32, (k + 1) * HK * HK / 32));
// Best interior frame of `it` for a target hash. Null if nothing clears the threshold, or if an end frame
// already matches (that is a boundary link, not a branch).
function bestFrame(it, target) {
  const f = frames(it), s = it.rec.fr.s, lo = blankBelow(it), maxd = Math.round((1 - S.thr) * BITS), n = f.length;
  if (n < 3 || [hf(it.rec), hl(it.rec)].some(x => ham(x, target) <= maxd)) return null;
  let bk = -1, bd = maxd + 1, d;
  for (let k = 1; k < n - 1; k++) if (s[k] >= lo && (d = ham(f[k], target)) < bd) { bd = d; bk = k; }
  return bk < 0 ? null : { t: it.rec.fr.t[bk], sim: 1 - bd / BITS };
}
function branchLinks(it) {
  const clips = S.items.filter(x => x.kind === 'video' && x.rec && x !== it), stills = S.items.filter(x => x.kind === 'image' && x.rec);
  const out = { to: [], from: [], still: [], offOf: [], into: [] }, add = (list, x, m) => m && list.push({ it: x, ...m });
  if (indexed(it)) {
    for (const b of clips) { add(out.to, b, bestFrame(it, hf(b.rec))); add(out.from, b, bestFrame(it, hl(b.rec))); }
    for (const s of stills) add(out.still, s, bestFrame(it, s.rec.hf));
  }
  for (const x of clips) if (indexed(x)) { add(out.offOf, x, bestFrame(x, hf(it.rec))); add(out.into, x, bestFrame(x, hl(it.rec))); }
  for (const k of ['to', 'from', 'still']) out[k].sort((a, b) => a.t - b.t);
  for (const k of ['offOf', 'into']) out[k].sort((a, b) => b.sim - a.sim);
  return out;
}
// Whole-clip similarity: for evenly sampled frames of each clip, how close is the nearest sampled frame of the other.
function samples(it) {
  const f = frames(it), n = Math.min(SAMPLES, f.length);
  return it.samp ||= Array.from({ length: n }, (_, i) => f[Math.floor((i + 0.5) * f.length / n)]);
}
function clipSim(a, b) {
  const dir = (x, y) => x.reduce((s, p) => s + Math.min(...y.map(q => ham(p, q))), 0) / x.length;
  return 1 - (dir(a, b) + dir(b, a)) / 2 / BITS;
}
function similarClips(it) {
  const mine = samples(it);
  return S.items.filter(x => x !== it && x.kind === 'video' && x.rec?.fr).map(x => ({ it: x, sim: clipSim(mine, samples(x)) }))
    .filter(l => l.sim >= SIMILAR_MIN).sort((a, b) => b.sim - a.sim).slice(0, 8);
}

// ---------- thumbnails: one filmstrip cell per clip, chosen so the grid's thumbnails differ as much as possible ----------
const cellOf = (it, t) => Math.min(FILM_N - 1, Math.max(0, Math.floor(t / it.rec.dur * FILM_N)));
function frameNear(it, t) {   // index of the distinct frame on screen at t
  const ts = it.rec.fr.t;
  let lo = 0, hi = ts.length - 1;
  while (lo < hi) { const m = (lo + hi + 1) >> 1; ts[m] <= t ? lo = m : hi = m - 1; }
  return lo;
}
function pickThumbs() {
  const clips = S.items.filter(it => it.kind === 'video' && indexed(it) && it.rec.film);
  for (const it of clips) {
    if (it.cands) continue;
    const f = frames(it), lo = blankBelow(it);
    const all = Array.from({ length: FILM_N }, (_, i) => {
      const j = frameNear(it, (i + 0.5) / FILM_N * it.rec.dur);
      return { i, h: f[j], s: it.rec.fr.s[j] };
    }).sort((a, b) => Math.abs(a.i - FILM_N / 2) - Math.abs(b.i - FILM_N / 2));   // ties go to the middle of the clip
    const ok = all.filter(c => c.s >= lo), prev = it.tc?.i;
    it.cands = ok.length ? ok : all;
    it.tc = it.cands.find(c => c.i === prev) || it.cands[0];   // a re-index keeps the thumbnail unless a better one exists
  }
  // Each clip in turn takes the cell farthest from every other clip's current thumbnail.
  for (let pass = 0; pass < 2; pass++) for (const it of clips) {
    let best = it.tc, bd = -1;
    for (const c of [it.tc, ...it.cands]) {
      let d = Infinity;
      for (const o of clips) if (o !== it && (d = Math.min(d, ham(c.h, o.tc.h))) <= bd) break;
      if (d > bd) { bd = d; best = c; }
    }
    it.tc = best;
  }
}
function spriteEl(url, n, i, ar) {   // cell i of a sprite of n frames
  return h('div', { class: 'cell', style: `--ar:${ar};background-image:url(${url});`
    + `background-size:${n * 100}% 100%;background-position:${n > 1 ? (100 * i / (n - 1)).toFixed(3) : 0}% 0` });
}
const cellEl = (it, i) => spriteEl(it.uf, FILM_N, i, it.rec.w / it.rec.h);

// ---------- joins: frames shared between clips, found through stills and through each clip's own start and end ----------
const REL = { out: '→', in: '←', same: '↔' };
function matchIn(it, target, maxd) {   // best non-blank frame of an indexed clip for a hash
  const f = frames(it), s = it.rec.fr.s, lo = blankBelow(it);
  let bk = -1, bd = maxd + 1, d;
  for (let k = 0; k < f.length; k++) if (s[k] >= lo && (d = ham(f[k], target)) < bd) { bd = d; bk = k; }
  return bk < 0 ? null : { t: it.rec.fr.t[bk], sim: 1 - bd / BITS };
}
const startT = it => it.rec.e1?.t ?? 0, endT = it => it.rec.e2?.t ?? it.rec.fr.t.at(-1);
function joins() {
  if (S.joins) return S.joins;
  const maxd = Math.round((1 - S.thr) * BITS), near = (a, b) => Math.abs(a - b) < 0.25;
  const clips = S.items.filter(it => it.kind === 'video' && indexed(it) && it.rec.fr.s.some(x => x >= blankBelow(it)));
  // Sources that hash alike are one entry: variants sharing a start, a still that is some clip's start, near-duplicate stills.
  const src = [];
  for (const c of clips) for (const [kind, hash, url, t] of [['start', hf(c.rec), c.u1, startT(c)], ['end', hl(c.rec), c.u2, endT(c)]]) {
    const e = src.find(x => x.kind === kind && ham(x.h, hash) <= maxd);
    e ? e.members.push({ it: c, t }) : src.push({ kind, h: hash, url, members: [{ it: c, t }], stills: [] });
  }
  for (const s of S.items) if (s.kind === 'image' && s.rec && !(s.rec.sd < BLANK)) {
    const e = src.find(x => ham(x.h, s.rec.hf) <= maxd);
    e ? e.stills.push(s) : src.push({ kind: 'still', h: s.rec.hf, url: s.u1, members: [], stills: [s] });
  }
  const list = [], links = new Map();
  const link = (a, b, rel, via, sim) => {
    const put = (x, y, rel) => (links.get(x.it) || links.set(x.it, []).get(x.it)).push({ t: x.t, other: y, rel, via, sim });
    put(a, b, rel);
    put(b, a, rel === 'out' ? 'in' : rel === 'in' ? 'out' : rel);
  };
  for (const e of src) {
    const own = new Set(e.members.map(m => m.it));
    let hits = [];
    for (const c of clips) { const m = !own.has(c) && matchIn(c, e.h, maxd); if (m) hits.push({ it: c, ...m }); }
    // The same start is a variant, not a join; an end that meets a start is listed once, under that start.
    if (e.kind === 'start') hits = hits.filter(x => !near(x.t, startT(x.it)));
    if (e.kind === 'end') hits = hits.filter(x => !near(x.t, endT(x.it)) && !near(x.t, startT(x.it)));
    if (hits.length < (e.kind === 'still' ? 2 : 1)) continue;
    hits = hits.sort((a, b) => b.sim - a.sim).slice(0, 12);
    list.push({ ...e, hits });
    const via = e.kind === 'still' ? `still ${e.stills[0].name}` : `${e.kind} of ${e.members[0].it.name}`;
    if (e.kind === 'still') hits.forEach((a, i) => hits.slice(i + 1).forEach(b => link(a, b, 'same', via, Math.min(a.sim, b.sim))));
    else for (const m of e.members) for (const x of hits) link(x, m, e.kind === 'start' ? 'out' : 'in', via, x.sim);
  }
  for (const [it, L] of links) {
    const seen = new Map();
    for (const l of L.sort((a, b) => b.sim - a.sim)) {
      const k = `${l.other.it.id}|${l.t.toFixed(1)}|${l.other.t.toFixed(1)}`;
      if (!seen.has(k)) seen.set(k, l);
    }
    links.set(it, [...seen.values()].sort((a, b) => a.t - b.t));
  }
  return S.joins = { list, links };
}

// ---------- scanning ----------
async function* walk(dir, prefix = '') {
  for await (const [name, handle] of dir.entries()) {
    if (name.startsWith('.') || (!prefix && (name === SDIR || name === CDIR))) continue;   // exported frames and sub-clips are output, not source
    if (handle.kind === 'directory') yield* walk(handle, prefix + name + '/');
    else if (IMG.test(name) || VID.test(name)) yield { path: prefix + name, getFile: () => handle.getFile() };
  }
}
async function pool(list, n, fn) {
  let i = 0;
  await Promise.all(Array.from({ length: n }, async () => { while (i < list.length) await fn(list[i++]); }));
}
function status(text, done, total) {
  $('#statusText').replaceChildren(...[].concat(text));
  const p = $('#prog');
  p.hidden = total == null;
  if (total != null) { p.max = total; p.value = done; }
}

async function makeItem(path, getFile, cache) {
  const kind = VID.test(path) ? 'video' : 'image', slash = path.lastIndexOf('/');
  const it = { id: path, path, name: path.slice(slash + 1), folder: slash < 0 ? '' : path.slice(0, slash), kind };
  try {
    const file = await getFile();
    Object.assign(it, { file, size: file.size, mtime: file.lastModified,
      key: `${file.name}|${file.size}|${file.lastModified}`, mkey: `${file.name}|${file.size}` });
    it.rec = cache?.get(it.key);
    it.hit = !!it.rec;
    if (!it.rec) { it.rec = await analyze(file, kind); DB.set('cache', it.key, it.rec); }
    setUrls(it);
  } catch (x) { it.err = x.message || String(x); it.mkey ||= it.path; }
  return it;
}
// entries: [{ path, getFile() }] where path is relative to the chosen folder
async function scan(entries) {
  const id = ++S.scanId;
  S.scanning = true;
  closeCompare(); closeDetail(); closeSheet();
  const [cache, meta] = await Promise.all([DB.map('cache'), DB.map('meta')]);
  S.meta = meta;
  for (const it of S.items) for (const u of [it.u1, it.u2, it.uf, it.ug, ...(it.us || [])]) u && URL.revokeObjectURL(u);
  S.items = [];
  let done = 0, hits = 0;
  const timer = setInterval(() => refresh(), 3000);
  await pool(entries, 4, async e => {
    if (id !== S.scanId) return;
    const it = await makeItem(e.path, e.getFile, cache);
    if (id !== S.scanId) return;
    if (it.hit) hits++;
    S.items.push(it);
    status(`Scanning ${++done} / ${entries.length} (${hits} from cache)`, done, entries.length);
  });
  clearInterval(timer);
  if (id !== S.scanId) return;
  S.scanning = false;
  $('#rescan').hidden = false;
  refresh();
  $('#grid').focus();
}

async function pickFolder() {
  if (!window.showDirectoryPicker) return $('#fallback').click();
  let dir;
  try { dir = await showDirectoryPicker({ id: 'media', mode: 'read' }); } catch { return; }
  DB.set('kv', 'dir', dir);
  useDir(dir);
}
function useDir(dir) {
  $('#reopen').hidden = true;
  S.dir = dir;
  S.source = async () => {
    status(`Listing ${dir.name}…`);
    const entries = [];
    for await (const e of walk(dir)) entries.push(e);
    await scan(entries);
    await loadCollections();
    indexAll();
  };
  loadBoard();
  S.source();
}
function useFileList(files) {   // fallback for browsers without the File System Access API
  const entries = [...files].map(f => ({ path: f.webkitRelativePath.split('/').slice(1).join('/') || f.name, getFile: async () => f }))
    .filter(e => (IMG.test(e.path) || VID.test(e.path)) && !e.path.split('/').some(p => p.startsWith('.')));
  S.dir = null;
  K.cols = [];
  S.source = async () => { await scan(entries); indexAll(); };
  loadBoard();
  S.source();
}

// ---------- matching ----------
function computeLinks() {
  const maxd = Math.round((1 - S.thr) * BITS), sim = d => 1 - d / BITS;
  const ok = S.items.filter(it => it.rec);
  const clips = ok.filter(it => it.kind === 'video'), stills = ok.filter(it => it.kind === 'image');
  for (const it of S.items) { it.links = { fromStill: [], endStill: [], prev: [], next: [], starts: [], ends: [] }; it.group = null; }
  const parent = clips.map((_, i) => i);
  const find = i => { while (parent[i] !== i) i = parent[i] = parent[parent[i]]; return i; };
  let d;
  clips.forEach((a, i) => {
    const A = a.rec, La = a.links;
    for (const s of stills) {
      if ((d = ham(s.rec.hf, hf(A))) <= maxd) { La.fromStill.push({ it: s, sim: sim(d) }); s.links.starts.push({ it: a, sim: sim(d) }); }
      if ((d = ham(s.rec.hf, hl(A))) <= maxd) { La.endStill.push({ it: s, sim: sim(d) }); s.links.ends.push({ it: a, sim: sim(d) }); }
    }
    for (let j = i + 1; j < clips.length; j++) {
      const b = clips[j], B = b.rec;
      if (ham(hf(A), hf(B)) <= maxd) parent[find(i)] = find(j);
      if ((d = ham(hl(A), hf(B))) <= maxd) { La.next.push({ it: b, sim: sim(d) }); b.links.prev.push({ it: a, sim: sim(d) }); }
      if ((d = ham(hl(B), hf(A))) <= maxd) { b.links.next.push({ it: a, sim: sim(d) }); La.prev.push({ it: b, sim: sim(d) }); }
    }
  });
  for (const it of ok) for (const k in it.links) it.links[k].sort((x, y) => y.sim - x.sim);
  const by = new Map();
  clips.forEach((c, i) => { const r = find(i); by.has(r) ? by.get(r).push(c) : by.set(r, [c]); });
  S.groups = [...by.values()].filter(m => m.length > 1).map((members, id) => {
    const g = { id, members: members.sort((a, b) => a.path.localeCompare(b.path)) };
    for (const m of members) m.group = g;
    return g;
  });
}

// ---------- metadata (tags, stars) ----------
const meta = it => S.meta.get(it.mkey) || { tags: [], star: false };
function saveMeta(it, m) {
  if (m.star || m.tags.length) { S.meta.set(it.mkey, m); DB.set('meta', it.mkey, m); }
  else { S.meta.delete(it.mkey); DB.del('meta', it.mkey); }
  if (S.view === 'groups') renderMain();
  else $(`#grid .tile[data-id="${CSS.escape(it.id)}"]`)?.replaceWith(tile(it));
}
function toggleStar(it) { const m = meta(it); saveMeta(it, { ...m, star: !m.star }); }

// ---------- grid ----------
function visible() {
  const toks = S.q.toLowerCase().split(/\s+/).filter(Boolean);
  const list = S.items.filter(it => {
    const m = meta(it);
    return (S.type === 'all' || it.kind === S.type)
      && (S.folder === '*' || it.folder === S.folder || it.folder.startsWith(S.folder + '/'))
      && (!S.starred || m.star) && (!S.failed || it.err)
      && toks.every(t => t[0] === '#' ? m.tags.some(x => x.startsWith(t.slice(1)))
        : it.path.toLowerCase().includes(t) || m.tags.some(x => x.includes(t)));
  });
  const cmp = { path: (a, b) => a.path.localeCompare(b.path, undefined, { numeric: true }),
    name: (a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }), mtime: (a, b) => b.mtime - a.mtime }[S.sort];
  return list.sort(cmp);
}
function tile(it) {
  const m = meta(it), L = it.links, chip = (cls, text, title) => h('span', { class: 'chip ' + cls, title }, text);
  return h('div', { class: 'tile' + (it.id === S.sel ? ' sel' : ''), 'data-id': it.id },
    h('div', { class: 'thumb' },
      !it.u1 ? h('div', { class: 'err' }, 'Cannot decode: ' + it.err)
        : it.tc && it.uf ? cellEl(it, it.tc.i) : h('img', { loading: 'lazy', src: it.u1, alt: '', draggable: false }),
      h('span', { class: 'kind' }, it.kind === 'image' ? 'still' : it.rec ? '▶ ' + secs(it.rec.dur) : '▶'),
      m.star && h('span', { class: 'starmark' }, '★'),
      it.kind === 'image' && it.rec && h('button', { class: 'addbtn', title: `Add to the collection “${K.active}” (A)`,
        onclick: e => { e.stopPropagation(); collect(it); } }, '+'),
      h('div', { class: 'chips' }, it.kind === 'video' ? [
        L.fromStill.length && chip('still', '◧ still', 'First frame matches a still'),
        L.prev.length && chip('prev', '← ' + L.prev.length, 'Clips this one can follow'),
        L.next.length && chip('next', L.next.length + ' →', 'Clips that can follow this one'),
        it.group && chip('var', '≋ ' + it.group.members.length, 'Variants sharing this first frame'),
      ] : [
        L.starts.length && chip('next', '▶ ' + L.starts.length, 'Clips that start on this still'),
        L.ends.length && chip('prev', L.ends.length + ' ■', 'Clips that end on this still'),
      ])),
    h('div', { class: 'name', title: it.path }, it.name),
    h('div', { class: 'tags' }, m.tags.map(t => '#' + t).join(' ')));
}
function mini(it, url, note, onclick, cell) {   // cell: show that filmstrip cell instead of an image
  return h('button', { class: 'mini' + (meta(it).star ? ' star' : ''), title: it.path, onclick },
    cell != null && it.uf ? h('div', { class: 'cellbox' }, cellEl(it, cell)) : h('img', { loading: 'lazy', src: url || it.u1, alt: '' }),
    h('span', {}, (meta(it).star ? '★ ' : '') + it.name), note && h('span', { class: 'sim' }, note));
}
function groupTile(g) {
  const picked = g.members.filter(m => meta(m).star);
  return h('div', { class: 'tile group' + (S.sel === 'g' + g.id ? ' sel' : ''), 'data-id': 'g' + g.id },
    h('h3', {}, `${g.members.length} variants `, h('small', {}, g.members[0].folder || '/',
      picked.length ? ` · picked: ${picked.map(p => p.name).join(', ')}` : ' · none picked')),
    h('div', { class: 'row' }, mini(g.members[0], g.members[0].u1, 'shared first frame'),
      g.members.map(m => mini(m, m.u2, 'last frame · ' + secs(m.rec.dur)))));
}
const picksIn = (cuts, s) => cuts.filter(f => f.time >= s.t0 && f.time < s.t1).sort((a, b) => a.mtime - b.mtime);
function shotTile(it, s, i, cuts) {
  const r = it.rec, id = `s${i}|${it.id}`, n = segs(it).length, mine = picksIn(cuts, s);
  const g = clipRange(it, s), fd = fdur(it), len = s.t1 - s.t0, saved = K.saved.has(clipName(it, g));
  const keep = g.a !== g.a0 || g.b !== g.b0;   // the sub-clip is shorter than the shot
  return h('div', { class: 'tile shot' + (id === S.sel ? ' sel' : ''), 'data-id': id },
    h('div', { class: 'thumb', title: 'Click to lay out every frame of this shot and pick the ones to keep · move across to preview it' },
      // a shot is shown by the frame picked for it; until one is picked, by its typical frame
      mine.length ? h('img', { loading: 'lazy', src: mine.at(-1).url, alt: '' }) : spriteEl(it.ug, n, i, r.w / r.h),
      h('span', { class: 'kind' }, `${secs(s.t1 - s.t0)} · ${Math.max(1, Math.round((s.t1 - s.t0) / fdur(it)))} frames`),
      (mine.length > 0 || saved) && h('span', { class: 'starmark', title: 'Frames picked from this shot · ▶ saved as a clip' },
        [mine.length > 0 && '★ ' + mine.length, saved && '▶'].filter(Boolean).join(' ')),
      keep && h('span', { class: 'keep', title: 'The part of this shot that is saved as a clip',
        style: `left:${(100 * (g.a * fd - s.t0) / len).toFixed(2)}%;width:${(100 * (g.b - g.a + 1) * fd / len).toFixed(2)}%` }),
      h('button', { class: 'addbtn', title: `Save this shot as its own clip in “${CDIR}” (V). Open the shot to set where the clip starts and ends.`,
        onclick: e => { e.stopPropagation(); saveClip([it, s]); } }, '▶ clip'),
      h('span', { class: 'span', title: 'Where this shot sits in its clip',
        style: `left:${(100 * s.t0 / r.dur).toFixed(2)}%;width:${(100 * (s.t1 - s.t0) / r.dur).toFixed(2)}%` })),
    h('div', { class: 'name', title: it.path }, h('a', { title: 'Open the clip at this shot' }, it.name)),
    h('div', { class: 'tags' }, `shot ${i + 1} of ${n} · ${s.t0.toFixed(2)}–${s.t1.toFixed(2)}s` + (keep ? ` · clip ${(g.a * fd).toFixed(2)}–${((g.b + 1) * fd).toFixed(2)}s` : '')));
}
function cutsByClip() {   // clip path -> frames cut from it, across all collections
  const cuts = new Map();
  for (const c of K.cols) for (const f of c.frames) if (f.clip) cuts.set(f.clip, [...(cuts.get(f.clip) || []), { ...f, col: c }]);
  return cuts;
}
function joinTile(e, id) {
  const who = e.members.map(m => m.it.name).join(', '), n = e.hits.length, clips = n === 1 ? '1 clip passes' : `${n} clips pass`;
  const first = e.members[0], also = e.stills.length && e.kind !== 'still' ? ` · also the still ${e.stills[0].name}` : '';
  return h('div', { class: 'tile group' + (S.sel === 'j' + id ? ' sel' : ''), 'data-id': 'j' + id },
    h('h3', {}, e.kind === 'still' ? `${n} clips share the still ${e.stills[0].name} ` : `${clips} through the ${e.kind} of ${who} `,
      h('small', {}, (e.kind === 'start' ? 'cut there and continue with this clip' : e.kind === 'end' ? 'this clip can lead into them there'
        : 'they can be cut together at this frame') + also)),
    h('div', { class: 'row' },
      first ? mini(first.it, e.url, `${e.kind} · ${first.t.toFixed(2)}s`, () => openDetail(first.it, first.t))
        : mini(e.stills[0], e.url, 'still', () => openDetail(e.stills[0])),
      e.hits.map(x => mini(x.it, null, `at ${x.t.toFixed(2)}s · ${pct(x.sim)}`, () => openDetail(x.it, x.t), cellOf(x.it, x.t)))));
}
function renderMain() {
  const grid = $('#grid');
  S.list = visible();
  grid.classList.toggle('groups', S.view === 'groups' || S.view === 'joins');
  grid.classList.toggle('harvest', S.view === 'harvest');
  grid.classList.toggle('board', S.view === 'board');
  if (S.view !== 'board') stopBoardPlay();
  if (S.view === 'harvest') return renderHarvest();
  K.v.pause();
  if (S.view === 'board') return renderBoard();
  let nodes;
  if (S.view === 'groups') {
    const vis = new Set(S.list);
    nodes = S.groups.filter(g => g.members.some(m => vis.has(m))).map(groupTile);
  } else if (S.view === 'joins') {
    const vis = new Set(S.list);
    nodes = joins().list.map((e, id) => [...e.members, ...e.hits].some(x => vis.has(x.it)) && joinTile(e, id)).filter(Boolean);
  } else if (S.view === 'shots') {
    const cuts = cutsByClip(), clips = S.list.filter(it => segs(it).length);
    nodes = clips.flatMap(it => segs(it).map((s, i) => shotTile(it, s, i, cuts.get(it.path) || [])));
    if (nodes.length) nodes.unshift(h('div', { class: 'shotbar' },
      `${nodes.length} shots in ${clips.length} clips · click a shot to see every frame and pick the ones to keep · move across a shot to preview it`,
      S.dir && h('button', { title: `Writes one full-size frame per shot to the folder “${SDIR}” inside the opened folder: the frames you picked, or the typical frame of a shot you have not picked from. Replaces the last export.`,
        onclick: exportShots }, 'Save one frame per shot'),
      h('button', { title: `Cuts every shot listed here into its own clip in the folder “${CDIR}”. Filter by name first to do one clip.`,
        onclick: () => saveClip(...clips.flatMap(it => segs(it).map(s => [it, s]))) }, 'Save every shot as a clip'),
      h('label', { title: 'Frames dropped from each end of a shot when it is saved as a clip, for cuts that smear across a frame or two. Open a shot to set its ends by hand.' },
        'Trim ends ', h('input', { type: 'number', min: 0, max: 30, value: K.pad, style: 'width:4em',
          onchange: e => { K.pad = Math.max(0, Math.min(30, Math.round(+e.target.value) || 0)); DB.set('kv', 'kpad', K.pad); renderMain(); } }), ' frames')));
  } else nodes = S.list.map(tile);
  if (!nodes.length) nodes = [h('div', { class: 'empty' }, !S.items.length ? 'No folder open.'
    : S.view === 'groups' ? 'No variant groups at this threshold.'
    : S.view === 'shots' ? (pending() ? 'Shots appear as frames are indexed.' : 'No clips match the current filters.')
    : S.view === 'joins' ? (unindexed().length ? 'Joins appear as frames are indexed.' : 'No clips share a frame at this threshold. Lower “Match ≥” to loosen it.')
    : 'Nothing matches the current filters.')];
  grid.replaceChildren(...nodes);
  $('#exportStills').hidden = !(S.view === 'library' && S.type === 'image' && S.list.some(it => it.file));
  if (!$('#grid .tile.sel')) { S.sel = null; const first = $('#grid .tile'); if (first) select(first.dataset.id, false); }
}
function refresh() {
  computeLinks();
  pickThumbs();
  S.joins = null;
  const folders = [...new Set(S.items.flatMap(it => it.folder.split('/').map((_, i, a) => a.slice(0, i + 1).join('/'))))].filter(Boolean).sort();
  $('#folder').replaceChildren(h('option', { value: '*' }, 'All folders'), ...folders.map(f => h('option', { value: f }, f)));
  if (!folders.includes(S.folder)) S.folder = '*';
  $('#folder').value = S.folder;
  renderMain();
  const todo = pending(), ib = $('#index');
  ib.hidden = !S.items.some(it => it.kind === 'video' && it.rec);
  ib.disabled = !todo && !S.indexing;
  ib.textContent = S.indexing ? `Stop indexing (${todo} left)` : todo ? `Index frames (${todo})` : 'Frames indexed ✓';
  const bad = S.items.filter(it => it.err).length, n = k => S.items.filter(it => it.kind === k).length;
  if (S.items.length) status([`${n('image')} stills · ${n('video')} clips · ${S.groups.length} variant groups`,
    bad ? [' · ', h('a', { onclick: () => { S.failed = !S.failed; renderMain(); } }, `${bad} could not be decoded`)] : ''].flat());
}
function select(id, scroll = true) {
  $('#grid .tile.sel')?.classList.remove('sel');
  S.sel = id;
  const t = $(`#grid .tile[data-id="${CSS.escape(id)}"]`);
  t?.classList.add('sel');
  if (scroll) t?.scrollIntoView({ block: 'nearest' });
}
function openSelected() {
  if (S.sel == null) return;
  if (S.view === 'groups') openCompare(S.groups[+S.sel.slice(1)]);
  else if (S.view === 'shots') { const x = shotOf(S.sel); if (x) openSheet(x.it, segs(x.it).indexOf(x.s)); }
  else if (S.view === 'joins') { const e = joins().list[+S.sel.slice(1)], m = e?.members[0]; m ? openDetail(m.it, m.t) : openDetail(e?.stills[0]); }
  else openDetail(S.items.find(it => it.id === S.sel));
}
function moveSel(dx, dy) {
  const tiles = [...document.querySelectorAll('#grid .tile')];
  if (!tiles.length) return;
  const cols = tiles.filter(t => t.offsetTop === tiles[0].offsetTop).length;
  let i = tiles.findIndex(t => t.dataset.id === S.sel);
  const j = i + dx + dy * cols;
  if (j >= 0 && j < tiles.length) i = j; else if (dy > 0) i = tiles.length - 1;
  select(tiles[Math.max(i, 0)].dataset.id);
}

// hover-to-play in the grid
let hoverVid = null;
function stopHover() {
  if (!hoverVid) return;
  hoverVid.v.remove(); URL.revokeObjectURL(hoverVid.url); hoverVid = null;
}
function startHover(t) {
  stopHover();
  const it = S.items.find(x => x.id === t.dataset.id);
  if (!it || it.kind !== 'video' || !it.rec) return;
  const url = URL.createObjectURL(it.file);
  const v = h('video', { src: url, muted: true, loop: true, autoplay: true, playsInline: true });
  hoverVid = { v, url, t };
  $('.thumb', t).append(v);
}

// ---------- detail ----------
function closeDetail() {
  if (S.detailUrl) URL.revokeObjectURL(S.detailUrl);
  S.detail = S.detailUrl = null;
  $('#detail').hidden = true;
  $('#detail').replaceChildren();
}
function openDetail(it, at) {
  if (!it) return;
  closeDetail();
  S.detail = it;
  if (S.view === 'library' && S.list.includes(it)) select(it.id);
  const L = it.links, r = it.rec;
  const section = (title, hint, links, frame) => links.length ? h('section', { class: 'links' },
    h('h3', {}, title, h('small', {}, hint)),
    h('div', { class: 'row' }, links.map(l => mini(l.it, frame === 'last' ? l.it.u2 : l.it.u1, pct(l.sim) + ' match', () => openDetail(l.it))))) : null;

  let frames, sections, branchSections = [], hint = null;
  if (it.err) frames = h('p', {}, `This file could not be decoded by the browser (${it.err}).`);
  else if (it.kind === 'image') {
    S.detailUrl = URL.createObjectURL(it.file);
    frames = h('div', { class: 'frames one' }, h('figure', {}, h('img', { src: S.detailUrl, alt: '' })));
    const maxd = Math.round((1 - S.thr) * BITS);
    const inside = S.items.filter(c => c.kind === 'video' && indexed(c)).map(c => ({ it: c, ...matchIn(c, r.hf, maxd) }))
      .filter(l => l.t != null).sort((a, b) => b.sim - a.sim);
    sections = [section('Clips that start on this still', 'their first frame', L.starts, 'first'),
      section('Clips that end on this still', 'their last frame', L.ends, 'last'),
      inside.length ? h('section', { class: 'links' },
        h('h3', {}, 'Appears inside clips', h('small', {}, (inside.length > 1 ? 'these clips can be cut together at this frame · ' : '') + 'opens at that frame')),
        h('div', { class: 'row' }, inside.map(l => mini(l.it, null, `at ${l.t.toFixed(2)}s · ${pct(l.sim)}`, () => openDetail(l.it, l.t), cellOf(l.it, l.t))))) : null];
  } else {
    S.detailUrl = URL.createObjectURL(it.file);
    const first = h('video', { id: 'dplay', src: S.detailUrl, muted: true, playsInline: true, preload: 'auto', controls: true });
    const last = h('video', { src: S.detailUrl, muted: true, playsInline: true, preload: 'auto' });
    const t1 = at ?? r.e1?.t;   // skip a blank lead-in
    last.addEventListener('loadedmetadata', () => { last.currentTime = r.e2 ? r.e2.t + 0.005 : endTime(last); }, { once: true });
    if (t1 != null) first.addEventListener('loadedmetadata', () => { first.currentTime = t1 + 0.005; }, { once: true });
    const B = branchLinks(it), at2 = l => `at ${l.t.toFixed(2)}s · ${pct(l.sim)}`;
    const jump = t => { first.pause(); first.currentTime = t + 0.005; };   // nudge past the frame boundary
    const bsec = (title, hint, links, frame, inOther) => links.length ? h('section', { class: 'links' },
      h('h3', {}, title, h('small', {}, hint)),
      h('div', { class: 'row' }, links.map(l => mini(l.it, frame === 'last' ? l.it.u2 : l.it.u1, at2(l), () => inOther ? openDetail(l.it, l.t) : openDetail(l.it))))) : null;
    const ticks = [['to', '→ branch out to '], ['from', '← join from '], ['still', '◧ passes '] ].flatMap(([k, label]) =>
      B[k].map(l => h('button', { class: 'tk ' + k, style: `left:${(100 * l.t / r.dur).toFixed(2)}%`, title: `${l.t.toFixed(2)}s ${label}${l.it.name}`, onclick: () => jump(l.t) })));
    frames = [h('div', { class: 'frames' },
      h('figure', {}, h('figcaption', {}, (at != null ? `Frame at ${at.toFixed(2)}s` : r.e1 ? `First frame after ${secs(r.e1.t)} of blank` : 'First frame')
        + ' · Space plays from here · A cuts the frame on screen into the collection'), first),
      h('figure', {}, h('figcaption', {}, r.e2 ? 'Last frame before the blank tail' : 'Last frame'), last)),
      ticks.length ? h('div', { class: 'tl', title: 'Branch points in this clip; click to jump the player there' }, ticks) : null];
    const J = indexed(it) ? joins().links.get(it) || [] : [];
    branchSections = [
      J.length ? h('section', { class: 'links' },
        h('h3', {}, 'Can be combined with', h('small', {}, 'a frame both clips share · this clip ↔ theirs · opens their clip at that frame')),
        h('div', { class: 'row' }, J.map(l => mini(l.other.it, null, `${l.t.toFixed(2)}s ${REL[l.rel]} ${l.other.t.toFixed(2)}s · ${l.via}`,
          () => openDetail(l.other.it, l.other.t), cellOf(l.other.it, l.other.t))))) : null,
      bsec('Branch out mid-clip', 'their first frame matches a frame inside this clip', B.to, 'first'),
      bsec('Join in mid-clip', 'their last frame matches a frame inside this clip', B.from, 'last'),
      bsec('Passes through still', 'a frame inside this clip matches', B.still, 'first'),
      bsec('Branches off from', 'this first frame appears inside these clips · opens at that frame', B.offOf, 'first', true),
      bsec('Can rejoin', 'this last frame appears inside these clips · opens at that frame', B.into, 'first', true),
    ];
    if (indexed(it)) {
      const sim = similarClips(it);
      branchSections.push(sim.length ? h('section', { class: 'links' }, h('h3', {}, 'Visually similar clips', h('small', {}, 'by sampled frames across the whole clip')),
        h('div', { class: 'row' }, sim.map(l => mini(l.it, l.it.u1, pct(l.sim) + ' similar', () => openDetail(l.it))))) : null);
    } else hint = h('p', {}, h('small', {}, 'Frames of this clip are not indexed yet. '), h('button', { onclick: () => indexAll(true) }, 'Index frames'),
      h('small', {}, ' to find branch points inside it and visually similar clips.'));
    sections = [
      section('Starts from still', 'matches this first frame', L.fromStill, 'first'),
      section('Can come before', 'their last frame matches this first frame', L.prev, 'last'),
      section('Can come after', 'their first frame matches this last frame', L.next, 'first'),
      section('Ends on still', 'matches this last frame', L.endStill, 'first'),
      it.group && h('section', { class: 'links' },
        h('h3', {}, `Variants (${it.group.members.length})`, h('small', {}, 'same first frame · last frames shown'),
          ' ', h('button', { onclick: () => openCompare(it.group) }, 'Compare (C)')),
        h('div', { class: 'row' }, it.group.members.map(m => mini(m, m.u2, m === it ? 'this clip' : secs(m.rec.dur), () => openDetail(m))))),
      ...branchSections,
    ];
  }
  if (sections && !sections.some(Boolean)) sections = [h('p', {}, h('small', {}, 'No boundary matches or variants at the current threshold.'))];

  const info = r ? [`${r.w}×${r.h}`, r.dur && secs(r.dur), (it.size / 1e6).toFixed(1) + ' MB', new Date(it.mtime).toLocaleDateString()].filter(Boolean).join(' · ') : '';
  const head = h('div', { class: 'ohead' },
    h('button', { onclick: closeDetail }, '← Back'),
    h('span', { class: 'path', title: 'Click to copy path', onclick: () => navigator.clipboard?.writeText(it.path) }, it.path),
    h('small', {}, info), h('span', { class: 'grow' }),
    h('span', { id: 'dmeta', class: 'tagrow' }));
  $('#detail').replaceChildren(head, h('div', { class: 'obody' }, frames, sections, hint));
  renderDetailMeta();
  $('#detail').hidden = false;
}
function renderDetailMeta() {
  const it = S.detail, m = meta(it);
  const add = e => {
    if (e.key !== 'Enter') return;
    const t = e.target.value.trim().toLowerCase().replace(/^#/, '');
    if (t && !m.tags.includes(t)) saveMeta(it, { ...m, tags: [...m.tags, t] });
    renderDetailMeta();
    $('#tagIn').focus();
  };
  $('#dmeta').replaceChildren(
    h('select', { class: 'kcolsel', title: 'Collection that A adds to' }),
    h('button', { title: it.kind === 'video' ? 'Cut the frame on screen into the collection' : 'Add this still to the collection',
      onclick: () => collect(it) }, it.kind === 'video' ? '+ Cut frame (A)' : '+ Add (A)'),
    ...m.tags.map(t => h('span', { class: 'tag' }, '#' + t,
      h('button', { title: 'Remove tag', onclick: () => { saveMeta(it, { ...m, tags: m.tags.filter(x => x !== t) }); renderDetailMeta(); } }, '×'))),
    h('input', { id: 'tagIn', type: 'text', placeholder: 'Add tag (T)', onkeydown: add }),
    h('button', { class: 'starbtn' + (m.star ? ' on' : ''), onclick: () => { toggleStar(it); renderDetailMeta(); } }, m.star ? '★ Starred' : '☆ Star (S)'));
  fillColSelects();
}
function stepDetail(d) {
  const i = S.list.indexOf(S.detail), n = S.list[i + d];
  if (i >= 0 && n) openDetail(n);
}

// ---------- variant comparison ----------
const LUT = (() => {   // black -> indigo -> magenta -> orange -> pale yellow
  const stops = [[0, 0, 0], [40, 20, 120], [170, 40, 130], [240, 110, 40], [255, 215, 80], [255, 255, 235]], lut = new Uint8Array(768);
  for (let i = 0; i < 256; i++) {
    const p = i / 255 * (stops.length - 1), a = stops[Math.floor(p)], b = stops[Math.min(Math.ceil(p), stops.length - 1)], f = p % 1;
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = a[c] + (b[c] - a[c]) * f;
  }
  return lut;
})();
const DIFF_W = 320, DIFF_GAIN = 4;

function closeCompare() {
  if (!C.open) return;
  for (const v of C.vids) { v.pause(); v.removeAttribute('src'); v.load(); }
  C.urls.forEach(u => URL.revokeObjectURL(u));
  C.open = false;
  $('#compare').hidden = true;
  $('#compare').replaceChildren();
  if (S.detail) openDetail(S.detail);   // refresh stars shown underneath
}
function openCompare(g) {
  if (!g) return;
  closeCompare();
  const mem = g.members;
  Object.assign(C, { open: true, mem, frac: 0, playing: false, diff: false, a: 0, b: 1, diffKey: '',
    urls: mem.map(m => URL.createObjectURL(m.file)) });
  C.vids = C.urls.map(u => h('video', { src: u, muted: true, playsInline: true, preload: 'auto' }));
  C.vids[0].addEventListener('ended', () => { if (C.playing) { setFrac(0); C.vids.forEach(v => v.play()); } });

  const head = h('div', { class: 'ohead' },
    h('button', { onclick: closeCompare }, '← Back'),
    h('b', {}, `${mem.length} variants`), h('small', {}, mem[0].folder || '/'),
    h('button', { id: 'cplay', onclick: () => playAll(!C.playing) }, '▶ Play all'),
    h('input', { id: 'scrub', type: 'range', min: 0, max: 1000, value: 0, oninput: e => { playAll(false); setFrac(e.target.value / 1000); } }),
    h('span', { id: 'ctime' }),
    h('label', {}, h('input', { type: 'checkbox', id: 'chover', checked: S.hover, onchange: e => setHover(e.target.checked) }), 'Hover plays'),
    h('label', {}, h('input', { type: 'checkbox', id: 'cdiff', onchange: e => setDiff(e.target.checked) }), 'Difference heatmap (A vs B)'));

  const players = h('div', { class: 'players', id: 'players' }, mem.map((m, i) => h('div', { class: 'player',
    onmouseenter: () => { if (S.hover && !C.playing) C.vids[i].play(); },
    onmouseleave: () => { if (S.hover && !C.playing) { C.vids[i].pause(); setFrac(C.frac); } } },
    C.vids[i], h('div', { class: 'plabel', 'data-i': i }))));
  C.diffCell = h('div', { class: 'player' }, h('canvas', { id: 'diffcv' }), h('div', { class: 'plabel' }, h('span', { class: 'n', id: 'difflabel' })));

  const cols = ['Start', '25%', '50%', '75%', 'End'], fr = [0, ...STRIP, 1];
  const strip = h('div', { class: 'strip' }, h('span'), cols.map((c, k) => h('span', { class: 'hd', title: 'Scrub here', onclick: () => { playAll(false); setFrac(fr[k]); } }, c)),
    mem.map((m, i) => [h('span', { class: 'rn', title: m.path }, `${i + 1}. ${m.name}`),
      fr.map((f, k) => {
        const url = k === 0 ? m.u1 : k === 4 ? m.u2 : m.us?.[k - 1];
        return url ? h('img', { src: url, alt: '', onclick: () => { playAll(false); setFrac(f); } }) : h('div', { class: 'wait', 'data-cell': `${i}-${k}` });
      })]));

  $('#compare').replaceChildren(head, h('div', { class: 'obody' }, players, h('h3', {}, 'Sampled frames ', h('small', {}, 'click a frame or column to scrub there')), strip));
  $('#compare').hidden = false;
  layoutPlayers(); renderLabels(); setFrac(0);
  requestAnimationFrame(tick);
  loadStrips(mem);
}
async function loadStrips(mem) {
  for (const [i, m] of mem.entries()) {
    if (!m.us) {
      try { await ensureStrip(m); } catch (e) { console.warn('strip failed', m.path, e); continue; }
      m.us = m.rec.strip.map(b => URL.createObjectURL(b));
    }
    if (C.mem !== mem || !C.open) return;
    m.us.forEach((u, k) => $(`[data-cell="${i}-${k + 1}"]`)?.replaceWith(
      h('img', { src: u, alt: '', onclick: () => { playAll(false); setFrac(STRIP[k]); } })));
  }
}
function layoutPlayers() {
  const n = C.mem.length + (C.diff ? 1 : 0);
  $('#players').style.gridTemplateColumns = `repeat(${Math.min(n, 5)}, 1fr)`;
}
function renderLabels() {
  for (const lab of document.querySelectorAll('#players .plabel[data-i]')) {
    const i = +lab.dataset.i, m = C.mem[i], star = meta(m).star;
    lab.replaceChildren(h('span', { class: 'n', title: m.path }, `${i + 1}. ${m.name} · ${secs(m.rec.dur)}`),
      C.diff && h('button', { class: C.a === i ? 'on' : '', title: 'Use as A in the heatmap', onclick: () => { C.a = i; renderLabels(); } }, 'A'),
      C.diff && h('button', { class: C.b === i ? 'on' : '', title: 'Use as B in the heatmap', onclick: () => { C.b = i; renderLabels(); } }, 'B'),
      h('button', { class: 'starbtn' + (star ? ' on' : ''), title: `Star (${i + 1})`, onclick: () => { toggleStar(m); renderLabels(); } }, star ? '★' : '☆'));
  }
  C.diffKey = '';
}
function setFrac(f) {
  C.frac = Math.min(Math.max(f, 0), 1);
  for (const v of C.vids) {
    if (v.readyState >= 1) v.currentTime = C.frac * endTime(v);
    else v.addEventListener('loadedmetadata', () => { v.currentTime = C.frac * endTime(v); }, { once: true });
  }
  showFrac();
}
function showFrac() {
  $('#scrub').value = C.frac * 1000;
  const d = C.vids[0].duration;
  $('#ctime').textContent = `${pct(C.frac)}${isFinite(d) ? ' · ' + (C.frac * d).toFixed(2) + 's' : ''}`;
}
function playAll(on) {
  if (on === C.playing) return;
  C.playing = on;
  $('#cplay').textContent = on ? '❚❚ Pause' : '▶ Play all';
  if (on) { if (C.frac >= 0.999) C.frac = 0; setFrac(C.frac); C.vids.forEach(v => v.play()); }
  else { C.vids.forEach(v => v.pause()); const v = C.vids[0]; if (v.duration) C.frac = v.currentTime / v.duration; setFrac(C.frac); }
}
function setHover(on) {
  S.hover = on;
  $('#hover').checked = on;
  if ($('#chover')) $('#chover').checked = on;
  DB.set('kv', 'hover', on);
  if (!on) stopHover();
}
function setDiff(on) {
  C.diff = on;
  $('#cdiff').checked = on;
  on ? $('#players').append(C.diffCell) : C.diffCell.remove();
  layoutPlayers(); renderLabels();
}
function drawDiff() {
  const a = C.vids[C.a], b = C.vids[C.b], cv = $('#diffcv');
  if (!a || !b || a.readyState < 2 || b.readyState < 2 || a.seeking || b.seeking) return;
  const key = `${C.a}:${a.currentTime}|${C.b}:${b.currentTime}`;
  if (key === C.diffKey) return;
  C.diffKey = key;
  const w = DIFF_W, hh = Math.round(w * a.videoHeight / a.videoWidth);
  if (cv.width !== w || cv.height !== hh) { cv.width = w; cv.height = hh; }
  const cx = cv.getContext('2d', { willReadFrequently: true });
  cx.drawImage(a, 0, 0, w, hh); const A = cx.getImageData(0, 0, w, hh).data;
  cx.drawImage(b, 0, 0, w, hh); const img = cx.getImageData(0, 0, w, hh), B = img.data;
  let sum = 0;
  for (let i = 0; i < B.length; i += 4) {
    const d = (Math.abs(A[i] - B[i]) + Math.abs(A[i + 1] - B[i + 1]) + Math.abs(A[i + 2] - B[i + 2])) / 3;
    sum += d;
    const l = Math.min(255, Math.round(d * DIFF_GAIN)) * 3;
    B[i] = LUT[l]; B[i + 1] = LUT[l + 1]; B[i + 2] = LUT[l + 2]; B[i + 3] = 255;
  }
  cx.putImageData(img, 0, 0);
  $('#difflabel').textContent = C.a === C.b ? 'Pick two different variants (A / B)'
    : `Difference ${C.a + 1} vs ${C.b + 1} · mean ${(sum / (B.length / 4) / 2.55).toFixed(1)}%`;
}
function tick() {
  if (!C.open) return;
  const lead = C.vids[0];
  if (C.playing && lead.duration) { C.frac = lead.currentTime / lead.duration; showFrac(); }
  if (C.diff) drawDiff();
  requestAnimationFrame(tick);
}

// ---------- harvest: scrub every clip, cut frames into key-frame collections ----------
// Collections are folders of PNGs under <opened folder>/keyframes, so any other tool can use them.
// keyframes.json beside them records each frame's source clip and which frame is the collection's reference.
const FILM_N = 12, FILM_W = 200;               // filmstrip cells per clip, cell width
const KDIR = 'keyframes', KMAN = 'keyframes.json';
const SDIR = 'shot-frames';                    // one frame per shot, exported for other tools
const CDIR = 'shot-clips';                     // shots saved as clips of their own
const VIEWS = ['library', 'shots', 'groups', 'joins', 'harvest', 'board'];
const K = { cols: [], clips: [], active: 'unsorted', it: null, url: null, want: 0, lock: null, mx: 0, my: 0, busy: false,
  pad: 0, saved: new Set(), encQ: Promise.resolve(), encN: 0 };
K.v = h('video', { muted: true, playsInline: true, preload: 'auto', loop: true });
K.head = h('div', { class: 'hhead' });
K.at = h('span', { class: 'kind at' });   // frame readout over a shot tile while it is scrubbed
K.info = h('div', { class: 'hinfo' }, 'No clip yet.');
K.colsEl = h('div', {});
K.joinEl = h('div', { class: 'hjoins' });
K.list = h('div', { class: 'hlist' });
K.side = h('aside', { class: 'hside' },
  h('div', { class: 'hprev' }, K.v, K.info,
    h('small', {}, 'Hover a strip to scrub. Click or Enter cuts the frame; ← → step one frame; 1–9 cut into that collection. Drop image files on a collection to add them.')),
  K.joinEl, K.colsEl);

function filmCanvas(v) {
  const ch = Math.max(1, Math.round(FILM_W * v.videoHeight / v.videoWidth));
  const cv = new OffscreenCanvas(FILM_N * FILM_W, ch), cx = cv.getContext('2d');
  cx.imageSmoothingQuality = 'high';
  return { draw: (src, i) => cx.drawImage(src, i * FILM_W, 0, FILM_W, ch), blob: () => cv.convertToBlob({ type: 'image/jpeg', quality: 0.8 }) };
}
// One sprite of evenly spaced frames per clip, cached. Indexing draws it too; this is the quick path for a row on screen.
function ensureFilm(it) {
  return it.filmP ||= (async () => {
    const r = it.rec;
    if (!r.film) {
      const { v, done } = await openVideo(it.file);
      try {
        if (!isFinite(v.duration)) throw new Error('no duration');
        const fd = await frameStep(v), film = filmCanvas(v);
        for (let i = 0; i < FILM_N; i++) {
          await seek(v, (i + 0.5) / FILM_N * v.duration);
          film.draw(v, i);
        }
        const blob = await film.blob();
        if (!r.film) { r.film = blob; r.fd ??= fd; DB.set('cache', it.key, r); }
      } finally { done(); }
    }
    it.uf ||= URL.createObjectURL(r.film);
  })();
}
const filmQ = [];
let filmBusy = 0;
function pumpFilm() {
  while (filmBusy < 2 && filmQ.length) {
    const it = filmQ.pop();   // newest first: what just scrolled into view
    filmBusy++;
    ensureFilm(it).then(() => {
      const s = rowOf(it)?.querySelector('.hstrip');
      if (s && !s.querySelector('img')) s.prepend(h('img', { src: it.uf, alt: '', draggable: false }));
    }, e => console.warn('filmstrip failed', it.path, e)).finally(() => { filmBusy--; pumpFilm(); });
  }
}
const filmIO = new IntersectionObserver(es => {
  for (const e of es) {
    const it = e.isIntersecting && S.items.find(x => x.id === e.target.dataset.id);
    if (it && !it.uf && !it.filmQ) { it.filmQ = true; filmQ.push(it); }
  }
  pumpFilm();
}, { root: K.list, rootMargin: '300px' });

const fdur = it => it.rec.fd || 1 / 30;
const nframes = it => Math.max(1, Math.round(it.rec.dur / fdur(it)));
const frameAt = (it, t) => Math.min(nframes(it) - 1, Math.max(0, Math.floor(t / fdur(it) + 1e-3)));
const frameTime = (it, k) => (k + 0.5) * fdur(it);   // the middle of a frame, so the seek cannot land on its neighbour
const rowOf = it => K.list.querySelector(`.hrow[data-id="${CSS.escape(it.id)}"]`);
const holdMouse = () => { K.lock = { x: K.mx, y: K.my }; };   // keep a keyboard-chosen frame until the mouse really moves

function showAt(it, t) {
  if (K.it !== it) {
    if (K.url) URL.revokeObjectURL(K.url);
    K.it = it;
    K.v.src = K.url = URL.createObjectURL(it.file);
    K.v.classList.add('wait');   // a shot tile keeps its own frame until this clip's lands
    K.list.querySelector('.hrow.on')?.classList.remove('on');
    const row = rowOf(it);
    row?.classList.add('on');
    row ? row.querySelector('.hstrip').append(K.head) : K.head.remove();
    renderJoinSide();
  }
  K.v.pause();
  K.want = frameTime(it, frameAt(it, t));
  pumpSeek();
}
// One seek at a time; the next starts when the last lands, so fast scrubbing never queues up.
function pumpSeek() {
  const v = K.v;
  if (v.readyState < 1 || v.seeking || !v.paused) return;
  if (Math.abs(v.currentTime - K.want) > 1e-3) v.currentTime = K.want;
}
function showHead() {
  const it = K.it;
  if (!it) return;
  const t = K.v.currentTime;
  K.head.style.left = (100 * t / it.rec.dur).toFixed(2) + '%';
  K.info.textContent = `${it.name} · frame ${frameAt(it, t)} of ${nframes(it) - 1} · ${t.toFixed(2)}s`;
  K.at.textContent = `frame ${frameAt(it, t)} · ${t.toFixed(2)}s`;
}
K.v.addEventListener('loadedmetadata', pumpSeek);
K.v.addEventListener('seeked', () => { K.v.classList.remove('wait'); pumpSeek(); showHead(); if (Q.open) armDrag(); });
K.v.addEventListener('timeupdate', showHead);

function nudge(d) {
  const it = K.it;
  if (!it) return;
  showAt(it, frameTime(it, frameAt(it, K.v.paused ? K.want : K.v.currentTime) + d));
  holdMouse();
}
function stepClip(d) {
  const i = K.clips.indexOf(K.it), it = K.clips[i < 0 ? 0 : i + d];
  if (!it) return;
  showAt(it, (K.it ? K.want / K.it.rec.dur : 0) * it.rec.dur);
  holdMouse();
  rowOf(it)?.scrollIntoView({ block: 'nearest' });
}
function togglePlay() {
  const v = K.v;
  if (!K.it) return;
  if (v.paused) v.play().catch(() => {});
  else { v.pause(); K.want = v.currentTime; }
}

// ----- collections on disk -----
const cleanName = s => s.trim().replace(/[\/\\:]/g, '-').replace(/^\.+/, '');
const getCol = name => K.cols.find(c => c.name === name) || (K.cols.push({ name, frames: [], canon: null }), K.cols.at(-1));
async function ensureWrite() {
  if (S.dir && await S.dir.requestPermission({ mode: 'readwrite' }) !== 'granted') throw new Error('write access to the folder was not granted');
}
async function kdir(col, create) {
  const d = await S.dir.getDirectoryHandle(KDIR, { create });
  return col ? d.getDirectoryHandle(col, { create }) : d;
}
async function writeFile(dir, name, data) {
  const w = await (await dir.getFileHandle(name, { create: true })).createWritable();
  await w.write(data);
  await w.close();
}
function download(blob, name) {   // no folder access (fallback picker): the browser saves the frame instead
  const a = h('a', { href: URL.createObjectURL(blob), download: name });
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
async function saveManifest() {
  if (!S.dir) return;
  const collections = {};
  for (const c of K.cols) collections[c.name] = { canonical: c.canon || null,
    frames: Object.fromEntries(c.frames.map(f => [f.name, { clip: f.clip, frame: f.frame, time: f.time, from: f.from }])) };
  await writeFile(await kdir(null, true), KMAN, JSON.stringify({ version: 1, collections }, null, 2));
}
// The folders are the truth, so frames added or removed outside this page show up; the manifest only annotates them.
async function loadCollections() {
  for (const c of K.cols) for (const f of c.frames) URL.revokeObjectURL(f.url);
  K.cols = [];
  if (S.dir) try {
    const root = await kdir(null, false);
    let man = {};
    try { man = JSON.parse(await (await (await root.getFileHandle(KMAN)).getFile()).text()).collections || {}; } catch {}
    for await (const [name, d] of root.entries()) {
      if (d.kind !== 'directory' || name.startsWith('.')) continue;
      const frames = [];
      for await (const [fn, fh] of d.entries()) {
        if (fh.kind !== 'file' || !IMG.test(fn)) continue;
        const file = await fh.getFile();
        frames.push({ name: fn, url: URL.createObjectURL(file), mtime: file.lastModified, ...man[name]?.frames?.[fn] });
      }
      frames.sort((a, b) => a.mtime - b.mtime);
      const canon = man[name]?.canonical;
      K.cols.push({ name, frames, canon: frames.some(f => f.name === canon) ? canon : frames[0]?.name || null });
    }
  } catch (e) { if (e.name !== 'NotFoundError') console.warn('key frames not loaded', e); }
  K.cols.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  if (!K.cols.some(c => c.name === K.active) && K.cols.length) K.active = K.cols[0].name;
  renderSide();
  if (S.view === 'harvest') renderHarvest();
}
function setActive(name) {
  K.active = name;
  DB.set('kv', 'kcol', name);
  renderSide();
}
async function addCollection(raw) {
  const name = cleanName(raw);
  if (!name) return;
  getCol(name);
  setActive(name);
  try { await ensureWrite(); if (S.dir) await kdir(name, true); }
  catch (e) { say(`Could not create the collection folder: ${e.message}`); }
}
function say(text) {   // the status line is hidden behind an open clip, so also show a toast
  status(text);
  const t = $('#toast');
  t.textContent = text;
  t.hidden = false;
  clearTimeout(say.timer);
  say.timer = setTimeout(() => { t.hidden = true; }, 3500);
}
function fillColSelects() {
  for (const sel of document.querySelectorAll('select.kcolsel')) {
    sel.replaceChildren(...K.cols.map(c => h('option', { value: c.name }, `${c.name} (${c.frames.length})`)), h('option', { value: '\0new' }, 'New collection…'));
    sel.value = K.active;
    sel.onchange = () => {
      if (sel.value !== '\0new') return setActive(sel.value);
      const name = prompt('Name of the new collection');
      name && cleanName(name) ? addCollection(name) : fillColSelects();
    };
  }
}

// Put an image into a collection: on disk, in the manifest, and in the library as a still so it takes part in matching.
async function saveFrame(col, name, blob, info) {
  if (S.dir) await writeFile(await kdir(col.name, true), name, blob);
  for (const f of col.frames) if (f.name === name) URL.revokeObjectURL(f.url);
  col.frames = [...col.frames.filter(f => f.name !== name), { name, url: URL.createObjectURL(blob), mtime: Date.now(), ...info }];
  col.canon ||= name;
  K.active = col.name;
  DB.set('kv', 'kcol', col.name);
  await saveManifest();
  if (S.dir) {
    const path = `${KDIR}/${col.name}/${name}`, id = S.scanId;
    const it = await makeItem(path, async () => new File([blob], name, { type: blob.type, lastModified: Date.now() }));
    if (id === S.scanId) S.items = [...S.items.filter(x => x.path !== path), it];
  }
  refresh();
  renderSide();
}
async function addStill(it, colName = K.active) {
  const col = getCol(colName);
  if (it.path.startsWith(`${KDIR}/${col.name}/`)) return say(`${it.name} is already in “${col.name}”`);
  try {
    await ensureWrite();
    let name = it.name;   // another still of the same name keeps its place
    for (let n = 2; col.frames.some(f => f.name === name && f.from !== it.path); n++) name = it.name.replace(/(\.[^.]+)?$/, `-${n}$1`);
    await saveFrame(col, name, it.file, { from: it.path });
    say(`Added ${it.name} to “${col.name}”`);
  } catch (e) { say(`Could not add the still: ${e.message}`); }
}
async function dropFiles(col, files) {
  const imgs = [...files].filter(f => IMG.test(f.name));
  if (!imgs.length) return say('Only image files can be dropped on a collection.');
  try {
    await ensureWrite();
    for (const f of imgs) await saveFrame(col, f.name, f, { from: 'dropped file' });
    say(`Added ${imgs.length} still${imgs.length > 1 ? 's' : ''} to “${col.name}”`);
  } catch (e) { say(`Could not add: ${e.message}`); }
}
// A: the selected or open still goes into the collection; in an open clip, the frame on screen does.
function collect(it) {
  if (!it?.rec) return;
  if (it.kind === 'image') return addStill(it);
  const v = S.detail === it && $('#dplay');
  if (v) return cutFrame(it, v, v.currentTime);
  say('Open the clip and press A on the frame you want, or pick frames in Harvest.');
}

// Cut one frame of a clip into a collection, at the clip's full resolution. `v` is a player showing that clip.
async function cutFrame(it, v, at, colName = K.active) {
  if (K.busy) return;
  K.busy = true;
  const src = v.currentSrc;
  try {
    v.pause();
    await ensureWrite();   // first, while the click still counts as a user gesture
    await ensureFilm(it);  // the frame duration comes with the filmstrip
    const k = frameAt(it, at), t = frameTime(it, k);
    if (v === K.v) K.want = t;
    if (v.readyState < 2 && !v.seeking) await once(v, 'loadeddata');   // a seek in flight also lowers readyState; the loop waits for that
    for (let i = 0; i < 20 && (v.seeking || Math.abs(v.currentTime - t) > 1e-3); i++) {
      if (!v.seeking) v.currentTime = t;
      await once(v, 'seeked');
    }
    if (v.currentSrc !== src || Math.abs(v.currentTime - t) > 1e-3) return;   // moved on while the frame was loading
    const cv = new OffscreenCanvas(v.videoWidth, v.videoHeight);
    cv.getContext('2d').drawImage(v, 0, 0);
    const blob = await cv.convertToBlob({ type: 'image/png' });
    const name = `${it.name.replace(/\.[^.]+$/, '')}_f${String(k).padStart(5, '0')}.png`;
    const col = getCol(colName);
    if (!S.dir) download(blob, `${col.name}__${name}`);
    await saveFrame(col, name, blob, { clip: it.path, frame: k, time: t });
    say(`Cut frame ${k} of ${it.name} into “${col.name}”${S.dir ? ` → ${KDIR}/${col.name}/${name}` : ' (downloaded)'}`);
  } catch (e) { say(`Could not cut the frame: ${e.message}`); }
  finally { K.busy = false; }
}
function cut(colName) {
  const it = K.it, v = K.v;
  if (!it) return;
  if (!v.paused) { v.pause(); K.want = v.currentTime; }
  return cutFrame(it, v, K.want, colName);
}
// A shot tile cuts the frame it is showing: the scrubbed one under the mouse, otherwise the shot's typical frame.
function cutShot(id) {
  const x = shotOf(id);
  if (!x) return;
  if (K.it !== x.it || K.v.closest('.tile')?.dataset.id !== id) showAt(x.it, x.s.tk);
  return cut();
}
// One full-size frame per shot into a plain folder, for tools that sort or edit images: the picked frames where there are any.
async function exportShots() {
  if (K.busy) return;
  K.busy = true;
  try {
    await ensureWrite();
    const dir = await S.dir.getDirectoryHandle(SDIR, { create: true }), old = [];
    for await (const [name, fh] of dir.entries()) if (fh.kind === 'file' && /_f\d{5}\.png$/.test(name)) old.push(name);
    for (const name of old) await dir.removeEntry(name);   // shots move when a clip is re-indexed; only this export's own files go
    const clips = S.list.filter(it => segs(it).length), cuts = cutsByClip();
    let n = 0, done = 0;
    for (const it of clips) {
      const { v, done: close } = await openVideo(it.file);
      try {
        const cv = new OffscreenCanvas(v.videoWidth, v.videoHeight), cx = cv.getContext('2d');
        const stem = it.path.replace(/\.[^.]+$/, '').replace(/\//g, '__');
        for (const s of segs(it)) {
          const picked = [...new Set(picksIn(cuts.get(it.path) || [], s).map(f => f.frame))];
          for (const k of picked.length ? picked : [frameAt(it, s.tk)]) {
            await seek(v, frameTime(it, k));
            cx.drawImage(v, 0, 0);
            await writeFile(dir, `${stem}_f${String(k).padStart(5, '0')}.png`, await cv.convertToBlob({ type: 'image/png' }));
            n++;
          }
        }
      } finally { close(); }
      status(`Saving shot frames ${++done} / ${clips.length}`, done, clips.length);
    }
    say(`Saved ${n} frames to ${SDIR}/: your picks, and the typical frame of each shot without one`);
  } catch (e) { say(`Could not save the shot frames: ${e.message}`); }
  finally { K.busy = false; }
}

// ----- frame sheet: every frame of one shot laid out as its own thumbnail, to pick from -----
const Q = { open: false, id: 0, cache: new Map(), chain: Promise.resolve(), mx: 0, my: 0, fulls: [] };
Q.info = h('div', { class: 'hinfo' });
const shotList = () => S.list.filter(it => segs(it).length).flatMap(it => segs(it).map((s, i) => ({ it, i })));
function closeSheet() {
  if (!Q.open) return;
  Q.open = false;
  Q.id++;
  K.v.pause();
  K.v.remove();
  if (Q.cutV) { Q.cutV.removeAttribute('src'); Q.cutV.load(); URL.revokeObjectURL(Q.url); Q.cutV = null; }
  clearTimeout(Q.armT);
  for (const f of Q.fulls.splice(0)) f.own && URL.revokeObjectURL(f.url);
  $('#sheet').hidden = true;
  $('#sheet').replaceChildren();
}
function openSheet(it, i) {
  const s = segs(it)[i];
  if (!s) return;
  closeSheet();
  const fd = fdur(it), ka = Math.min(nframes(it) - 1, Math.round(s.t0 / fd)), kb = Math.max(ka, Math.min(nframes(it), Math.round(s.t1 / fd)) - 1);
  Object.assign(Q, { open: true, it, i, ka, kb, sel: -1, id: Q.id + 1, url: URL.createObjectURL(it.file) });
  Q.cutV = h('video', { src: Q.url, muted: true, playsInline: true, preload: 'auto' });   // picks are cut here, so the preview stays free
  Q.grid = h('div', { class: 'qgrid',
    onmousemove: e => {   // only a real movement: scrolling under a still mouse must not steal the keyboard's frame
      if (e.clientX === Q.mx && e.clientY === Q.my) return;
      Q.mx = e.clientX; Q.my = e.clientY;
      const c = e.target.closest('.qf');
      if (c) selFrame(+c.dataset.k, false);
    },
    onclick: e => {
      const c = e.target.closest('.qf');
      if (!c) return;
      if (e.target.closest('.qdel')) return unpick(+c.dataset.k);
      selFrame(+c.dataset.k, false);
      pickFrame(+c.dataset.k);
    } },
    Array.from({ length: kb - ka + 1 }, (_, j) => h('div', { class: 'qf', 'data-k': ka + j, style: `aspect-ratio:${it.rec.w}/${it.rec.h}` },
      h('span', { class: 'qn' }, String(ka + j)))));
  const list = shotList(), at = list.findIndex(x => x.it === it && x.i === i), go = d => { const x = list[at + d]; if (x) openSheet(x.it, x.i); };
  Q.go = go;
  $('#sheet').replaceChildren(
    h('div', { class: 'ohead' },
      h('button', { onclick: closeSheet }, '← Back'),
      h('span', { class: 'path', title: it.path }, it.name),
      h('small', {}, `shot ${i + 1} of ${segs(it).length} · frames ${ka}–${kb} · ${s.t0.toFixed(2)}–${s.t1.toFixed(2)}s`),
      h('button', { disabled: at <= 0, title: 'Previous shot ( [ )', onclick: () => go(-1) }, '◀ Shot'),
      h('button', { disabled: at < 0 || at >= list.length - 1, title: 'Next shot ( ] )', onclick: () => go(1) }, 'Shot ▶'),
      h('span', { class: 'grow' }),
      h('label', {}, 'Pick into ', h('select', { class: 'kcolsel' }))),
    h('div', { class: 'qbody' }, Q.grid,
      h('aside', { class: 'qside' }, K.v, Q.info,
        h('p', {}, Q.playBtn = h('button', { title: 'Play this shot on a loop; the frame on screen is highlighted. Pausing, or pointing at a frame, stops on that frame.',
          onclick: () => playSheet() }, '▶ Play shot (Space)'), ' ',
          h('button', { title: 'Copy this frame, full size, to the clipboard (C)', onclick: () => copyFrame() }, 'Copy frame (C)'),
          h('small', {}, ' then paste it anywhere, or drag the frame’s tile straight into another tab or app.')),
        h('p', {}, h('button', { title: `Save the frames that are not dimmed as their own clip in “${CDIR}”`, onclick: () => saveClip([it, s]) }, '▶ Save as clip (V)'), ' ',
          h('button', { title: 'Start the clip at this frame (I)', onclick: () => setTrim('a') }, 'Start here (I)'), ' ',
          h('button', { title: 'End the clip at this frame (O)', onclick: () => setTrim('b') }, 'End here (O)'), ' ',
          Q.clipInfo = h('small', {})),
        h('small', {}, 'Each tile is one frame. Click a frame (or Enter) to pick it: it is saved full size into the collection and becomes the frame shown for this shot. '
          + 'Space plays the shot, ← → ↑ ↓ move between frames, 1–9 pick into that collection, [ ] change shot, × unpicks. '
          + 'I and O set where the saved clip starts and ends, for a cut that smears into the next scene; V saves it.'))));
  $('#sheet').hidden = false;
  fillColSelects();
  markSheet();
  markTrim();
  const mine = picksIn(cutsByClip().get(it.path) || [], s);
  selFrame(mine.length ? mine.at(-1).frame : frameAt(it, s.tk));
  fillSheet();
}
// Thumbnails come from playing the shot once; frames the player skipped are then fetched one by one.
async function fillSheet() {
  const { it, ka, kb, id } = Q, key = `${it.key}|${ka}|${kb}`;
  let cvs = Q.cache.get(key);
  Q.cache.delete(key);
  Q.cache.set(key, cvs ||= new Map());
  if (Q.cache.size > 6) Q.cache.delete(Q.cache.keys().next().value);   // the shots opened longest ago go first
  const put = k => Q.grid.querySelector(`[data-k="${k}"]`)?.prepend(cvs.get(k));
  for (const k of cvs.keys()) put(k);
  if (cvs.size > kb - ka) return;
  const { v, done } = await openVideo(it.file);
  try {
    const w = SEG_W, hh = Math.max(1, Math.round(w * v.videoHeight / v.videoWidth));
    const take = k => {
      if (cvs.has(k) || k < ka || k > kb) return;
      const cv = h('canvas', { width: w, height: hh });
      cv.getContext('2d').drawImage(v, 0, 0, w, hh);
      cvs.set(k, cv);
      if (Q.id === id) put(k);
    };
    await playThrough(v, fdur(it), nframes(it), k => Q.id === id && take(k), ka, kb, 2);
    for (let k = ka; k <= kb && Q.id === id; k++) if (!cvs.has(k)) { await seek(v, frameTime(it, k)); take(k); }
  } catch (e) { console.warn('frame sheet failed', it.path, e); }
  finally { done(); }
}
function markSheet() {   // star the frames already picked from this clip
  if (!Q.open) return;
  const mine = new Map((cutsByClip().get(Q.it.path) || []).map(f => [f.frame, f]));
  for (const c of Q.grid.children) {
    const f = mine.get(+c.dataset.k);
    c.classList.toggle('picked', !!f);
    c.classList.remove('busy');
    c.querySelector('.qdel')?.remove();
    if (f) c.append(h('button', { class: 'qdel', title: `Picked into “${f.col.name}” · click to unpick` }, '×'));
  }
}
function markSel(k, scroll = true) {
  const k2 = Math.min(Q.kb, Math.max(Q.ka, k));
  if (k2 === Q.sel) return k2;
  Q.grid.querySelector('.qf.on')?.classList.remove('on');
  Q.sel = k2;
  const c = Q.grid.querySelector(`[data-k="${k2}"]`);
  c?.classList.add('on');
  if (scroll) c?.scrollIntoView({ block: 'nearest' });
  Q.info.textContent = `frame ${k2} · ${(k2 * fdur(Q.it)).toFixed(2)}s · ${k2 - Q.ka + 1} of ${Q.kb - Q.ka + 1} in this shot`;
  return k2;
}
function selFrame(k, scroll = true) {   // also stops playback: pointing at a frame holds the preview on it
  showAt(Q.it, frameTime(Q.it, markSel(k, scroll)));
  armDrag();
}
// Play the shot in the preview, looping inside it; the tile of the frame on screen is highlighted as it goes.
function playSheet(on = K.v.paused) {
  const v = K.v, it = Q.it, id = Q.id, fd = fdur(it), a = frameTime(it, Q.ka), end = (Q.kb + 1) * fd;
  if (!on) return selFrame(frameAt(it, v.currentTime));   // stop on the frame that was showing
  if (Q.sel >= Q.kb) v.currentTime = a;
  const tick = (_, m) => {
    if (!Q.open || Q.id !== id || v.paused) return;
    if (m.mediaTime >= end - 1e-3 || m.mediaTime < Q.ka * fd - 1e-3) v.currentTime = a;   // ran off the shot, or the clip wrapped round
    else markSel(Math.round(m.mediaTime / fd));
    v.requestVideoFrameCallback(tick);
  };
  v.requestVideoFrameCallback(tick);
  v.play().catch(() => {});
}
K.v.addEventListener('play', () => { if (Q.open) Q.playBtn.textContent = '❚❚ Pause (Space)'; });
K.v.addEventListener('pause', () => { if (Q.open) Q.playBtn.textContent = '▶ Play shot (Space)'; });
// The frame being pointed at is put in its tile as a full-size image, so the tile can be dragged out of the page as that image.
function armDrag() {
  clearTimeout(Q.armT);
  const { id, sel: k, it } = Q, here = () => Q.open && Q.id === id && Q.sel === k;
  Q.armT = setTimeout(async () => {
    const cell = here() && Q.grid.querySelector(`[data-k="${k}"]`), v = K.v;
    if (!cell || !v.paused || cell.querySelector('.qfull')) return;
    const picked = (cutsByClip().get(it.path) || []).find(f => f.frame === k);
    let url = picked?.url;
    if (!url) {
      if (v.seeking || v.readyState < 2 || Math.abs(v.currentTime - frameTime(it, k)) > 1e-3) return;   // the preview's `seeked` comes back here
      const cv = new OffscreenCanvas(v.videoWidth, v.videoHeight);
      cv.getContext('2d').drawImage(v, 0, 0);
      const blob = await cv.convertToBlob({ type: 'image/png' });
      if (!here() || cell.querySelector('.qfull')) return;
      url = URL.createObjectURL(blob);
    }
    const img = h('img', { class: 'qfull', src: url, alt: `frame ${k} of ${it.name}`, title: 'Drag this frame into another tab or app · click picks it' });
    cell.append(img);
    Q.fulls.push({ img, url, own: !picked });
    while (Q.fulls.length > 8) { const f = Q.fulls.shift(); f.img.remove(); f.own && URL.revokeObjectURL(f.url); }   // full-size frames are large
  }, 120);
}
async function sheetBlob(k) {   // frame k of the sheet's clip at full size, from the sheet's own player
  const v = Q.cutV;
  if (v.readyState < 1) await once(v, 'loadedmetadata');
  await seek(v, frameTime(Q.it, k));
  const cv = new OffscreenCanvas(v.videoWidth, v.videoHeight);
  cv.getContext('2d').drawImage(v, 0, 0);
  return cv.convertToBlob({ type: 'image/png' });
}
function copyFrame(k = Q.sel) {
  const p = Q.chain.then(() => sheetBlob(k));
  Q.chain = p.catch(() => {});
  navigator.clipboard.write([new ClipboardItem({ 'image/png': p })])
    .then(() => say(`Copied frame ${k} of ${Q.it?.name}. Paste it with ⌘V.`), e => say(`Could not copy the frame: ${e.message}`));
}
function pickFrame(k = Q.sel, colName = K.active) {
  const { it, id, cutV } = Q;
  Q.grid.querySelector(`[data-k="${k}"]`)?.classList.add('busy');
  // one after another, so several quick clicks all land
  Q.chain = Q.chain.then(() => Q.id === id && cutFrame(it, cutV, frameTime(it, k), colName)).then(markSheet);
}
function unpick(k) {
  const f = (cutsByClip().get(Q.it.path) || []).find(f => f.frame === k);
  if (f) delFrame(f.col, f.col.frames.find(x => x.name === f.name)).then(markSheet);
}
function sheetKey(k, e) {
  const cells = [...Q.grid.children], cols = cells.filter(c => c.offsetTop === cells[0].offsetTop).length;
  if (k === 'ArrowLeft' || k === 'ArrowRight') selFrame(Q.sel + (k === 'ArrowLeft' ? -1 : 1) * (e.shiftKey ? 10 : 1));
  else if (k === 'ArrowUp') selFrame(Q.sel - cols);
  else if (k === 'ArrowDown') selFrame(Q.sel + cols);
  else if (k === ' ') playSheet();
  else if (k === 'Enter' || k === 'a') { if (!K.v.paused) playSheet(false); pickFrame(); }
  else if (k >= '1' && k <= '9') { if (K.cols[k - 1]) pickFrame(Q.sel, K.cols[k - 1].name); }
  else if (k === 'c') copyFrame();
  else if (k === 'i') setTrim('a');
  else if (k === 'o') setTrim('b');
  else if (k === 'v') saveClip([Q.it, segs(Q.it)[Q.i]]);
  else if (k === '[' || k === 'PageUp') Q.go(-1);
  else if (k === ']' || k === 'PageDown') Q.go(1);
  else return false;
  return true;
}
// ----- sub-clips: a shot, or a hand-set part of it, saved as a clip of its own -----
// Frames of the shot kept in its clip: set by hand with I and O in the frame sheet, otherwise the shot less the trim at each end.
function clipRange(it, s) {
  const a0 = frameAt(it, s.t0), b0 = Math.max(a0, frameAt(it, s.t1 - fdur(it) / 2)), tr = it.trims?.get(s.t0) || {};
  let a = tr.a ?? a0 + K.pad, b = tr.b ?? b0 - K.pad;
  if (b < a) { a = tr.a ?? a0; b = Math.max(a, tr.b ?? b0); }   // the trim would leave nothing
  return { a, b, a0, b0 };
}
const pad5 = k => String(k).padStart(5, '0');
const clipName = (it, g) => `${it.path.replace(/\.[^.]+$/, '').replace(/\//g, '__')}_f${pad5(g.a)}-${pad5(g.b)}.mp4`;
function setTrim(end, k = Q.sel) {   // I / O in the frame sheet: this frame becomes the clip's first / last
  const it = Q.it, s = segs(it)[Q.i], g = clipRange(it, s), tr = { a: g.a, b: g.b, [end]: k };
  if (tr.b < tr.a) tr[end === 'a' ? 'b' : 'a'] = end === 'a' ? g.b0 : g.a0;
  (it.trims ||= new Map()).set(s.t0, tr);
  markTrim();
}
function markTrim() {   // dim the frames the clip leaves out
  if (!Q.open) return;
  const it = Q.it, g = clipRange(it, segs(it)[Q.i]), fd = fdur(it);
  for (const c of Q.grid.children) c.classList.toggle('out', +c.dataset.k < g.a || +c.dataset.k > g.b);
  Q.clipInfo.textContent = `Clip: frames ${g.a}–${g.b} · ${((g.b - g.a + 1) * fd).toFixed(2)}s` + (g.a !== g.a0 || g.b !== g.b0 ? ` of the shot's ${((g.b0 - g.a0 + 1) * fd).toFixed(2)}s` : ' · the whole shot');
}
// Frames k0..k1 of a clip re-encoded as H.264, since a cut rarely falls on a key frame. Picture only.
async function encodeRange(it, k0, k1, onFrame) {
  if (!window.VideoEncoder) throw new Error('this browser cannot encode video');
  const { v, done } = await openVideo(it.file);
  let enc;
  try {
    const fd = fdur(it), w = v.videoWidth & ~1, hh = v.videoHeight & ~1, us = k => Math.round(k * fd * 1e6);
    const base = { width: w, height: hh, framerate: 1 / fd, bitrate: Math.round(Math.max(2e6, w * hh / fd * 0.2)) };
    let cfg;
    for (const codec of ['avc1.640034', 'avc1.640033', 'avc1.64002A', 'avc1.4D4028', 'avc1.42E01F'])
      if ((await VideoEncoder.isConfigSupported({ ...base, codec })).supported) { cfg = { ...base, codec }; break; }
    if (!cfg) throw new Error(`this browser cannot encode ${w}×${hh} H.264`);
    const mux = new Mp4Muxer.Muxer({ target: new Mp4Muxer.ArrayBufferTarget(), fastStart: 'in-memory',
      video: { codec: 'avc', width: w, height: hh, frameRate: Math.round(1 / fd) } });
    let err;
    enc = new VideoEncoder({ output: (c, m) => mux.addVideoChunk(c, m), error: e => { err = e; } });
    enc.configure(cfg);
    const cv = w === v.videoWidth && hh === v.videoHeight ? null : new OffscreenCanvas(w, hh);   // odd sizes are cropped to even
    for (let k = k0; k <= k1; k++) {
      await seek(v, frameTime(it, k));
      cv?.getContext('2d').drawImage(v, 0, 0);
      const f = new VideoFrame(cv || v, { timestamp: us(k - k0), duration: us(1) });
      enc.encode(f, { keyFrame: (k - k0) % 48 === 0 });
      f.close();
      while (enc.encodeQueueSize > 4) await new Promise(r => setTimeout(r, 5));
      if (err) throw err;
      onFrame?.(k - k0 + 1, k1 - k0 + 1);
    }
    await enc.flush();
    if (err) throw err;
    mux.finalize();
    return new Blob([mux.target.buffer], { type: 'video/mp4' });
  } finally { if (enc && enc.state !== 'closed') enc.close(); done(); }
}
// Save shots as clips, one at a time in the order asked for. Each argument is [clip, shot].
async function saveClip(...pairs) {
  try { await ensureWrite(); } catch (e) { return say(`Could not save: ${e.message}`); }   // while the click still counts as a user gesture
  for (const [it, s] of pairs) {
    const g = clipRange(it, s), name = clipName(it, g);
    K.encN++;
    K.encQ = K.encQ.then(async () => {
      try {
        const blob = await encodeRange(it, g.a, g.b, (i, n) => status(`Cutting ${name} · frame ${i} / ${n}${K.encN > 1 ? ` · ${K.encN - 1} more queued` : ''}`, i, n));
        if (S.dir) await writeFile(await S.dir.getDirectoryHandle(CDIR, { create: true }), name, blob);
        else download(blob, name);
        K.saved.add(name);
        say(`Saved ${g.b - g.a + 1} frames of ${it.name} as ${S.dir ? `${CDIR}/` : ''}${name}`);
      } catch (e) { say(`Could not cut ${name}: ${e.message}`); }
      // show the saved marks once the queue drains, unless a shot is being previewed in the grid
      if (!--K.encN && S.view === 'shots' && !K.v.closest('.shot')) renderMain();
    });
  }
  return K.encQ;
}
// Copy every still listed in the library into a folder the user picks. Names are kept; two stills of one name get -2, -3…
async function exportStills() {
  if (!window.showDirectoryPicker) return say('This browser cannot write to a folder.');
  const stills = S.list.filter(it => it.kind === 'image' && it.file);
  let dir;
  try { dir = await showDirectoryPicker({ id: 'stills-out', mode: 'readwrite' }); } catch { return; }
  const have = new Map();   // name -> size of what is already there
  for await (const [name, fh] of dir.entries()) have.set(name, fh.kind === 'file' ? (await fh.getFile()).size : -1);
  let done = 0, copied = 0, skipped = 0;
  try {
    for (const it of stills) {
      let name = it.name;
      for (let n = 2; have.has(name) && have.get(name) !== it.size; n++) name = it.name.replace(/(\.[^.]+)?$/, `-${n}$1`);
      if (have.has(name)) skipped++;   // same name and size: already copied
      else { await writeFile(dir, name, it.file); have.set(name, it.size); copied++; }
      status(`Copying stills ${++done} / ${stills.length}`, done, stills.length);
    }
    say(`Copied ${copied} still${copied === 1 ? '' : 's'} to “${dir.name}”` + (skipped ? ` · ${skipped} already there` : ''));
  } catch (e) { say(`Could not copy the stills: ${e.message}`); }
}
async function setCanon(c, f) {
  try { await ensureWrite(); c.canon = f.name; await saveManifest(); renderSide(); }
  catch (e) { say(`Could not save: ${e.message}`); }
}
async function delFrame(c, f) {
  if (!confirm(`Delete ${f.name} from “${c.name}”?${S.dir ? ' The file is removed from disk.' : ''}`)) return;
  try {
    await ensureWrite();
    if (S.dir) await (await kdir(c.name, false)).removeEntry(f.name);
    URL.revokeObjectURL(f.url);
    c.frames = c.frames.filter(x => x !== f);
    if (c.canon === f.name) c.canon = c.frames[0]?.name || null;
    await saveManifest();
    S.items = S.items.filter(x => x.path !== `${KDIR}/${c.name}/${f.name}`);
    refresh();
    renderSide();
  } catch (e) { say(`Could not delete: ${e.message}`); }
}
function gotoClip(it, t) {
  showAt(it, t);
  holdMouse();
  rowOf(it)?.scrollIntoView({ block: 'nearest' });
}
function gotoFrame(f) {
  const it = S.items.find(x => x.path === f.clip && x.rec);
  if (it && f.time != null) gotoClip(it, f.time);
}
// Frames of the previewed clip that other clips share: where it could be cut together with them.
function renderJoinSide() {
  const it = K.it, L = it && indexed(it) ? joins().links.get(it) || [] : [];
  K.joinEl.replaceChildren(...(L.length ? [h('h3', {}, 'Combines with ', h('small', {}, 'frames this clip shares with others')),
    ...L.slice(0, 12).map(l => h('div', { class: 'jrow' },
      h('button', { title: 'Go to this frame of this clip', onclick: () => gotoClip(it, l.t) }, `${l.t.toFixed(2)}s`),
      h('span', { title: { out: 'cut here, continue with that clip', in: 'that clip leads in here', same: 'both show this frame' }[l.rel] }, REL[l.rel]),
      h('button', { class: 'jother', title: `${l.other.it.path}\nvia the ${l.via} · ${pct(l.sim)} match · click to go there`, onclick: () => gotoClip(l.other.it, l.other.t) },
        l.other.it.uf && h('div', { class: 'cellbox' }, cellEl(l.other.it, cellOf(l.other.it, l.other.t))),
        h('span', {}, `${l.other.it.name} · ${l.other.t.toFixed(2)}s`))))] : []));
}

// ----- rendering -----
function renderSide() {
  getCol(K.active);
  K.colsEl.replaceChildren(
    ...K.cols.map((c, i) => h('section', { class: 'kcol' + (c.name === K.active ? ' on' : ''),
      ondragover: e => { e.preventDefault(); e.currentTarget.classList.add('drop'); },
      ondragleave: e => e.currentTarget.classList.remove('drop'),
      ondrop: e => { e.preventDefault(); e.currentTarget.classList.remove('drop'); dropFiles(c, e.dataTransfer.files); } },
      h('h3', { title: 'Click and Enter cut into the highlighted collection', onclick: () => setActive(c.name) },
        i < 9 && h('kbd', {}, String(i + 1)), h('span', {}, c.name), h('small', {}, String(c.frames.length))),
      h('div', { class: 'kframes' }, c.frames.map(f => h('div', { class: 'kf' + (f.name === c.canon ? ' canon' : ''),
        title: `${f.name}${f.clip ? `\nframe ${f.frame} of ${f.clip} · click to go there` : f.from ? `\nfrom ${f.from}` : ''}` },
        h('img', { loading: 'lazy', src: f.url, alt: '', onclick: () => gotoFrame(f) }),
        h('button', { class: 'kstar', title: 'Use as the reference frame for this collection', onclick: () => setCanon(c, f) }, f.name === c.canon ? '★' : '☆'),
        h('button', { class: 'kdel', title: 'Delete this key frame', onclick: () => delFrame(c, f) }, '×')))))),
    h('input', { id: 'colIn', type: 'text', placeholder: 'New collection (N)',
      onkeydown: e => { if (e.key === 'Enter') { addCollection(e.target.value); e.target.blur(); } } }));
  fillColSelects();
}
function renderHarvest() {
  const clips = K.clips = S.list.filter(it => it.kind === 'video' && it.rec && isFinite(it.rec.dur));
  const cuts = new Map();
  for (const c of K.cols) for (const f of c.frames) if (f.clip) cuts.set(f.clip, [...(cuts.get(f.clip) || []), { ...f, col: c.name }]);
  const J = joins().links;
  filmIO.disconnect();
  K.list.replaceChildren(...clips.map(it => {
    const r = it.rec;
    const row = h('div', { class: 'hrow' + (it === K.it ? ' on' : ''), 'data-id': it.id },
      h('div', { class: 'hname' }, h('a', { title: 'Open this clip', onclick: () => openDetail(it) }, it.name),
        h('small', {}, `${secs(r.dur)} · ${r.w}×${r.h}`)),
      h('div', { class: 'hstrip', style: `aspect-ratio:${FILM_N * r.w}/${r.h}` },
        it.uf && h('img', { src: it.uf, alt: '', draggable: false }),
        (cuts.get(it.path) || []).map(f => h('span', { class: 'hcut', style: `left:${(100 * f.time / r.dur).toFixed(2)}%`, title: `${f.col} · frame ${f.frame}` })),
        (indexed(it) && J.get(it) || []).map(l => h('span', { class: 'hjoin', 'data-t': l.t, style: `left:${(100 * l.t / r.dur).toFixed(2)}%`,
          title: `${l.t.toFixed(2)}s ${REL[l.rel]} ${l.other.it.name} at ${l.other.t.toFixed(2)}s · via the ${l.via} · click to go to this frame` })),
        segs(it).slice(1).map(s => h('span', { class: 'hseg', style: `left:${(100 * s.t0 / r.dur).toFixed(2)}%` })),
        it === K.it && K.head));
    filmIO.observe(row);
    return row;
  }));
  if (!clips.length) K.list.append(h('div', { class: 'empty' }, S.items.length ? 'No clips match the current filters.' : 'No folder open.'));
  renderSide();
  renderJoinSide();
  const grid = $('#grid'), prev = $('.hprev', K.side);
  if (K.v.parentNode !== prev) prev.prepend(K.v);   // the shots view borrows the player
  if (K.list.parentNode !== grid) grid.replaceChildren(K.list, K.side);   // re-inserting would reset the scroll and the player
}

K.list.addEventListener('mousemove', e => {
  K.mx = e.clientX; K.my = e.clientY;
  if (K.lock && Math.hypot(e.clientX - K.lock.x, e.clientY - K.lock.y) < 8) return;
  K.lock = null;
  const s = e.target.closest('.hstrip');
  const it = s && S.items.find(x => x.id === s.parentNode.dataset.id);
  if (!it) return;
  const b = s.getBoundingClientRect();
  showAt(it, (e.clientX - b.left) / b.width * it.rec.dur);
});
K.list.addEventListener('click', e => {
  const s = e.target.closest('.hstrip');
  if (!s) return;
  if (e.target.closest('.hjoin')) {   // a join mark jumps to its frame instead of cutting
    const it = S.items.find(x => x.id === s.parentNode.dataset.id);
    return it && gotoClip(it, +e.target.closest('.hjoin').dataset.t);
  }
  if (s.parentNode.dataset.id !== K.it?.id) {   // clicked a strip the mouse has not scrubbed yet
    const it = S.items.find(x => x.id === s.parentNode.dataset.id), b = s.getBoundingClientRect();
    if (!it) return;
    showAt(it, (e.clientX - b.left) / b.width * it.rec.dur);
  }
  cut();
});

// ---------- storyboard: a canvas of panels (stills, clips, notes) joined by transition ideas ----------
// Kept as storyboard.json in the opened folder, so the plan travels with the footage; a copy stays in this browser.
// A link from one still to another is a clip still to be made: its start frame, its end frame, and what happens between.
const BFILE = 'storyboard.json', BMIME = 'application/x-media-item';
const B = { cards: [], links: [], found: [], state: new Map(), byPath: new Map(), x: 40, y: 60, z: 1, sel: null, undo: null,
  busy: false, pull: null, play: null };
const sv = (tag, attrs) => {
  const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
};
B.wg = sv('g', {});
B.wires = sv('svg', { class: 'bwires' });
B.wires.append(sv('defs', {}), B.wg);
B.wires.firstChild.append(sv('marker', { id: 'barrow', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto' }));
B.wires.firstChild.firstChild.append(sv('path', { d: 'M0,0 L10,5 L0,10 z', fill: 'context-stroke', stroke: 'none' }));
B.world = h('div', { class: 'bworld' });
B.zoomEl = h('small', {});
B.hint = h('p', { class: 'bhint' }, 'Drag stills and clips in from the right. Double-click the board for a note. ',
  'Drag the dot on a panel to another panel to link them, or onto empty space to branch into a new idea.');
B.el = h('div', { class: 'bcanvas' }, B.world, B.hint, h('div', { class: 'bbar' },
  h('button', { title: 'A note panel in the middle of the view (N), or double-click the board', onclick: () => newNote() }, '+ Note'),
  h('button', { title: 'Show the whole board (F)', onclick: () => fitBoard() }, 'Fit'), B.zoomEl));
B.tray = h('aside', { class: 'btray' });

const uid = () => Math.random().toString(36).slice(2, 10);
const bkey = () => 'board:' + (S.dir?.name || '');
const cardOf = id => B.cards.find(c => c.id === id);
const cardEl = id => B.world.querySelector(`.bcard[data-id="${id}"]`);
const itemOf = id => B.byPath.get(cardOf(id)?.path);
const linked = (a, b) => B.links.some(l => (l.from === a && l.to === b) || (l.from === b && l.to === a));
function toWorld(e) {
  const r = B.el.getBoundingClientRect();
  return { x: (e.clientX - r.left - B.x) / B.z, y: (e.clientY - r.top - B.y) / B.z };
}

function saveBoard() {
  clearTimeout(B.saveT);
  B.saveT = setTimeout(async () => {
    const text = JSON.stringify({ version: 1, saved: Date.now(), cards: B.cards, links: B.links }, null, 2);
    DB.set('kv', bkey(), text);
    if (!S.dir) return;
    try { await ensureWrite(); await writeFile(S.dir, BFILE, text); B.offDisk = false; }
    catch (e) { if (!B.offDisk) say(`The storyboard is kept in this browser only: ${e.message}`); B.offDisk = true; }
  }, 400);
}
// The newer of the file in the folder and the browser's copy wins, so a save the folder refused is not lost.
async function loadBoard() {
  clearTimeout(B.saveT);
  const read = async p => { try { return JSON.parse(await p); } catch { return null; } };
  const [disk, mine, view] = await Promise.all([
    S.dir ? read(S.dir.getFileHandle(BFILE).then(f => f.getFile()).then(f => f.text())) : null,
    read(DB.get('kv', bkey())), DB.get('kv', 'view:' + bkey()).catch(() => null)]);
  const d = (mine?.saved || 0) > (disk?.saved || 0) ? mine : disk || mine || {};
  Object.assign(B, { cards: d.cards || [], links: d.links || [], sel: null, undo: null, offDisk: false, x: 40, y: 60, z: 1, ...view });
  if (S.view === 'board') renderBoard();
}
function setView() {
  B.world.style.transform = `translate(${B.x}px,${B.y}px) scale(${B.z})`;
  B.el.style.backgroundPosition = `${B.x}px ${B.y}px`;
  B.el.style.backgroundSize = `${24 * B.z}px ${24 * B.z}px`;
  B.zoomEl.textContent = pct(B.z);
  clearTimeout(B.viewT);
  B.viewT = setTimeout(() => DB.set('kv', 'view:' + bkey(), { x: B.x, y: B.y, z: B.z }), 300);
}
function fitBoard() {
  const els = [...B.world.querySelectorAll('.bcard')], r = B.el.getBoundingClientRect();
  if (!els.length || !r.width) return;
  const x0 = Math.min(...els.map(e => e.offsetLeft)), x1 = Math.max(...els.map(e => e.offsetLeft + e.offsetWidth));
  const y0 = Math.min(...els.map(e => e.offsetTop)), y1 = Math.max(...els.map(e => e.offsetTop + e.offsetHeight));
  B.z = Math.min(1, Math.max(0.2, Math.min((r.width - 80) / (x1 - x0), (r.height - 80) / (y1 - y0))));
  B.x = (r.width - (x0 + x1) * B.z) / 2;
  B.y = (r.height - (y0 + y1) * B.z) / 2;
  setView();
}

// What the library already knows about a link: whether the footage for it exists.
function linkState(l) {
  const a = itemOf(l.from), b = itemOf(l.to);
  if (!a?.links || !b?.links) return {};
  if (a.kind === 'image' && b.kind === 'image') {
    const clips = a.links.starts.map(x => x.it).filter(c => b.links.ends.some(x => x.it === c));
    return clips.length ? { clips } : { gap: true };
  }
  const hit = a.kind === 'image' ? a.links.starts : b.kind === 'video' ? a.links.next : a.links.endStill;
  return { match: hit.some(x => x.it === b) };
}
// Panels on the board that share a frame but are not linked yet: offered as links.
function foundLinks() {
  const on = new Map(B.cards.map(c => [B.byPath.get(c.path), c]));
  const out = [];
  for (const [it, c] of on) {
    if (!it?.links) continue;
    for (const l of it.kind === 'video' ? [...it.links.next, ...it.links.endStill] : it.links.starts) {
      const d = on.get(l.it);
      if (d && d !== c && !linked(c.id, d.id)) out.push({ from: c.id, to: d.id });
    }
  }
  return out;
}

function cardNode(c) {
  const it = c.path && B.byPath.get(c.path), clip = it?.kind === 'video';
  const frame = (url, label) => h('figure', {}, h('img', { src: url, alt: '', draggable: false }), h('figcaption', {}, label));
  return h('div', { class: `bcard${c.path ? clip ? ' clip' : '' : ' note'}${c.id === B.sel ? ' sel' : ''}`, 'data-id': c.id, style: `left:${c.x}px;top:${c.y}px` },
    c.path && h('div', { class: 'bmedia', title: `${c.path}\nDouble-click to open${clip ? ' · hover to play' : ''}` },
      !it?.u1 ? h('div', { class: 'err' }, it ? 'Cannot decode' : S.scanning ? 'Scanning…' : 'Not in this folder any more')
        : clip ? [frame(it.u1, 'start'), it.u2 && frame(it.u2, 'end')] : h('img', { src: it.u1, alt: '', draggable: false }),
      it?.rec && h('span', { class: 'kind' }, clip ? '▶ ' + secs(it.rec.dur) : 'still')),
    c.path && h('div', { class: 'name' }, c.path.slice(c.path.lastIndexOf('/') + 1)),
    h('textarea', { class: 'btext', spellcheck: false, placeholder: c.path ? 'Note…' : 'An idea, a beat, a gap to fill…', value: c.text || '',
      oninput: e => { c.text = e.target.value; drawWires(); saveBoard(); } }),
    h('button', { class: 'bport', title: 'Drag to another panel to link them · drop on empty space to branch into a new idea' }),
    h('button', { class: 'bdel', title: 'Remove from the board (Delete)', onclick: () => removeCard(c) }, '×'));
}
function linkNode(l) {
  const s = B.state.get(l.id);
  return h('div', { class: 'blink', 'data-link': l.id },
    h('textarea', { class: 'btext', rows: 1, spellcheck: false, placeholder: 'Transition…', value: l.text || '',
      oninput: e => { l.text = e.target.value; saveBoard(); } }),
    s.clips && h('button', { class: 'chip still', title: `${s.clips.map(c => c.path).join('\n')}\nStarts on the first still and ends on the second. Click to open.`,
      onclick: () => openDetail(s.clips[0]) }, `▶ ${s.clips.length} clip${s.clips.length > 1 ? 's' : ''}`),
    s.gap && h('span', { class: 'chip', title: 'No clip runs from the first still to the second yet' }, 'to make'),
    s.match && h('span', { class: 'chip still', title: 'The first panel ends on the frame the second starts on' }, '✓ same frame'),
    h('button', { class: 'bdel', title: 'Remove this link', onclick: () => removeLink(l) }, '×'));
}
function drawBoard() {
  stopBoardPlay();
  B.byPath = new Map(S.items.map(it => [it.path, it]));
  B.state = new Map(B.links.map(l => [l.id, linkState(l)]));
  B.found = foundLinks();
  B.world.replaceChildren(B.wires, ...B.cards.map(cardNode), ...B.links.map(linkNode),
    ...B.found.map(f => h('button', { class: 'blink found', 'data-link': `${f.from}>${f.to}`,
      title: 'These two share a frame, so they cut together. Click to link them.',
      onclick: () => { addLink(f.from, f.to); drawBoard(); saveBoard(); } }, '+ link')));
  B.hint.hidden = B.cards.length > 0;
  drawWires();
}
// Wires leave a panel's right edge and arrive at the other's left; each label sits at the middle of its wire.
function drawWires() {
  const out = [];
  const wire = (a, b, cls) => {
    const dx = Math.max(40, Math.abs(b.x - a.x) / 2);
    out.push(sv('path', { class: cls, 'marker-end': 'url(#barrow)', d: `M${a.x},${a.y} C${a.x + dx},${a.y} ${b.x - dx},${b.y} ${b.x},${b.y}` }));
  };
  for (const l of [...B.links, ...B.found]) {
    const a = cardEl(l.from), b = cardEl(l.to), lab = B.world.querySelector(`[data-link="${l.id || `${l.from}>${l.to}`}"]`);
    if (!a || !b) continue;
    const p = { x: a.offsetLeft + a.offsetWidth, y: a.offsetTop + a.offsetHeight / 2 }, q = { x: b.offsetLeft, y: b.offsetTop + b.offsetHeight / 2 };
    wire(p, q, l.id ? B.state.get(l.id)?.gap ? 'gap' : '' : 'found');
    if (lab) { lab.style.left = (p.x + q.x) / 2 + 'px'; lab.style.top = (p.y + q.y) / 2 + 'px'; }
  }
  if (B.pull) wire(B.pull.a, B.pull.b, 'pull');
  B.wg.replaceChildren(...out);
}
function renderTray() {
  const used = new Set(B.cards.map(c => c.path)), top = B.tray.scrollTop;
  B.tray.replaceChildren(h('p', {}, 'Drag a still or clip onto the board, or click it. The toolbar filters this list.'),
    h('div', { class: 'btiles' }, S.list.filter(it => it.u1).map(it => h('div', { class: 'bt' + (used.has(it.path) ? ' used' : ''), draggable: true, title: it.path,
      ondragstart: e => { e.dataTransfer.setData(BMIME, it.id); e.dataTransfer.effectAllowed = 'copy'; }, onclick: () => boardAdd(it) },
      it.tc && it.uf ? h('div', { class: 'cellbox' }, cellEl(it, it.tc.i)) : h('img', { loading: 'lazy', src: it.u1, alt: '', draggable: false }),
      h('span', { class: 'kind' }, it.kind === 'image' ? 'still' : '▶ ' + secs(it.rec.dur))))));
  B.tray.scrollTop = top;
}
function renderBoard() {
  const grid = $('#grid');
  if (B.el.parentNode !== grid) grid.replaceChildren(B.el, B.tray);
  $('#exportStills').hidden = true;
  renderTray();
  if (!B.busy && !B.world.contains(document.activeElement)) drawBoard();   // a rescan must not pull a note out from under the cursor
  setView();
}

function selectCard(id) {
  B.sel = id;
  for (const e of B.world.querySelectorAll('.bcard')) e.classList.toggle('sel', e.dataset.id === id);
}
function addCard(at, more) {
  const c = { id: uid(), x: Math.round(at.x), y: Math.round(at.y), ...more };
  B.cards.push(c);
  return c;
}
function addLink(from, to) { B.links.push({ id: uid(), from, to, text: '' }); }
function midView(w, hgt) {   // where a panel of that size sits in the middle of the view, stepped so several do not stack
  const r = B.el.getBoundingClientRect(), n = B.cards.length % 6 * 24;
  return { x: (r.width / 2 - B.x) / B.z - w / 2 + n, y: (r.height / 2 - B.y) / B.z - hgt / 2 + n };
}
function newNote(at = midView(200, 60)) {
  const c = addCard(at);
  drawBoard();
  selectCard(c.id);
  $('.btext', cardEl(c.id)).focus();
  saveBoard();
}
// B in the library or an open item, a click or a drop from the tray: the still or clip becomes a panel.
function boardAdd(it, at) {
  if (!it?.rec) return;
  const last = B.cards.at(-1);
  const c = addCard(at || (S.view === 'board' ? midView(230, 180) : last ? { x: last.x + 360, y: last.y } : { x: 0, y: 0 }), { path: it.path });
  if (S.view === 'board') { renderTray(); drawBoard(); selectCard(c.id); }
  else say(`Added ${it.name} to the storyboard`);
  saveBoard();
}
function removeCard(c) {
  B.undo = JSON.stringify({ cards: B.cards, links: B.links });
  B.cards = B.cards.filter(x => x !== c);
  B.links = B.links.filter(l => l.from !== c.id && l.to !== c.id);
  if (B.sel === c.id) B.sel = null;
  renderTray(); drawBoard(); saveBoard();
  say('Panel removed. Z puts it back.');
}
function removeLink(l) {
  B.undo = JSON.stringify({ cards: B.cards, links: B.links });
  B.links = B.links.filter(x => x !== l);
  drawBoard(); saveBoard();
  say('Link removed. Z puts it back.');
}
function undoBoard() {
  if (!B.undo) return;
  Object.assign(B, JSON.parse(B.undo), { undo: null });
  renderTray(); drawBoard(); saveBoard();
}
// A still at full size, or a clip's last frame: the picture the next clip would start from.
async function fullFrame(it) {
  const png = (src, w, hgt) => { const cv = new OffscreenCanvas(w, hgt); cv.getContext('2d').drawImage(src, 0, 0); return cv.convertToBlob({ type: 'image/png' }); };
  if (it.kind === 'image') {
    const bmp = await createImageBitmap(it.file);
    try { return await png(bmp, bmp.width, bmp.height); } finally { bmp.close(); }
  }
  const { v, done } = await openVideo(it.file);
  try { await seek(v, it.rec.e2?.t ?? Infinity); return await png(v, v.videoWidth, v.videoHeight); } finally { done(); }
}
function copyCard(it) {
  navigator.clipboard.write([new ClipboardItem({ 'image/png': fullFrame(it) })])
    .then(() => say(it.kind === 'image' ? `Copied ${it.name} at full size. Paste it with ⌘V.` : `Copied the last frame of ${it.name}, to start the next clip from.`),
      e => say(`Could not copy: ${e.message}`));
}
function stopBoardPlay() {
  if (!B.play) return;
  B.play.v.remove(); URL.revokeObjectURL(B.play.url); B.play = null;
}

function track(e, move, up) {   // follow the mouse from a mousedown until it is released
  e.preventDefault();
  B.busy = true;
  const mv = m => move((m.clientX - e.clientX), (m.clientY - e.clientY), m);
  const end = m => { removeEventListener('mousemove', mv); removeEventListener('mouseup', end); B.busy = false; up?.(m); };
  addEventListener('mousemove', mv);
  addEventListener('mouseup', end);
}
B.el.addEventListener('mousedown', e => {
  if (e.button || e.target.closest('.bbar, .blink, .bdel')) return;
  const el = e.target.closest('.bcard'), c = el && cardOf(el.dataset.id);
  if (c && e.target.tagName === 'TEXTAREA') return selectCard(c.id);
  document.activeElement?.blur();
  if (!c) {
    const x0 = B.x, y0 = B.y;
    selectCard(null);
    return track(e, (dx, dy) => { B.x = x0 + dx; B.y = y0 + dy; setView(); });
  }
  selectCard(c.id);
  if (e.target.classList.contains('bport')) {
    const a = { x: c.x + el.offsetWidth, y: c.y + el.offsetHeight / 2 };
    return track(e, (dx, dy, m) => { B.pull = { a, b: toWorld(m) }; drawWires(); }, m => {
      const pulled = B.pull, t = document.elementFromPoint(m.clientX, m.clientY), to = t?.closest('.bcard')?.dataset.id;
      B.pull = null;
      let fresh;
      if (to && to !== c.id) { if (!linked(c.id, to)) addLink(c.id, to); }
      else if (pulled && t === B.el) { fresh = addCard({ x: pulled.b.x, y: pulled.b.y - 30 }); addLink(c.id, fresh.id); }
      drawBoard();
      if (fresh) { selectCard(fresh.id); $('.btext', cardEl(fresh.id)).focus(); }
      saveBoard();
    });
  }
  const x0 = c.x, y0 = c.y;
  let moved = false;
  track(e, (dx, dy) => {
    moved = true;
    el.style.left = (c.x = Math.round(x0 + dx / B.z)) + 'px';
    el.style.top = (c.y = Math.round(y0 + dy / B.z)) + 'px';
    drawWires();
  }, () => moved && saveBoard());
});
B.el.addEventListener('dblclick', e => {
  if (e.target === B.el) { const p = toWorld(e); return newNote({ x: p.x - 100, y: p.y - 20 }); }
  const m = e.target.closest('.bmedia'), it = m && itemOf(m.parentNode.dataset.id);
  if (it) openDetail(it);
});
B.el.addEventListener('wheel', e => {   // scroll pans; pinch, or scroll with ⌘ or Ctrl, zooms about the pointer
  e.preventDefault();
  if (e.ctrlKey || e.metaKey) {
    const r = B.el.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    const z = Math.min(2, Math.max(0.2, B.z * Math.exp(-e.deltaY * 0.01)));
    B.x = mx - (mx - B.x) * z / B.z;
    B.y = my - (my - B.y) * z / B.z;
    B.z = z;
  } else { B.x -= e.deltaX; B.y -= e.deltaY; }
  setView();
}, { passive: false });
B.el.addEventListener('dragover', e => { if (e.dataTransfer.types.includes(BMIME)) e.preventDefault(); });
B.el.addEventListener('drop', e => {
  const it = S.items.find(x => x.id === e.dataTransfer.getData(BMIME));
  if (!it) return;
  e.preventDefault();
  const p = toWorld(e);
  boardAdd(it, { x: p.x - 115, y: p.y - 60 });
});
B.el.addEventListener('mouseover', e => {
  const m = e.target.closest('.bcard.clip .bmedia'), it = m && itemOf(m.parentNode.dataset.id);
  if (!it?.file || B.busy || B.play?.m === m) return;
  stopBoardPlay();
  const url = URL.createObjectURL(it.file);
  B.play = { m, url, v: h('video', { src: url, muted: true, loop: true, autoplay: true, playsInline: true }) };
  m.append(B.play.v);
});
B.el.addEventListener('mouseout', e => { if (B.play && !B.play.m.contains(e.relatedTarget)) stopBoardPlay(); });

// ---------- wiring ----------
function setSeg(id, key, v) {
  S[key] = v;
  for (const b of document.querySelectorAll(`#${id} button`)) b.classList.toggle('on', b.dataset.v === v);
  renderMain();
}
function setThr(v) {
  S.thr = v;
  $('#thr').value = Math.round(v * 100);
  $('#thrOut').textContent = pct(v);
}

$('#open').onclick = pickFolder;
$('#rescan').onclick = () => S.source?.();
$('#index').onclick = toggleIndex;
$('#exportStills').onclick = exportStills;
$('#fallback').onchange = e => useFileList(e.target.files);
$('#type').onclick = e => e.target.dataset.v && setSeg('type', 'type', e.target.dataset.v);
$('#view').onclick = e => e.target.dataset.v && setSeg('view', 'view', e.target.dataset.v);
$('#folder').onchange = e => { S.folder = e.target.value; renderMain(); };
$('#sort').onchange = e => { S.sort = e.target.value; renderMain(); };
$('#q').oninput = e => { S.q = e.target.value; renderMain(); };
$('#starred').onchange = e => { S.starred = e.target.checked; renderMain(); };
$('#hover').onchange = e => setHover(e.target.checked);
$('#thr').oninput = e => { setThr(e.target.value / 100); };
$('#thr').onchange = () => { DB.set('kv', 'thr', S.thr); refresh(); if (S.detail) openDetail(S.detail); };
$('#help').onclick = () => $('#keys').showModal();

$('#grid').addEventListener('click', e => {
  const t = e.target.closest('.tile');
  if (!t) return;
  select(t.dataset.id, false);
  if (S.view === 'shots') { const x = e.target.closest('a') && shotOf(t.dataset.id); return x ? openDetail(x.it, x.s.tk) : openSelected(); }
  if (S.view !== 'joins') openSelected();   // a join's thumbnails open their own clip
});
// Shots: the tile under the mouse becomes a scrubber over just that shot.
$('#grid').addEventListener('mousemove', e => {
  const th = S.view === 'shots' && e.target.closest('.shot .thumb'), x = th && shotOf(th.parentNode.dataset.id);
  if (!x) return;
  const b = th.getBoundingClientRect(), f = Math.min(1, Math.max(0, (e.clientX - b.left) / b.width));
  if (K.v.parentNode !== th) th.append(K.v, K.at);
  showAt(x.it, Math.min(x.s.t0 + f * (x.s.t1 - x.s.t0), x.s.t1 - 0.75 * fdur(x.it)));
});
$('#grid').addEventListener('mouseover', e => {
  const t = e.target.closest('.tile:not(.group)');
  if (!S.hover || !t || hoverVid?.t === t) return;
  startHover(t);
});
$('#grid').addEventListener('mouseout', e => {
  if (hoverVid && !hoverVid.t.contains(e.relatedTarget)) stopHover();
  const th = K.v.parentNode;
  if (th?.classList.contains('thumb') && !th.contains(e.relatedTarget)) { K.v.remove(); K.at.remove(); }
});

document.addEventListener('keydown', e => {
  const t = e.target, typing = (t.tagName === 'INPUT' && !/checkbox|range/.test(t.type)) || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA';
  if ($('#keys').open) return;
  if (e.key === 'Escape') {
    if (typing) t.blur(); else if (C.open) closeCompare(); else if (Q.open) closeSheet(); else if (S.detail) closeDetail();
    return;
  }
  if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  let used = true;
  if (k === '?') $('#keys').showModal();
  else if (C.open) {
    if (k === ' ') playAll(!C.playing);
    else if (k === 'ArrowLeft' || k === 'ArrowRight') { playAll(false); setFrac(C.frac + (k === 'ArrowLeft' ? -1 : 1) * (e.shiftKey ? 0.1 : 0.01)); }
    else if (k >= '1' && k <= '9' && C.mem[k - 1]) { toggleStar(C.mem[k - 1]); renderLabels(); }
    else if (k === 'd') setDiff(!C.diff);
    else if (k === 'h') setHover(!S.hover);
    else used = false;
  } else if (Q.open) {
    used = sheetKey(k, e);
  } else if (S.detail) {
    const it = S.detail;
    if (k === 'ArrowLeft') stepDetail(-1);
    else if (k === 'ArrowRight') stepDetail(1);
    else if (k === ' ') { const v = $('#dplay'); if (v) v.paused ? v.play() : v.pause(); }
    else if (k === 's') { toggleStar(it); renderDetailMeta(); }
    else if (k === 't') $('#tagIn').focus();
    else if (k === 'a') collect(it);
    else if (k === 'b') boardAdd(it);
    else if (k === 'c') openCompare(it.group);
    else used = false;
  } else if (S.view === 'harvest') {
    if (k === 'ArrowLeft' || k === 'ArrowRight') nudge((k === 'ArrowLeft' ? -1 : 1) * (e.shiftKey ? 10 : 1));
    else if (k === 'ArrowUp') stepClip(-1);
    else if (k === 'ArrowDown') stepClip(1);
    else if (k === ' ') togglePlay();
    else if (k === 'Enter') cut();
    else if (k >= '1' && k <= '9') { if (K.cols[k - 1]) cut(K.cols[k - 1].name); }
    else if (k === 'n') $('#colIn').focus();
    else if (k === '/') $('#q').focus();
    else if (k === 'g') setSeg('view', 'view', 'board');
    else used = false;
  } else if (S.view === 'board') {
    const c = cardOf(B.sel), it = c && itemOf(c.id);
    if (k === 'n') newNote();
    else if (k === 'f') fitBoard();
    else if (k === 'z') undoBoard();
    else if ((k === 'Delete' || k === 'Backspace') && c) removeCard(c);
    else if (k === 'Enter' && it) openDetail(it);
    else if (k === 'c' && it) copyCard(it);
    else if (k === '/') $('#q').focus();
    else if (k === 'g') setSeg('view', 'view', 'library');
    else used = false;
  } else {
    if (k === 'ArrowLeft') moveSel(-1, 0);
    else if (k === 'ArrowRight') moveSel(1, 0);
    else if (k === 'ArrowUp') moveSel(0, -1);
    else if (k === 'ArrowDown') moveSel(0, 1);
    else if (k === 'Enter' || k === ' ') openSelected();
    else if (k === 's' && S.view === 'library') { const it = S.items.find(x => x.id === S.sel); if (it) toggleStar(it); }
    else if (k === 'a' && S.view === 'library') collect(S.items.find(x => x.id === S.sel));
    else if (k === 'b' && S.view === 'library') boardAdd(S.items.find(x => x.id === S.sel));
    else if (k === 'a' && S.view === 'shots') cutShot(S.sel);
    else if (k === 'v' && S.view === 'shots') { const x = shotOf(S.sel); if (x) saveClip([x.it, x.s]); }
    else if (k === '/') $('#q').focus();
    else if (k === 'g') setSeg('view', 'view', VIEWS[(VIEWS.indexOf(S.view) + 1) % VIEWS.length]);
    else if ('123'.includes(k)) setSeg('type', 'type', ['all', 'image', 'video'][k - 1]);
    else used = false;
  }
  if (used) e.preventDefault();
});

(async () => {
  const [thr, hover, dir, kcol, kpad] = await Promise.all(['thr', 'hover', 'dir', 'kcol', 'kpad'].map(k => DB.get('kv', k).catch(() => undefined)));
  K.pad = kpad || 0;
  setThr(thr ?? S.thr);
  if (kcol) K.active = kcol;
  renderSide();
  S.hover = $('#hover').checked = !!hover;
  if (dir && dir.requestPermission) {
    const b = $('#reopen');
    b.textContent = `Reopen “${dir.name}”`;
    b.hidden = false;
    b.onclick = async () => { if (await dir.requestPermission({ mode: 'read' }) === 'granted') useDir(dir); };
  }
})();
