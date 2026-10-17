"use strict";

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const store = {
  get: (k) => { try { return localStorage.getItem(k) || ""; } catch { return ""; } },
  set: (k, v) => { try { v ? localStorage.setItem(k, v) : localStorage.removeItem(k); } catch { /* private mode */ } },
};

const state = {
  sid: Math.random().toString(36).slice(2, 8), // lets a partner tell this tab's segments from an older tab's
  role: "A", room: "", ws: null, joined: false,
  peer: { connected: false, lang: null },
  languages: {}, seq: 0, listening: false, mic: null,
  voiceId: store.get("voiceId"),
};

// Audio worklet: batches 2048 mono samples at a time and posts them to the main thread.
const WORKLET = `
class Pcm extends AudioWorkletProcessor {
  constructor() { super(); this.buf = new Float32Array(2048); this.n = 0; }
  process(inputs) {
    const ch = inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.buf[this.n++] = ch[i];
      if (this.n === this.buf.length) { this.port.postMessage(this.buf.slice(0)); this.n = 0; }
    }
    return true;
  }
}
registerProcessor("pcm", Pcm);`;

const name = (code) => state.languages[code] || code;
const setStatus = (t) => { $("status").textContent = t; };

// ---------------------------------------------------------------- audio capture

async function startMic(onChunk, raw = false) {
  // Humming needs the raw signal: noise suppression treats a steady hum as noise and removes it.
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: raw
      ? { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false }
      : { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });
  const ctx = new AudioContext();
  const url = URL.createObjectURL(new Blob([WORKLET], { type: "text/javascript" }));
  await ctx.audioWorklet.addModule(url);
  const source = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, "pcm");
  node.port.onmessage = (e) => onChunk(e.data);
  const mute = ctx.createGain(); // keeps the node pulled by the graph without playing the mic back
  mute.gain.value = 0;
  source.connect(node); node.connect(mute); mute.connect(ctx.destination);
  return { stream, ctx, sampleRate: ctx.sampleRate };
}

function stopMic(m) {
  m.stream.getTracks().forEach((t) => t.stop());
  m.ctx.close();
}

function concat(chunks) {
  const out = new Float32Array(chunks.reduce((n, c) => n + c.length, 0));
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

// Box-filter downsample; good enough for speech at 16 kHz.
function resample(x, from, to) {
  if (from === to) return x;
  const ratio = from / to, n = Math.floor(x.length / ratio), y = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const a = Math.floor(i * ratio), b = Math.max(a + 1, Math.floor((i + 1) * ratio));
    let s = 0;
    for (let j = a; j < b && j < x.length; j++) s += x[j];
    y[i] = s / (b - a);
  }
  return y;
}

function encodeWav(f32, rate) {
  const buf = new ArrayBuffer(44 + f32.length * 2), v = new DataView(buf);
  const tag = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  tag(0, "RIFF"); v.setUint32(4, 36 + f32.length * 2, true); tag(8, "WAVE"); tag(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  tag(36, "data"); v.setUint32(40, f32.length * 2, true);
  for (let i = 0; i < f32.length; i++) {
    const s = Math.max(-1, Math.min(1, f32[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

const toWav16k = (chunks, rate) => encodeWav(resample(concat(chunks), rate, 16000), 16000);

// ---------------------------------------------------------------- pause-triggered segmentation
// The cascaded cousin of simultaneous translation: translate each phrase as soon as the speaker
// pauses, instead of waiting for the whole turn.

const seg = { chunks: [], speechMs: 0, silenceMs: 0, totalMs: 0, noise: 0.004 };
const resetSeg = () => { seg.chunks = []; seg.speechMs = 0; seg.silenceMs = 0; seg.totalMs = 0; };

function onPcm(f32) {
  if (!state.mic) return; // a late chunk after stop, or one that beat startListening's assignment
  const rate = state.mic.sampleRate, ms = (f32.length / rate) * 1000;
  let sum = 0;
  for (let i = 0; i < f32.length; i++) sum += f32[i] * f32[i];
  const rms = Math.sqrt(sum / f32.length);
  $("levelBar").style.width = Math.min(100, rms * 600) + "%";

  const speaking = rms > Math.max(0.012, seg.noise * 3);
  if (!speaking) seg.noise = seg.noise * 0.9 + rms * 0.1;
  seg.chunks.push(f32);
  seg.totalMs += ms;
  if (speaking) { seg.speechMs += ms; seg.silenceMs = 0; } else { seg.silenceMs += ms; }

  if (!seg.speechMs) { // still waiting for speech: keep only about 300 ms of lead-in
    while (seg.chunks.length > 7) seg.chunks.shift();
    seg.totalMs = seg.chunks.length * ms;
    return;
  }
  const pauseCut = $("streaming").checked && seg.speechMs >= 500 && seg.silenceMs >= 650;
  if (pauseCut || seg.totalMs >= 12000) flushSegment();
}

function flushSegment() {
  const { chunks, speechMs } = seg;
  resetSeg();
  if (speechMs < 300 || !state.mic) return;
  send(toWav16k(chunks, state.mic.sampleRate));
}

async function startListening() {
  if (state.listening) return;
  stopPlayback(); // barge-in: talking over the interpreter silences it
  try {
    state.mic = await startMic(onPcm);
  } catch (e) {
    setStatus("Microphone unavailable: " + e.message);
    return;
  }
  resetSeg();
  state.listening = true;
  $("mic").textContent = "Stop";
  $("mic").classList.add("live");
  setStatus("Listening. Pause briefly and each phrase is interpreted.");
}

function stopListening() {
  if (!state.listening) return;
  state.listening = false;
  flushSegment();
  stopMic(state.mic);
  state.mic = null;
  $("levelBar").style.width = "0";
  $("mic").textContent = "Start talking";
  $("mic").classList.remove("live");
  setStatus("");
  pump();
}

// ---------------------------------------------------------------- send and render

function baseForm(seq) {
  const form = new FormData();
  form.append("role", state.role);
  form.append("room", state.joined ? state.room : "");
  form.append("seq", seq);
  form.append("sid", state.sid);
  form.append("voice_id", state.voiceId);
  form.append("glossary", $("glossary").value);
  return form;
}

// Spoken phrase: audio goes to speech recognition first.
function send(wav) {
  const seq = state.seq++;
  const form = baseForm(seq);
  form.append("file", wav, "segment.wav");
  form.append("src", $("mine").value);
  form.append("tgt", $("theirs").value);
  form.append("tone_aware", $("toneAware").checked);
  return submit("/api/interpret", form, seq);
}

// Typed text takes the same translate-and-speak path.
function say(text, src, tgt) {
  const seq = state.seq++;
  const form = baseForm(seq);
  form.append("text", text);
  form.append("src", src);
  form.append("tgt", tgt);
  return submit("/api/say", form, seq);
}

async function submit(path, form, seq) {
  const started = performance.now();
  const entry = document.createElement("article");
  entry.className = "entry me";
  renderEntry(entry, "me", { pending: true });
  $("log").prepend(entry);
  try {
    const init = form instanceof FormData
      ? { method: "POST", body: form }
      : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(form) };
    const r = await fetch(path, init);
    const body = await r.json();
    if (!r.ok) throw new Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail));
    if (body.empty) {
      entry.remove();
      if (!body.delivered) enqueue({ sid: state.sid, seq, skip: true });
      return;
    }
    body.roundtrip_ms = Math.round(performance.now() - started);
    renderEntry(entry, "me", body);
    if (!body.delivered) enqueue(body); // solo mode: play the translation here
    if ($("readback").checked && body.kind !== "song") runCheck(entry, body); // after playback is queued
  } catch (e) {
    renderEntry(entry, "me", { error: "Interpretation failed: " + e.message });
    if (!state.peer.connected) enqueue({ sid: state.sid, seq, skip: true });
  }
}

// Read-back: translate the translation back and show the speaker what the partner actually heard.
async function runCheck(entry, d) {
  try {
    const r = await fetch("/api/check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transcript: d.transcript, translation: d.translation, src: d.src, tgt: d.tgt, judge: $("judge").checked }),
    });
    const body = await r.json();
    if (!r.ok) throw new Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail));
    d.check = body;
  } catch (e) {
    d.check = { error: e.message };
  }
  renderEntry(entry, "me", d);
}

function renderEntry(el, who, d) {
  el.replaceChildren();
  const line = (cls, text) => {
    const p = document.createElement("p");
    p.className = cls; p.textContent = text; el.append(p);
  };
  if (d.pending) return line("meta", "Interpreting...");
  if (d.error) return line("error", d.error);
  const what = d.kind === "song" ? `song${d.title ? " “" + d.title + "”" : ""}, ` : "";
  line("meta", `${who === "me" ? "You" : "Partner"}: ${what}${name(d.src)} to ${name(d.tgt)}${d.mock ? " (mock)" : ""}`);
  line("orig", d.transcript);
  line("trans", d.translation);
  const check = d.check || {};
  if (check.back_translation) line("readback", "They heard: " + check.back_translation);

  const chips = document.createElement("div");
  chips.className = "chips";
  const chip = (text, cls = "") => {
    const s = document.createElement("span");
    s.className = "chip " + cls; s.textContent = text; chips.append(s);
  };
  const tone = d.tone || {};
  const guard = d.guard || { status: "none" };
  if (tone.summary) chip("delivery: " + tone.summary);
  if (tone.labels && tone.labels.hesitant) chip("sounded hesitant", "warn");
  if (guard.status === "ok") chip("numbers match: " + guard.source.join(", "));
  if (guard.status === "differs") {
    chip(`numbers differ: said ${guard.source.join(", ") || "none"}, translated ${guard.translated.join(", ") || "none"}`, "warn");
  }
  if (d.glossary && d.glossary.length) chip("glossary applied: " + d.glossary.join(", "));
  if (d.typed) chip("typed");
  if (d.song) chip(`syllables fit the notes on ${d.song.exact} of ${d.song.total} lines`, d.song.exact === d.song.total ? "" : "warn");
  chip(d.translator === "llm" ? "LLM translation" : "plain MT");
  if (d.voice) chip("voice: " + d.voice);
  if (check.error) chip("read-back failed: " + check.error.slice(0, 80), "warn");
  else if (check.level) chip(`read-back: ${check.level} (similarity ${check.similarity})`, check.level === "ok" ? "" : "warn");
  for (const reason of check.reasons || []) chip(reason, "warn");
  const t = d.timings_ms || {};
  chip(`stt ${t.stt ?? "-"} | tone ${t.tone ?? "-"} | translate ${t.translate ?? "-"} | tts ${t.tts ?? "-"} | server ${t.total ?? "-"} ms`);
  if (d.roundtrip_ms) chip(`round trip ${d.roundtrip_ms} ms`);
  for (const w of d.warnings || []) chip(w, "warn");
  el.append(chips);

  // Hear2Act's point: what the delivery says should change what the system does. When the partner
  // sounded hesitant, or the message carries numbers, offer a one-tap confirmation request.
  if (who === "them" && ((tone.labels && tone.labels.hesitant) || guard.status !== "none")) {
    const button = document.createElement("button");
    button.className = "ghost small";
    button.textContent = "Ask them to confirm";
    button.onclick = () => {
      button.disabled = true;
      say("Sorry, could you confirm that once more? I want to be sure I understood.", "en-IN", d.src);
    };
    el.append(button);
  }
}

// ---------------------------------------------------------------- playback queue
// Segments finish out of order, so each sender's segments are played by sequence number.

const inbox = { sid: null, next: 0, items: new Map(), stalledAt: 0 };
let current = null;

function enqueue(msg) {
  if (msg.sid !== inbox.sid) { inbox.sid = msg.sid; inbox.next = 0; inbox.items.clear(); }
  inbox.items.set(msg.seq, msg);
  pump();
}

function pump() {
  if (current || state.listening) return; // half-duplex: hold playback while you are talking
  const m = inbox.items.get(inbox.next);
  if (!m) return;
  inbox.items.delete(inbox.next);
  inbox.next++;
  if (m.skip || !m.audio) return pump();
  const audio = new Audio("data:audio/wav;base64," + m.audio);
  current = audio;
  const done = () => { if (current === audio) { current = null; pump(); } };
  audio.onended = done;
  audio.onerror = done;
  audio.play().catch(done);
}

function stopPlayback() {
  if (current) { const a = current; current = null; a.pause(); }
}

setInterval(() => { // do not wait forever on a segment that never arrives
  if (!current && inbox.items.size && !inbox.items.has(inbox.next)) {
    if (!inbox.stalledAt) inbox.stalledAt = Date.now();
    else if (Date.now() - inbox.stalledAt > 8000) {
      inbox.next = Math.min(...inbox.items.keys());
      inbox.stalledAt = 0;
      pump();
    }
  } else {
    inbox.stalledAt = 0;
  }
}, 1000);

// ---------------------------------------------------------------- room

function setPeer(connected, lang) {
  state.peer = { connected, lang };
  if (lang) $("theirs").value = lang;
  $("peerStatus").textContent = connected
    ? `Partner online${lang ? ", speaking " + name(lang) : ""}. Your translations go to them.`
    : "Waiting for a partner. Until they join, translations play on this device (solo mode).";
}

function sendHello() {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ type: "hello", lang: $("mine").value }));
  }
}

function connectWs() {
  if (state.ws) { state.ws.onclose = null; state.ws.close(); }
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${scheme}://${location.host}/ws/${encodeURIComponent(state.room)}/${state.role}`);
  state.ws = ws;
  ws.onopen = () => { state.joined = true; sendHello(); setPeer(false, null); };
  ws.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if (m.type === "peer") setPeer(m.connected, m.lang);
    else if (m.type === "skip") enqueue({ sid: m.sid, seq: m.seq, skip: true });
    else if (m.type === "utterance") {
      const entry = document.createElement("article");
      entry.className = "entry them";
      renderEntry(entry, "them", m);
      $("log").prepend(entry);
      enqueue(m);
    }
  };
  ws.onclose = () => {
    if (state.ws !== ws) return;
    state.joined = false;
    setPeer(false, null);
    setTimeout(connectWs, 2000);
  };
}

function join() {
  // Browsers only allow autoplay after a click, so unlock audio here.
  new Audio("data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=").play().catch(() => {});
  state.room = $("room").value.trim() || "room" + Math.floor(Math.random() * 1000);
  $("room").value = state.room;
  state.role = $("role").value;
  connectWs();
  const other = state.role === "A" ? "B" : "A";
  const link = `${location.origin}/?room=${encodeURIComponent(state.room)}&role=${other}`;
  $("share").hidden = false;
  $("share").textContent = "Partner link: " + link;
}

// ---------------------------------------------------------------- voice clone (beta)

function renderVoice() {
  const has = Boolean(state.voiceId);
  $("clearVoice").hidden = !has;
  $("voiceStatus").textContent = has
    ? `Using your cloned voice (${state.voiceId}).`
    : "No cloned voice. Stock voices are used.";
}

async function recordVoice() {
  $("recordVoice").disabled = true;
  const chunks = [];
  let m;
  try {
    m = await startMic((c) => chunks.push(c));
  } catch (e) {
    $("voiceStatus").textContent = "Microphone unavailable: " + e.message;
    $("recordVoice").disabled = false;
    return;
  }
  for (let s = 12; s > 0; s--) {
    $("voiceStatus").textContent = `Recording... keep talking in ${name($("mine").value)} (${s}s)`;
    await sleep(1000);
  }
  stopMic(m);
  $("voiceStatus").textContent = "Creating your voice...";
  const form = new FormData();
  form.append("file", toWav16k(chunks, m.sampleRate), "reference.wav");
  form.append("name", `${state.room || "solo"}-${state.role}-${state.sid}`);
  form.append("language", $("mine").value);
  try {
    const r = await fetch("/api/voice/create", { method: "POST", body: form });
    const body = await r.json();
    if (!r.ok) throw new Error(JSON.stringify(body.detail));
    state.voiceId = body.voice_id;
    store.set("voiceId", state.voiceId);
  } catch (e) {
    $("voiceStatus").textContent = "Voice creation failed: " + e.message;
    $("recordVoice").disabled = false;
    return;
  }
  renderVoice();
  $("recordVoice").disabled = false;
}

async function clearVoice() {
  const id = state.voiceId;
  state.voiceId = "";
  store.set("voiceId", "");
  renderVoice();
  try { await fetch("/api/voice/" + encodeURIComponent(id), { method: "DELETE" }); } catch { /* best effort */ }
}

// ---------------------------------------------------------------- song mode
// Hum or sing -> list of likely songs -> pick one -> say or tap a language -> hear it sung back.

const song = { phrases: [], candidates: [], chosen: null, transcript: "" };
let songRecording = null;
const setSong = (t) => { $("songStatus").textContent = t; };

// Records until a pause after sound (or maxMs); `ref.stop` lets a button end it early.
async function recordPhrase({ maxMs = 15000, endSilenceMs = 1400, minMs = 1200, raw = false, meter = null, ref = {} }) {
  const chunks = [];
  let m = null, heardMs = 0, quietMs = 0, totalMs = 0, noise = 0.004, finish;
  const done = new Promise((resolve) => { finish = resolve; });
  ref.stop = finish;
  m = await startMic((f32) => {
    if (!m) return;
    const ms = (f32.length / m.sampleRate) * 1000;
    let sum = 0;
    for (let i = 0; i < f32.length; i++) sum += f32[i] * f32[i];
    const rms = Math.sqrt(sum / f32.length);
    if (meter) meter.style.width = Math.min(100, rms * 600) + "%";
    chunks.push(f32);
    totalMs += ms;
    if (rms > Math.max(0.008, noise * 3)) { heardMs += ms; quietMs = 0; } else { quietMs += ms; noise = noise * 0.9 + rms * 0.1; }
    if (totalMs >= maxMs || (heardMs >= minMs && quietMs >= endSilenceMs)) finish();
  }, raw);
  await done;
  stopMic(m);
  if (meter) meter.style.width = "0";
  return heardMs < 300 ? null : toWav16k(chunks, m.sampleRate);
}

async function playPrompt(key) {
  try {
    const r = await fetch("/api/prompt", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key, lang: $("mine").value }),
    });
    if (!r.ok) return;
    const body = await r.json();
    await new Promise((resolve) => {
      const a = new Audio("data:audio/wav;base64," + body.audio);
      a.onended = resolve; a.onerror = resolve;
      a.play().catch(resolve);
    });
  } catch { /* the spoken prompt is optional */ }
}

async function songHum() {
  if (songRecording) { if (songRecording.stop) songRecording.stop(); return; }
  stopPlayback();
  const ref = {};
  songRecording = ref;
  $("songRec").textContent = "Stop";
  setSong("Listening: hum or sing, then pause.");
  let wav = null;
  try {
    if ($("spoken").checked) await playPrompt("hum");
    wav = await recordPhrase({ raw: true, meter: $("songLevel"), ref });
  } catch (e) {
    setSong("Microphone unavailable: " + e.message);
  }
  songRecording = null;
  $("songRec").textContent = "Hum or sing";
  if (!wav) { if (!$("songStatus").textContent.startsWith("Microphone")) setSong("I did not hear anything. Try again, a little louder."); return; }
  setSong("Working out which song it is...");
  const form = new FormData();
  form.append("file", wav, "hum.wav");
  try {
    const r = await fetch("/api/song/identify", { method: "POST", body: form });
    const body = await r.json();
    if (!r.ok) throw new Error(JSON.stringify(body.detail));
    showCandidates(body);
  } catch (e) {
    setSong("Could not analyse that: " + e.message);
  }
}

function chooseCandidate(value) {
  song.chosen = value === "none" ? null : song.candidates[Number(value)];
  $("lyrics").value = song.chosen ? song.chosen.lines.join("\n") : song.transcript;
}

function showCandidates(body) {
  Object.assign(song, { phrases: body.phrases, candidates: body.candidates, chosen: null, transcript: body.transcript });
  $("songPick").hidden = false;
  $("songHeard").textContent =
    (body.transcript ? `I heard the words "${body.transcript}". ` : "I heard a tune without words. ") +
    `${body.note_count} notes in ${body.phrases.length} phrase(s). ${body.warnings.join(" ")}`;
  const box = $("songCands");
  box.replaceChildren();
  const add = (value, label) => {
    const row = document.createElement("label");
    row.className = "cand";
    const input = document.createElement("input");
    input.type = "radio"; input.name = "cand"; input.value = value;
    input.onchange = () => chooseCandidate(value);
    row.append(input, document.createTextNode(label));
    box.append(row);
  };
  body.candidates.forEach((c, i) => add(String(i), `${c.title}  (${c.score}: ${c.why}; ${c.source})`));
  add("none", "None of these: use my own words and tune");
  const first = box.querySelector("input");
  first.checked = true;
  chooseCandidate(first.value);
  setSong(body.candidates.length ? "Is it one of these? Then pick a language below, or say it." : "No known song matched, so I will use your own words and tune.");
  if ($("spoken").checked) askLanguage();
}

async function askLanguage() {
  setSong("Which language should I sing it in?");
  await playPrompt("language");
  let wav = null;
  try { wav = await recordPhrase({ maxMs: 6000, endSilenceMs: 900, minMs: 400 }); } catch { return; }
  if (!wav) { setSong("I did not hear a language. Tap one below."); return; }
  const form = new FormData();
  form.append("file", wav, "language.wav");
  try {
    const r = await fetch("/api/song/language", { method: "POST", body: form });
    const body = await r.json();
    if (!r.ok) throw new Error(JSON.stringify(body.detail));
    if (body.lang) singIn(body.lang);
    else setSong(`I heard "${body.heard}", which is not a language I know. Tap one below.`);
  } catch (e) {
    setSong("Could not hear the language: " + e.message);
  }
}

async function singIn(lang) {
  const lines = $("lyrics").value.split("\n").map((s) => s.trim()).filter(Boolean);
  const melody = song.chosen ? song.chosen.melody : song.phrases;
  if (!lines.length) { setSong("Add some lyrics first."); return; }
  if (!melody.length) { setSong("I need a tune. Hum or sing first."); return; }
  const seq = state.seq++;
  setSong(`Singing it in ${name(lang)}...`);
  await submit("/api/song/convert", {
    lines, melody, src: song.chosen ? song.chosen.language : $("mine").value, tgt: lang,
    title: song.chosen ? song.chosen.title : "", role: state.role, room: state.joined ? state.room : "",
    seq, sid: state.sid, voice_id: state.voiceId,
  }, seq);
  setSong("");
}

async function saveSong() {
  const title = window.prompt("Name this song");
  if (!title) return;
  const lines = $("lyrics").value.split("\n").map((s) => s.trim()).filter(Boolean);
  try {
    const r = await fetch("/api/song/save", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, lines, melody: song.chosen ? song.chosen.melody : song.phrases, language: $("mine").value }),
    });
    const body = await r.json();
    if (!r.ok) throw new Error(JSON.stringify(body.detail));
    setSong(`Saved. Next time you hum it, "${title}" will be on the list.`);
  } catch (e) {
    setSong("Could not save: " + e.message);
  }
}

// ---------------------------------------------------------------- boot

async function boot() {
  const cfg = await (await fetch("/api/config")).json();
  state.languages = cfg.languages;
  $("mockBadge").hidden = !cfg.mock;
  for (const id of ["mine", "theirs"]) {
    for (const [code, label] of Object.entries(cfg.languages)) $(id).add(new Option(label, code));
  }
  $("mine").value = "ta-IN";
  $("theirs").value = "hi-IN";
  const params = new URLSearchParams(location.search);
  $("room").value = params.get("room") || "room" + Math.floor(Math.random() * 1000);
  $("role").value = params.get("role") === "B" ? "B" : "A";
  if (params.get("role") === "B") { $("mine").value = "hi-IN"; $("theirs").value = "ta-IN"; }

  $("glossary").value = store.get("glossary");
  $("glossary").oninput = () => store.set("glossary", $("glossary").value.trim() ? $("glossary").value : "");
  const sendTyped = () => {
    const text = $("typed").value.trim();
    if (!text) return;
    $("typed").value = "";
    say(text, $("mine").value, $("theirs").value);
  };
  $("typedSend").onclick = sendTyped;
  $("typed").onkeydown = (e) => { if (e.key === "Enter") sendTyped(); };

  for (const [code, label] of Object.entries(cfg.languages)) {
    const b = document.createElement("button");
    b.textContent = label;
    b.onclick = () => singIn(code);
    $("songLangs").append(b);
  }
  $("songRec").onclick = songHum;
  $("songSay").onclick = askLanguage;
  $("songSave").onclick = saveSong;

  $("join").onclick = join;
  $("mine").onchange = sendHello;
  $("mic").onclick = () => (state.listening ? stopListening() : startListening());
  $("recordVoice").onclick = recordVoice;
  $("clearVoice").onclick = clearVoice;
  renderVoice();
}

boot();
