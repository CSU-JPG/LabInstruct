#!/usr/bin/env node
/**
 * LabInstruct local evaluation server
 *
 * Usage:   node server.mjs [port]       port defaults to 8000
 * Open:    http://localhost:8000/human_eval/survey.html
 *
 * Ratings are written to human_eval/ratings.json, so evaluation can be resumed
 * on another machine by starting this server again.
 *
 * Listens on 127.0.0.1 only; never exposed to the network.
 * Opening survey.html directly over file:// falls back to browser localStorage.
 */
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..'); // served as the static root
const RATINGS_FILE = path.join(__dirname, 'ratings.json');

const _args = process.argv.slice(2);
const OPEN = _args.includes('--open');
const portArg = _args.find((a) => !a.startsWith('-'));
const PORT = Number(process.env.PORT) || Number(portArg) || 8000;
const HOST = '127.0.0.1';
const MAX_BODY = 32 * 1024 * 1024; // max PUT body size

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.csv': 'text/csv; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8',
  '.mp4': 'video/mp4',
  '.webm': 'video/webm',
  '.mov': 'video/quicktime',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.svg': 'image/svg+xml',
};

function sendJson(res, code, obj) {
  res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
  res.end(JSON.stringify(obj));
}

// ---------- ratings.json read/write (atomic, writes serialised) ----------
async function readRatings() {
  try {
    const raw = await fs.promises.readFile(RATINGS_FILE, 'utf8');
    const data = JSON.parse(raw);
    return (data && typeof data === 'object' && !Array.isArray(data)) ? data : {};
  } catch {
    return {}; // missing or corrupt -> treat as empty
  }
}

let writeChain = Promise.resolve();
function writeRatings(raw) {
  // Write the client's JSON string verbatim to keep its layout; never reformat it.
  const task = writeChain.then(async () => {
    const tmp = RATINGS_FILE + '.tmp';
    await fs.promises.writeFile(tmp, raw, 'utf8');
    await fs.promises.rename(tmp, RATINGS_FILE);
  });
  writeChain = task.catch(() => {}); // keep later writes going after a failure
  return task;
}

// ---------- static files (path-traversal guarded) ----------
function safePath(urlPath) {
  let decoded;
  try { decoded = decodeURIComponent(urlPath); } catch { return null; }
  const abs = path.resolve(ROOT, '.' + decoded);
  return (abs === ROOT || abs.startsWith(ROOT + path.sep)) ? abs : null;
}

function serveStatic(method, urlPath, res) {
  if (urlPath === '/' || urlPath === '') {
    res.writeHead(302, { Location: '/human_eval/survey.html' });
    res.end();
    return;
  }
  const file = safePath(urlPath);
  if (!file) { res.writeHead(403); res.end('Forbidden'); return; }
  fs.stat(file, (err, st) => {
    if (err || !st.isFile()) { res.writeHead(404); res.end('Not found'); return; }
    const ext = path.extname(file).toLowerCase();
    res.writeHead(200, { 'Content-Type': MIME[ext] || 'application/octet-stream', 'Cache-Control': 'no-store' });
    if (method === 'HEAD') { res.end(); return; }
    fs.createReadStream(file).pipe(res);
  });
}

const server = http.createServer(async (req, res) => {
  let url;
  try { url = new URL(req.url, `http://${req.headers.host || 'localhost'}`); }
  catch { res.writeHead(400); res.end('Bad request'); return; }
  const p = url.pathname;

  // API: probe / read ratings
  if (req.method === 'GET' && p === '/__api__/ping') return sendJson(res, 200, { ok: true });
  if (req.method === 'GET' && p === '/__api__/ratings') return sendJson(res, 200, await readRatings());

  // API: write ratings
  if (req.method === 'PUT' && p === '/__api__/ratings') {
    const chunks = [];
    let size = 0;
    for await (const chunk of req) {
      size += chunk.length;
      if (size > MAX_BODY) { res.writeHead(413); res.end('Too large'); return; }
      chunks.push(chunk);
    }
    const body = Buffer.concat(chunks).toString('utf8');
    let data;
    try { data = JSON.parse(body); }
    catch { return sendJson(res, 400, { ok: false, error: 'invalid json' }); }
    if (!data || typeof data !== 'object' || Array.isArray(data)) {
      return sendJson(res, 400, { ok: false, error: 'invalid payload' });
    }
    try {
      await writeRatings(body);
      return sendJson(res, 200, { ok: true });
    } catch (e) {
      console.error('write ratings failed:', e);
      return sendJson(res, 500, { ok: false, error: String((e && e.message) || e) });
    }
  }

  // static files
  if (req.method === 'GET' || req.method === 'HEAD') return serveStatic(req.method, p, res);
  res.writeHead(405); res.end('Method not allowed');
});

function openBrowser(url) {
  // cross-platform browser open: start / open / xdg-open
  try {
    if (process.platform === 'darwin') spawn('open', [url], { stdio: 'ignore', detached: true }).unref();
    else if (process.platform === 'win32') spawn('cmd', ['/c', 'start', '', url], { stdio: 'ignore', detached: true }).unref();
    else spawn('xdg-open', [url], { stdio: 'ignore', detached: true }).unref();
  } catch (e) { console.log('  (could not open a browser; visit ' + url + '）'); }
}

server.on('error', (err) => {
  if (err && err.code === 'EADDRINUSE') {
    console.error('  port ' + PORT + ' is already in use - the server may already be running.');
    console.error('  Open http://localhost:' + PORT + '/human_eval/survey.html, or close the old window and retry.');
    if (OPEN) openBrowser('http://localhost:' + PORT + '/human_eval/survey.html');
  } else {
    console.error('  server failed to start:', err);
  }
  process.exit(1);
});

server.listen(PORT, HOST, () => {
  console.log('');
  console.log('  LabInstruct local evaluation server started');
  console.log('  Open:      http://localhost:' + PORT + '/human_eval/survey.html');
  console.log('  Ratings:   ' + RATINGS_FILE);
  console.log('  Stop:      close this window or press Ctrl+C');
  console.log('');
  if (OPEN) openBrowser('http://localhost:' + PORT + '/human_eval/survey.html');
});
