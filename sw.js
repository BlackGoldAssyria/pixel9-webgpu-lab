// Prefer new same-origin app files online, keep the last good shell offline.
// External model/CDN requests are handled by the browser/Transformers.js.
const CACHE = 'pixel9-local-ai-lab-shell-v2-4';
const PREFIX = 'pixel9-local-ai-lab-shell-';
const SHELL = ['./', './index.html', './app.js', './manifest.webmanifest', './icon.png', './icon.svg'];
self.addEventListener('install', event => {
  event.waitUntil(caches.open(CACHE)
    .then(cache => cache.addAll(SHELL.map(path => new Request(path, {cache: 'reload'}))))
    .then(() => self.skipWaiting()));
});
self.addEventListener('activate', event => {
  event.waitUntil(caches.keys()
    .then(keys => Promise.all(keys.filter(k => k.startsWith(PREFIX) && k !== CACHE).map(k => caches.delete(k))))
    .then(() => self.clients.claim()));
});
self.addEventListener('fetch', event => {
  const request = event.request;
  if (request.method !== 'GET' || new URL(request.url).origin !== self.location.origin) return;
  event.respondWith((async () => {
    const cache = await caches.open(CACHE);
    try {
      const response = await fetch(request, {cache: 'no-cache'});
      if (response.ok) { try { await cache.put(request, response.clone()); } catch (_) { /* Online still works if cache is full. */ } }
      return response;
    } catch (error) {
      const cached = await caches.match(request);
      if (cached) return cached;
      if (request.mode === 'navigate') return (await caches.match('./index.html')) || Response.error();
      throw error;
    }
  })());
});
