// Only cache same-origin application shell. Remote model files and CDN/WASM assets
// are managed by browser/Transformers.js caches; offline model use is not guaranteed.
const CACHE='pixel9-local-ai-lab-shell-v2-2';
const SHELL=['./','./index.html','./app.js','./manifest.webmanifest','./icon.png','./icon.svg'];
self.addEventListener('install',event=>{event.waitUntil(caches.open(CACHE).then(cache=>cache.addAll(SHELL)));self.skipWaiting();});
self.addEventListener('activate',event=>{event.waitUntil(caches.keys().then(keys=>Promise.all(keys.filter(k=>k!==CACHE).map(k=>caches.delete(k)))));self.clients.claim();});
self.addEventListener('fetch',event=>{
  const url=new URL(event.request.url);
  if(event.request.method!=='GET'||url.origin!==self.location.origin)return;
  event.respondWith(caches.match(event.request).then(hit=>hit||fetch(event.request)));
});