# Pixel 9 Local AI Lab

A static, browser-based WebGPU memory and Transformers.js inference benchmark. No Python server, build system, or Android APK required.

## Publish with GitHub Pages

1. Open **Settings > Pages** in this repository.
2. Under **Build and deployment**, set source to **Deploy from a branch**, branch **main**, folder **/ (root)**, and save.
3. When GitHub finishes publishing, visit https://blackgoldassyria.github.io/pixel9-webgpu-lab/ in Chrome on the Pixel 9.

The app is already in the repository root: `index.html`, `app.js`, `sw.js`, `manifest.webmanifest`, `icon.png` and `icon.svg`. The manifest and URLs are relative so they work under a project path.

## First test

1. Load the page over Wi-Fi. Check that **Secure origin** says **Yes** and **WebGPU** says **Available**.
2. Reload if the PWA indicator initially reports **Shell pending**. It should report **Shell active** once the service worker controls the page.
3. In Chrome, use **Install app** or **Add to Home screen** if offered. Launch from the icon; the PWA indicator should say **Installed**.
4. Reopen in airplane mode. The offline application interface should load, but remote Transformers.js dependencies, WASM and model weights are **not guaranteed to work offline**.
5. Reconnect to Wi-Fi and start with GPU inspection or the smallest model. Run the 256 MiB memory test before trying more demanding sizes. Export results before reloading.

## Boundaries

- Inference is on-device after the model is downloaded, but libraries and models are retrieved from third-party servers. Do not enter confidential prompts.
- Benchmark results are session-only; only memory progress checkpoints persist in localStorage.
- The memory probe holds 32 MiB GPU storage buffers cumulatively, clears each on the GPU, and verifies five nonzero samples via readback. It does not measure total GPU VRAM or validate custom compute shaders.
- The service worker caches only the application shell. For updates, bump its CACHE version, reload to install it, and reload again if necessary.
- Running larger memory tests can terminate the Chrome tab or PWA. Large models can fail to load.
- The `icon.svg` is a scalable installed-app icon; `icon.png` supplies a 192px fallback. Exact Chrome install behavior should be verified on the Pixel.
