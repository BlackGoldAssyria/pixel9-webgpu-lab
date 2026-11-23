# Pixel 9 Local AI Lab (GitHub Pages PWA)

This is a static, browser-based GPU and Transformers.js benchmark. No backend, build pipeline or Python installation is needed. Files must remain in the same folder.

## Publish

1. Create a **public** repository named `pixel9-webgpu-lab` in your own GitHub account. Do not reuse an internal/company repository.
2. Upload the six website files from this folder to the **root** of the repository (not the ZIP file): `index.html`, `app.js`, `sw.js`, `manifest.webmanifest`, `icon.png`, `icon-512.png`. `README.md` is optional.
3. In **Settings > Pages**, choose **Deploy from a branch**, branch `main`, folder `/ (root)`, then Save.
4. Open `https://YOUR-USERNAME.github.io/pixel9-webgpu-lab/` on Android Chrome over Wi-Fi.

## First PWA test

1. Confirm `Secure origin: Yes`, `WebGPU: Available` and `PWA / offline shell: Shell active` (reload once if it initially says `Shell pending`).
2. From Chrome's menu, choose **Install app** or **Add to Home screen** if offered. Launch it from the home screen. Check the PWA status says `Installed`.
3. While online, load the page once fully. Turn on airplane mode, reopen the installed app and confirm the interface and GPU inspection still work. **Do not expect model downloads or model inference to work offline yet.**
4. Reconnect to Wi-Fi and run cautious memory tests and small-model inference. Export results before leaving, because model records are currently session-only. Avoid confidential prompts.

## Notes

- Only the same-origin application shell is precached by `sw.js`; third-party Transformers.js modules and model weights are *not* made reliably available offline by this package.
- The service worker uses cache-first shell assets. When publishing a new version, update the version string in `sw.js`, reload once to install the updated worker, and reload again if necessary to see the new code.
- In a GitHub Pages project path, all file references, manifest start URL and service worker registration deliberately use `./` relative paths.
- GPU memory allocation tests can crash the browser tab. Start at 256 MiB and only increase after successful runs.
- Do not commit credentials, user data, certificates, private models or company documents to a public repository.