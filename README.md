# WorldSmith project page — night carnival edition

Static site, no build step, no external requests (fonts are self-hosted in `assets/fonts`).
The earlier neutral version is in this repo's history (commit 34c6242).

```
index.html              page structure and all text
assets/css/style.css    night palette, one colour per task, hand-drawn frames, tickets, wheel
assets/js/main.js       fireworks sky, string lights, marquee bulbs, loop wheel, light switch,
                        flip-book cast, wipe sliders, film canvases, bars, lightbox
assets/img, assets/video   50 media files (images and videos)
assets/fonts            Fraunces (display), Figtree (text), JetBrains Mono (labels); all OFL
```

## Deploy on this server
`bash /root/worldsmith/deploy.sh` copies this folder to `/var/www/worldsmith`, which nginx serves on port 80
(config: `/etc/nginx/sites-available/worldsmith`).

## Before going public
- `index.html`: the three buttons (Paper / arXiv / Code). Replace `href="#"` and remove the `soon` class
  and the `<span class="tag">soon</span>` on each.
- No author names appear anywhere on purpose (hero and BibTeX).

## Things that depend on the media
- Reel chapters: `data-s` / `data-e` on the `.chap` buttons (seconds in `reel.mp4`).
- `reel.mp4`: `demo/worldsmith_demo_v6.mp4` re-encoded at 1920×1080 with its audio (x264 crf 20, AAC 160k, ~63 MB),
  because GitHub rejects files over 100 MB and the original is 126 MB. The server copy serves the original.
  The player (seek bar, volume, speed, PiP, download, full screen) is in `main.js`; the reel starts muted so it can autoplay.
- Flip-book cast: each `cast_<name>.jpg` is a strip of 6 equal frames; the CSS steps through them.
- Films: `film_90X.mp4` is [input | ours | orbit]; the page draws the first two panels into the wipe.
- Emission and scene images are shown no larger than their native size (300 px and 508 px).
