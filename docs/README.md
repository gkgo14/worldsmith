# Project page source

This folder is the source of the [WorldSmith project page](https://gkgo14.github.io/worldsmith/), served by GitHub
Pages from `main` / `docs`. It is a static site with no build step and no external requests (fonts are
self-hosted in `assets/fonts`).

```
index.html              page structure and all text
assets/css/style.css    night palette, one colour per task, hand-drawn frames, tickets, wheel
assets/js/main.js       fireworks sky, string lights, marquee bulbs, loop wheel, light switch,
                        flip-book cast, wipe sliders, film canvases, bars, lightbox
assets/img, assets/video   50 media files (images and videos)
assets/fonts            Fraunces (display), Figtree (text), JetBrains Mono (labels); all OFL
googlee9807486bfb22992.html   Google Search Console ownership file; keep it, or verification lapses
```

## Before going public
- `index.html`: the Paper and arXiv buttons. Replace `href="#"` and remove the `soon` class
  and the `<span class="tag">soon</span>` on each. The Code button already links to this repository.
- No author names appear anywhere on purpose (hero and BibTeX).

## Things that depend on the media
- Reel chapters: `data-s` / `data-e` on the `.chap` buttons (seconds in `reel.mp4`).
- `reel.mp4`: `demo/worldsmith_demo_v6.mp4` re-encoded at 1920×1080 with its audio (x264 crf 20, AAC 160k, ~63 MB),
  because GitHub rejects files over 100 MB and the original is 126 MB.
  The player (seek bar, volume, speed, PiP, download, full screen) is in `main.js`; the reel starts muted so it can autoplay.
- Flip-book cast: each `cast_<name>.jpg` is a strip of 6 equal frames; the CSS steps through them.
- Films: `film_90X.mp4` is [input | ours | orbit]; the page draws the first two panels into the wipe.
- Emission and scene images are shown no larger than their native size (300 px and 508 px).
