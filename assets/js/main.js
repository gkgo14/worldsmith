/* WorldSmith project page — night carnival edition. No dependencies. */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const NS = 'http://www.w3.org/2000/svg';
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const PALETTE = ['--lamp', '--c-edit', '--c-s4d', '--c-anim', '--c-emit', '--c-s3d'].map(css);
  const rnd = (a, b) => a + Math.random() * (b - a);
  // small seeded generator so the hand-drawn frames keep their shape between redraws
  const seeded = s => () => (s = (s * 16807) % 2147483647) / 2147483647;

  /* ---------- headline metric bars (numbers from the paper's tables) ---------- */
  const BARS = {
    edit: { title: 'Instruction following (IF)', note: 'EditHero · higher is better', rows: [
      ['PartFlow', .118], ['3DEditFormer', .191], ['Nano3D', .309], ['VoxHammer', .32],
      ['Alchemy3D', .410], ['Codex, same model', .619], ['WorldSmith', .638, 1]] },
    emit: { title: 'Emission IoU', note: 'LightGenBench · higher is better', rows: [
      ['TEXGen-Emission', .092], ['Hunyuan3D-Emission', .117], ['SegviGen-Emission', .147],
      ['TRELLIS.2-Emission', .271], ['WorldSmith', .339, 1]] },
    anim: { title: 'F-score @ 0.02', note: 'Motion-80 · higher is better', rows: [
      ['MoCapAnything V2', .0967], ['TopoCap', .1046], ['CAST', .1792], ['MorphGS', .2177], ['WorldSmith', .3686, 1]] },
    s3d: { title: 'Object-IoU (AABB)', note: 'CompoScene-3D · higher is better', rows: [
      ['SceneMaker', .112], ['SceneGen', .114], ['VIGA', .134], ['OVOW', .151], ['MIDI-3D', .187],
      ['Mira-Scene', .207], ['Same model, direct', .399], ['WorldSmith', .459, 1]] },
    s4d: { title: 'Object-IoU (AABB)', note: 'CompoScene-4D · higher is better', rows: [
      ['VIGA', .067], ['OVOW', .115], ['SceneMaker', .119], ['SceneGen', .123], ['MIDI-3D', .151],
      ['Mira-Scene', .152], ['WorldSmith', .340, 1]] },
  };
  $$('[data-bars]').forEach(el => {
    const d = BARS[el.dataset.bars]; if (!d) return;
    const max = Math.max(...d.rows.map(r => r[1]));
    el.innerHTML = `<div class="bars-head"><b>${d.title}</b><span>${d.note}</span></div>` +
      d.rows.map(([n, v, ours]) => `<div class="bar${ours ? ' ours' : ''}"><span class="name" title="${n}">${n}</span>` +
        `<span class="track"><span class="fill" data-w="${(v / max * 100).toFixed(1)}"></span></span>` +
        `<span class="val">${v < .1 ? v.toFixed(3) : v.toFixed(3).replace(/0$/, '')}</span></div>`).join('');
  });
  const fillBars = root => $$('.fill', root).forEach(f => { f.style.width = f.dataset.w + '%'; });

  /* ---------- geometry helper: points around a rounded rectangle ---------- */
  function perimeter(w, h, r, step) {
    r = Math.min(r, w / 2, h / 2);
    const pts = [], add = (x, y, nx, ny) => pts.push({ x, y, nx, ny });
    const line = (x0, y0, x1, y1, nx, ny) => {
      const len = Math.hypot(x1 - x0, y1 - y0), n = Math.max(1, Math.round(len / step));
      for (let i = 0; i < n; i++) add(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n, nx, ny);
    };
    const arc = (cx, cy, a0) => {
      const n = Math.max(2, Math.round(r * Math.PI / 2 / step));
      for (let i = 0; i < n; i++) { const a = a0 + Math.PI / 2 * i / n; add(cx + r * Math.cos(a), cy + r * Math.sin(a), Math.cos(a), Math.sin(a)); }
    };
    line(r, 0, w - r, 0, 0, -1); arc(w - r, r, -Math.PI / 2);
    line(w, r, w, h - r, 1, 0); arc(w - r, h - r, 0);
    line(w - r, h, r, h, 0, 1); arc(r, h - r, Math.PI / 2);
    line(0, h - r, 0, r, -1, 0); arc(r, r, Math.PI);
    return pts;
  }

  /* ---------- hand-drawn frames, like the panels of the teaser figure ---------- */
  function sketch(el, i) {
    const w = el.offsetWidth, h = el.offsetHeight; if (!w || !h) return;
    let svg = $(':scope > svg.sk-svg', el);
    if (!svg) { svg = document.createElementNS(NS, 'svg'); svg.setAttribute('class', 'sk-svg'); svg.setAttribute('aria-hidden', 'true'); el.prepend(svg); }
    svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
    const path = (seed, amp) => {
      const R = seeded(seed), pts = perimeter(w, h, 22, 46).map(p => {
        const j = (R() - .5) * 2 * amp;
        return [p.x + p.nx * j, p.y + p.ny * j];
      });
      const n = pts.length; let d = '';
      for (let k = 0; k < n; k++) {           // closed Catmull-Rom spline as cubic Béziers
        const p0 = pts[(k - 1 + n) % n], p1 = pts[k], p2 = pts[(k + 1) % n], p3 = pts[(k + 2) % n];
        if (!k) d += `M${p1[0].toFixed(1)} ${p1[1].toFixed(1)}`;
        d += `C${(p1[0] + (p2[0] - p0[0]) / 6).toFixed(1)} ${(p1[1] + (p2[1] - p0[1]) / 6).toFixed(1)} ${(p2[0] - (p3[0] - p1[0]) / 6).toFixed(1)} ${(p2[1] - (p3[1] - p1[1]) / 6).toFixed(1)} ${p2[0].toFixed(1)} ${p2[1].toFixed(1)}`;
      }
      return `<path d="${d}Z"/>`;
    };
    svg.innerHTML = path(101 + i * 37, 2.2) + path(977 + i * 53, 3.2);
  }
  const sks = $$('.sk');
  const ro = new ResizeObserver(es => es.forEach(e => sketch(e.target, sks.indexOf(e.target))));
  sks.forEach(el => ro.observe(el));

  /* ---------- marquee bulbs around the reel ---------- */
  $$('[data-bulbs]').forEach(el => {
    const svg = document.createElementNS(NS, 'svg'); svg.setAttribute('class', 'bulbs'); svg.setAttribute('aria-hidden', 'true'); el.prepend(svg);
    const draw = () => {
      const w = el.offsetWidth, h = el.offsetHeight, inset = w < 520 ? 6 : 9;
      const pts = perimeter(w - inset * 2, h - inset * 2, 18, w < 520 ? 20 : 26);
      svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
      svg.innerHTML = pts.map((p, i) => `<circle class="p${i % 3}" cx="${(p.x + inset).toFixed(1)}" cy="${(p.y + inset).toFixed(1)}" r="${w < 520 ? 2.2 : 3}"/>`).join('');
    };
    new ResizeObserver(draw).observe(el);
  });

  /* ---------- string lights ---------- */
  $$('[data-lights]').forEach(el => {
    const svg = document.createElementNS(NS, 'svg'); el.append(svg);
    const draw = () => {
      const w = el.offsetWidth, h = el.offsetHeight, small = el.classList.contains('small');
      const swags = Math.max(2, Math.round(w / (small ? 300 : 380))), sw = w / swags, sag = h * (small ? .62 : .7);
      let wire = '', bulbs = '', k = 0;
      for (let s = 0; s < swags; s++) {
        const x0 = s * sw, x1 = x0 + sw, n = Math.max(3, Math.round(sw / (small ? 44 : 52)));
        wire += `<path class="wire" d="M${x0} 0Q${x0 + sw / 2} ${sag * 2} ${x1} 0"/>`;
        for (let i = 1; i < n; i++, k++) {
          const t = i / n, x = x0 + sw * t, y = 2 * (1 - t) * t * sag * 2;   // point on the quadratic curve
          const c = PALETTE[k % PALETTE.length], r = small ? 3.6 : 4.8;
          bulbs += `<rect class="cap" x="${(x - 2).toFixed(1)}" y="${(y - 1).toFixed(1)}" width="4" height="5" rx="1"/>` +
            `<circle class="bulb" cx="${x.toFixed(1)}" cy="${(y + 4 + r).toFixed(1)}" r="${r}" style="--b:${c};--t:${(2.4 + (k * 7 % 5) * .5).toFixed(1)}s;--dl:-${(k * 13 % 10) * .3}s"/>`;
        }
      }
      svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
      svg.innerHTML = wire + bulbs;
    };
    let lw = 0;
    new ResizeObserver(() => { if (Math.abs(el.offsetWidth - lw) > 40) { lw = el.offsetWidth; draw(); } }).observe(el);
  });

  /* ---------- the big Ferris wheel behind the title ---------- */
  const hw = $('#heroWheel');
  if (hw) {
    const N = 16, R = 460; let spokes = '', cabs = '';
    for (let i = 0; i < N; i++) {
      const a = i / N * Math.PI * 2, x = 500 + R * Math.cos(a), y = 500 + R * Math.sin(a);
      const a2 = (i + 1) / N * Math.PI * 2, xi = 500 + 300 * Math.cos(a), yi = 500 + 300 * Math.sin(a), xj = 500 + 300 * Math.cos(a2), yj = 500 + 300 * Math.sin(a2);
      spokes += `<path class="hw-line" d="M500 500L${x.toFixed(1)} ${y.toFixed(1)}M${xi.toFixed(1)} ${yi.toFixed(1)}L${xj.toFixed(1)} ${yj.toFixed(1)}"/>`;
      cabs += `<g transform="translate(${x.toFixed(1)} ${y.toFixed(1)})"><g class="hw-cab"><circle class="hw-lamp" cx="0" cy="0" r="5"/><rect class="hw-car" style="--k:${PALETTE[i % PALETTE.length]}" x="-17" y="8" width="34" height="28" rx="9"/></g></g>`;
    }
    hw.innerHTML = `<svg viewBox="0 0 1000 1000"><path class="hw-leg" d="M500 500L330 1000M500 500L670 1000"/><g class="hw-spin"><circle class="hw-line rim" cx="500" cy="500" r="${R}"/><circle class="hw-line" cx="500" cy="500" r="430"/><circle class="hw-line" cx="500" cy="500" r="300"/>${spokes}${cabs}</g><circle cx="500" cy="500" r="26" fill="rgba(214,240,236,.4)"/></svg>`;
  }

  /* ---------- night sky: stars always, fireworks on a timer and wherever you click ---------- */
  const sky = $('#sky'), hero = $('.hero');
  if (sky && hero) {
    const ctx = sky.getContext('2d'); let W = 0, H = 0, dpr = 1, stars = [], parts = [], rockets = [], raf = null, visible = true, next = 1200;
    const size = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 2); W = hero.offsetWidth; H = hero.offsetHeight;
      sky.width = W * dpr; sky.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const R = seeded(7); stars = Array.from({ length: Math.round(W * H / 5200) }, () => ({ x: R() * W, y: R() * H * .9, r: .4 + R() * 1.1, p: R() * 6.28, s: .4 + R() * 1.4 }));
      if (reduced) frame(0);
    };
    const burst = (x, y) => {
      const c = PALETTE[Math.floor(Math.random() * PALETTE.length)], c2 = Math.random() < .5 ? '#fff3d6' : PALETTE[Math.floor(Math.random() * PALETTE.length)];
      const n = 70 + Math.floor(Math.random() * 40), sp = rnd(2.4, 4.4);
      for (let i = 0; i < n; i++) { const a = Math.PI * 2 * i / n + rnd(-.05, .05), v = sp * rnd(.55, 1); parts.push({ x, y, vx: Math.cos(a) * v, vy: Math.sin(a) * v, life: 1, d: rnd(.010, .018), c: i % 3 ? c : c2 }); }
    };
    const launch = (x, y) => { rockets.push({ x, y: H + 10, ty: y, vy: -rnd(9, 12) }); };
    let last = 0;
    function frame(t) {
      const dt = Math.min(40, t - last || 16) / 16.7; last = t;
      ctx.globalCompositeOperation = 'destination-out'; ctx.fillStyle = 'rgba(0,0,0,.22)'; ctx.fillRect(0, 0, W, H);
      ctx.globalCompositeOperation = 'lighter';
      // star alpha is scaled by .22 to match the fade above, so brightness settles instead of piling up
      for (const s of stars) { ctx.globalAlpha = reduced ? .7 : .22 * (.3 + .7 * (.5 + .5 * Math.sin(t / 1000 * s.s + s.p))); ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(s.x, s.y, s.r, 0, 6.28); ctx.fill(); }
      if (reduced) { ctx.globalAlpha = 1; return; }
      for (const r of rockets) { r.y += r.vy * dt; ctx.globalAlpha = .9; ctx.fillStyle = '#ffe3a6'; ctx.fillRect(r.x - 1, r.y, 2, 9); if (r.y <= r.ty) { burst(r.x, r.y); r.done = 1; } }
      rockets = rockets.filter(r => !r.done);
      for (const p of parts) { p.x += p.vx * dt; p.y += p.vy * dt; p.vy += .035 * dt; p.vx *= .985; p.vy *= .985; p.life -= p.d * dt; ctx.globalAlpha = Math.max(0, p.life); ctx.fillStyle = p.c; ctx.beginPath(); ctx.arc(p.x, p.y, 1.7, 0, 6.28); ctx.fill(); }
      parts = parts.filter(p => p.life > 0);
      ctx.globalAlpha = 1;
      next -= 16.7 * dt;
      if (next <= 0) { launch(rnd(W * .08, W * .92), rnd(H * .06, H * .34)); next = rnd(2200, 5200); }
      raf = visible && !document.hidden ? requestAnimationFrame(frame) : null;
    }
    if (window.matchMedia('(hover: none)').matches && $('#skyHint')) $('#skyHint').textContent = 'psst… tap the sky';
    const start = () => { if (!raf && !reduced && visible && !document.hidden) { last = 0; raf = requestAnimationFrame(frame); } };
    new ResizeObserver(size).observe(hero);
    new IntersectionObserver(([e]) => { visible = e.isIntersecting; start(); }).observe(hero);
    document.addEventListener('visibilitychange', start);
    hero.addEventListener('click', e => {
      if (reduced || e.target.closest('a, button, video, .reel')) return;
      const b = hero.getBoundingClientRect(); launch(e.clientX - b.left, e.clientY - b.top);
      $('#skyHint')?.classList.add('gone');
    });
  }

  /* ---------- lazy videos: load and play only while visible ---------- */
  const vio = new IntersectionObserver(es => es.forEach(e => {
    const v = e.target;
    if (e.isIntersecting) { if (!v.src && v.dataset.src) v.src = v.dataset.src; if (!reduced) v.play().catch(() => {}); }
    else v.pause();
  }), { threshold: .25 });
  $$('video[data-src]').forEach(v => vio.observe(v));

  /* ---------- reveal + bars on scroll ---------- */
  const rio = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    e.target.classList.add('in');
    if (e.target.matches('[data-bars]')) fillBars(e.target);
    rio.unobserve(e.target);
  }), { threshold: .15 });
  $$('.rv, [data-bars]').forEach(el => rio.observe(el));

  /* ---------- hero reel with chapters ---------- */
  /* player: one seek bar for the whole reel (coloured by chapter), volume, speed, PiP, download, full screen */
  const reel = $('#reelVideo'), chaps = $$('.chap'), player = $('#player');
  const svg = d => `<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">${d}</svg>`;
  const IC = {
    play: svg('<path d="M7 4l13 8-13 8z"/>'),
    pause: svg('<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>'),
    vol: svg('<path d="M4 9h4l5-4v14l-5-4H4z"/><path d="M16 8.5a5 5 0 0 1 0 7M18.5 6a8.5 8.5 0 0 1 0 12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>'),
    mute: svg('<path d="M4 9h4l5-4v14l-5-4H4z"/><path d="M16.5 9.5l5 5m0-5l-5 5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>'),
    pip: svg('<rect x="2.5" y="4.5" width="19" height="15" rx="2" fill="none" stroke="currentColor" stroke-width="2"/><rect x="12" y="11" width="7" height="6" rx="1"/>'),
    dl: svg('<path d="M12 4v11m-5-5l5 5 5-5M5 20h14" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>'),
    fs: svg('<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/>'),
    fsx: svg('<path d="M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/>'),
  };
  if (reel && player) {
    const playB = $('#pcPlay'), muteB = $('#pcMute'), vol = $('#pcVol'), rateB = $('#pcRate'), rates = $('#pcRates'),
      pipB = $('#pcPip'), fsB = $('#pcFs'), scrub = $('#scrub'), sIn = $('#scrubIn'), sFill = $('#scrubFill'),
      sBuf = $('#scrubBuf'), sTip = $('#scrubTip'), cur = $('#pcCur'), dur = $('#pcDur'), chapL = $('#pcChap');
    $('#pcDl').innerHTML = IC.dl; pipB.innerHTML = IC.pip;
    const D = () => reel.duration || 68;
    const fmt = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
    const chapAt = t => chaps.find(c => t >= +c.dataset.s && t < +c.dataset.e) || chaps[chaps.length - 1];

    // seek bar segments, one per chapter, in the chapter colour
    const drawSegs = () => {
      $('#scrubSegs').innerHTML = chaps.map(c => {
        const s = +c.dataset.s, e = Math.min(+c.dataset.e, D());
        return `<i style="left:${s / D() * 100}%;width:${(e - s) / D() * 100}%;${c.getAttribute('style')}"></i>`;
      }).join('');
      dur.textContent = fmt(D());
    };
    drawSegs(); reel.addEventListener('loadedmetadata', drawSegs);

    let seeking = false;
    const paint = () => {
      const t = reel.currentTime, p = t / D() * 100;
      sFill.style.width = p + '%';
      if (!seeking) sIn.value = Math.round(t / D() * 1000);
      sIn.setAttribute('aria-valuetext', `${fmt(t)} of ${fmt(D())}`);
      cur.textContent = fmt(t);
      const ch = chapAt(t); chapL.textContent = ch ? ch.firstChild.textContent.trim() : '';
      chapL.style.setProperty('--c', ch ? ch.style.getPropertyValue('--c') : '');
      chaps.forEach(c => {
        const s = +c.dataset.s, e = +c.dataset.e, on = c === ch;
        c.classList.toggle('on', on);
        c.querySelector('.p').style.width = (t >= e ? 100 : on ? (t - s) / (e - s) * 100 : 0) + '%';
      });
    };
    reel.addEventListener('timeupdate', paint);
    reel.addEventListener('progress', () => {
      const b = reel.buffered; if (b.length) sBuf.style.width = b.end(b.length - 1) / D() * 100 + '%';
    });
    sIn.addEventListener('input', () => { seeking = true; reel.currentTime = sIn.value / 1000 * D(); paint(); });
    sIn.addEventListener('change', () => { seeking = false; });
    const tip = e => {
      const r = scrub.getBoundingClientRect(), x = Math.min(Math.max(e.clientX - r.left, 0), r.width), t = x / r.width * D();
      const ch = chapAt(t);
      sTip.textContent = `${ch ? ch.firstChild.textContent.trim() + ' · ' : ''}${fmt(t)}`;
      sTip.style.left = x + 'px'; sTip.hidden = false;
    };
    scrub.addEventListener('pointermove', tip);
    scrub.addEventListener('pointerleave', () => { sTip.hidden = true; });

    // play / pause
    const syncPlay = () => { playB.innerHTML = reel.paused ? IC.play : IC.pause; playB.setAttribute('aria-label', reel.paused ? 'Play' : 'Pause'); };
    const toggle = () => { reel.paused ? reel.play().catch(() => {}) : reel.pause(); };
    reel.addEventListener('play', syncPlay); reel.addEventListener('pause', syncPlay);
    playB.addEventListener('click', toggle);
    reel.addEventListener('click', toggle);
    if (reduced) { reel.removeAttribute('autoplay'); reel.pause(); }
    syncPlay();
    chaps.forEach(c => c.addEventListener('click', () => { reel.currentTime = +c.dataset.s + .05; reel.play().catch(() => {}); }));

    // volume (the reel starts muted so it can autoplay)
    const syncVol = () => {
      const m = reel.muted || reel.volume === 0;
      muteB.innerHTML = m ? IC.mute : IC.vol; muteB.setAttribute('aria-label', m ? 'Unmute' : 'Mute');
      vol.value = m ? 0 : reel.volume; vol.style.setProperty('--v', (m ? 0 : reel.volume) * 100 + '%');
    };
    muteB.addEventListener('click', () => { if (reel.muted || reel.volume === 0) { reel.muted = false; if (!reel.volume) reel.volume = .8; } else reel.muted = true; });
    vol.addEventListener('input', () => { reel.volume = +vol.value; reel.muted = +vol.value === 0; });
    reel.addEventListener('volumechange', syncVol); syncVol();

    // speed
    const closeRates = () => { rates.hidden = true; rateB.setAttribute('aria-expanded', 'false'); };
    rateB.addEventListener('click', e => { e.stopPropagation(); rates.hidden = !rates.hidden; rateB.setAttribute('aria-expanded', String(!rates.hidden)); });
    $$('button', rates).forEach(b => b.addEventListener('click', () => {
      reel.playbackRate = +b.dataset.r; rateB.textContent = b.textContent;
      $$('button', rates).forEach(x => x.classList.toggle('on', x === b)); closeRates();
    }));
    document.addEventListener('click', e => { if (!e.target.closest('.pc-menu')) closeRates(); });

    // picture in picture
    if (document.pictureInPictureEnabled && !reel.disablePictureInPicture) {
      pipB.hidden = false;
      pipB.addEventListener('click', () => {
        (document.pictureInPictureElement ? document.exitPictureInPicture() : reel.requestPictureInPicture()).catch(() => {});
      });
    }

    // full screen: the whole player, so the seek bar and buttons come along
    const fsEl = () => document.fullscreenElement || document.webkitFullscreenElement;
    const syncFs = () => { const on = fsEl() === player; fsB.innerHTML = on ? IC.fsx : IC.fs; fsB.setAttribute('aria-label', on ? 'Exit full screen' : 'Full screen'); player.classList.toggle('is-fs', on); };
    fsB.addEventListener('click', () => {
      if (fsEl()) (document.exitFullscreen || document.webkitExitFullscreen).call(document);
      else if (player.requestFullscreen) player.requestFullscreen().catch(() => {});
      else if (player.webkitRequestFullscreen) player.webkitRequestFullscreen();
      else if (reel.webkitEnterFullscreen) reel.webkitEnterFullscreen();   // iPhone: native player only
    });
    document.addEventListener('fullscreenchange', syncFs); document.addEventListener('webkitfullscreenchange', syncFs); syncFs();
    reel.addEventListener('dblclick', () => fsB.click());

    // in full screen, hide the controls after a moment of stillness
    let idle = null;
    const wake = () => { player.classList.remove('idle'); clearTimeout(idle); if (player.classList.contains('is-fs') && !reel.paused) idle = setTimeout(() => player.classList.add('idle'), 2500); };
    player.addEventListener('pointermove', wake); reel.addEventListener('play', wake); reel.addEventListener('pause', wake);

    // keyboard, only while focus is inside the player
    player.addEventListener('keydown', e => {
      if (e.target.matches('input[type=range]') && /Arrow/.test(e.key)) return;
      const k = e.key.toLowerCase();
      if (k === ' ' || k === 'k') { if (e.target.matches('button, a')) return; toggle(); }
      else if (k === 'f') fsB.click();
      else if (k === 'm') muteB.click();
      else if (k === 'arrowright') reel.currentTime = Math.min(D(), reel.currentTime + 5);
      else if (k === 'arrowleft') reel.currentTime = Math.max(0, reel.currentTime - 5);
      else return;
      e.preventDefault(); wake();
    });
    paint();
  }

  /* ---------- the loop as a wheel: write → run → look → revise, round and round ---------- */
  const lwRoot = $('#loopWheel');
  if (lwRoot) {
    const lw = $('.lw', lwRoot), cabs = $$('.lw-cab', lw), steps = $$('.lw-step', lwRoot);
    let sp = '', dots = '';
    for (let i = 0; i < 8; i++) { const a = i * Math.PI / 4; sp += `<line x1="200" y1="200" x2="${(200 + 150 * Math.cos(a)).toFixed(1)}" y2="${(200 + 150 * Math.sin(a)).toFixed(1)}"/>`; }
    for (let i = 0; i < 16; i++) { const a = (i + .5) * Math.PI / 8; dots += `<circle cx="${(200 + 150 * Math.cos(a)).toFixed(1)}" cy="${(200 + 150 * Math.sin(a)).toFixed(1)}" r="3.4"/>`; }
    $('.lw-spokes', lw).innerHTML = sp; $('.lw-dots', lw).innerHTML = dots;
    new ResizeObserver(() => lw.style.setProperty('--lwpx', lw.offsetWidth * .375 + 'px')).observe(lw);
    let turn = 0, timer = null, hold = 0;
    const show = () => {
      const i = ((turn % 4) + 4) % 4;
      lw.style.setProperty('--rot', turn * 90 + 'deg');
      cabs.forEach((c, j) => c.classList.toggle('on', j === i));
      steps.forEach((s, j) => { s.classList.toggle('on', j === i); s.setAttribute('aria-current', j === i ? 'step' : 'false'); });
    };
    show();
    steps.forEach((s, j) => s.addEventListener('click', () => { const cur = ((turn % 4) + 4) % 4; turn += ((j - cur) + 4) % 4; hold = 3; show(); }));
    if (!reduced) new IntersectionObserver(([e]) => {
      if (e.isIntersecting && !timer) timer = setInterval(() => { if (hold > 0) { hold--; return; } turn++; show(); }, 2400);
      else if (!e.isIntersecting && timer) { clearInterval(timer); timer = null; }
    }, { threshold: .3 }).observe(lwRoot);
  }

  /* ---------- emission booth: flip the switch ---------- */
  const booth = $('#booth');
  if (booth) {
    const sw = $('#emitSwitch'), a = $('#emitIn'), b = $('#emitOut'), label = $('#emitLabel'), txt = $('.sw-text', sw), btns = $$('.gal-btn', booth);
    let touched = false;
    const set = on => {
      booth.classList.toggle('lit', on); sw.setAttribute('aria-checked', on);
      label.textContent = on ? 'emission' : 'input'; txt.textContent = on ? 'glowing' : 'flip the switch';
    };
    sw.addEventListener('click', () => { touched = true; set(sw.getAttribute('aria-checked') !== 'true'); });
    $('.booth-stage', booth).addEventListener('click', () => sw.click());
    btns.forEach(btn => btn.addEventListener('click', () => {
      touched = true; a.src = btn.dataset.a; b.src = btn.dataset.b;
      btns.forEach(x => x.classList.toggle('on', x === btn)); set(true);
    }));
    // the first time the booth scrolls into view, switch the lights on by itself
    new IntersectionObserver(([e], io) => {
      if (!e.isIntersecting) return; io.disconnect();
      setTimeout(() => { if (!touched) set(true); }, reduced ? 0 : 900);
    }, { threshold: .6 }).observe($('.booth-stage', booth));
  }

  /* ---------- wipe sliders ---------- */
  $$('.wipe').forEach(w => {
    const r = $('input[type=range]', w);
    r.addEventListener('input', () => w.style.setProperty('--x', r.value + '%'));
  });

  /* ---------- galleries: thumbnails swap the [data-a] / [data-b] images ---------- */
  $$('[data-gallery]').forEach(g => {
    const a = $('img[data-a]', g), b = $('img[data-b]', g), btns = $$('.gal-btn', g);
    btns.forEach(btn => btn.addEventListener('click', () => {
      if (a && btn.dataset.a) a.src = btn.dataset.a;
      b.src = btn.dataset.b;
      btns.forEach(x => x.classList.toggle('on', x === btn));
    }));
  });

  /* ---------- film: side-by-side video [input | ours | orbit]; the first two panels feed the wipe ---------- */
  const fv = $('#filmVideo'), cBase = $('#filmBase'), cTop = $('#filmTop');
  if (fv && cBase) {
    const ctx = [cBase, cTop].map(c => c.getContext('2d'));
    let raf = null, visible = false, cur = null;
    const paint = (src, h) => ctx.forEach((c, i) => c.drawImage(src, i * h, 0, h, h, 0, 0, 512, 512));   // each panel is square
    const draw = () => {
      if (fv.readyState >= 2) paint(fv, fv.videoHeight);
      raf = visible ? requestAnimationFrame(draw) : null;
    };
    const load = id => {
      cur = id;
      const poster = new Image(); poster.onload = () => { if (cur === id && fv.readyState < 2) paint(poster, poster.naturalHeight); };
      poster.src = `assets/video/film_${id}.jpg`;                 // show the still until the video arrives
      fv.src = `assets/video/film_${id}.mp4`;
      $$('.film-btn').forEach(b => b.classList.toggle('on', b.dataset.film == id));
      fv.addEventListener('loadeddata', () => { if (visible && !reduced) fv.play().catch(() => {}); }, { once: true });
    };
    $$('.film-btn').forEach(b => b.addEventListener('click', () => load(b.dataset.film)));
    new IntersectionObserver(([e]) => {
      visible = e.isIntersecting;
      if (visible) { if (!cur) load('905'); else if (!reduced) fv.play().catch(() => {}); if (!raf) draw(); }
      else fv.pause();
    }, { threshold: .2 }).observe($('#films'));
    if (reduced) $('#filmWipe').addEventListener('dblclick', () => fv.paused ? fv.play() : fv.pause());
  }

  /* ---------- lightbox for still images ---------- */
  const lb = $('#lb'), lbImg = $('#lbImg');
  $$('img.zoomable').forEach(im => im.addEventListener('click', () => { lbImg.src = im.src; lbImg.alt = im.alt; lb.showModal(); }));
  $('#lbClose').addEventListener('click', () => lb.close());
  lb.addEventListener('click', e => { if (e.target === lb) lb.close(); });

  /* ---------- active nav link + attraction map ---------- */
  const links = $$('.nav-links a');
  const sio = new IntersectionObserver(es => es.forEach(e => {
    if (e.isIntersecting) links.forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#' + e.target.id));
  }), { rootMargin: '-45% 0px -50% 0px' });
  ['loop', 'tasks', 'bibtex'].forEach(id => { const s = document.getElementById(id); if (s) sio.observe(s); });

  const mapLinks = $$('.map-row a'), mapRow = $('.map-row');
  const mio = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    mapLinks.forEach(a => {
      const on = a.getAttribute('href') === '#' + e.target.id; a.classList.toggle('active', on);
      if (on && mapRow.scrollWidth > mapRow.clientWidth) mapRow.scrollTo({ left: a.offsetLeft - 16, behavior: reduced ? 'auto' : 'smooth' });
    });
  }), { rootMargin: '-40% 0px -55% 0px' });
  $$('.ride').forEach(r => mio.observe(r));

  /* ---------- copy bibtex, with a little confetti ---------- */
  $('#copyBib').addEventListener('click', async e => {
    const btn = e.currentTarget;
    try { await navigator.clipboard.writeText($('#bibText').textContent); btn.textContent = 'copied ✓'; }
    catch { btn.textContent = 'select & copy'; }
    if (!reduced) {
      const b = btn.getBoundingClientRect();
      for (let i = 0; i < 18; i++) {
        const s = document.createElement('span'); s.className = 'confetti';
        s.style.cssText = `left:${b.left + b.width / 2}px;top:${b.top + b.height / 2}px;background:${PALETTE[i % PALETTE.length]};--dx:${rnd(-140, 140).toFixed(0)}px;--dy:${rnd(-150, 60).toFixed(0)}px;--r:${rnd(-540, 540).toFixed(0)}deg`;
        document.body.append(s); setTimeout(() => s.remove(), 1050);
      }
    }
    setTimeout(() => { btn.textContent = 'copy'; }, 1600);
  });

  /* "soon" buttons do nothing yet */
  $$('.btn.soon').forEach(b => b.addEventListener('click', e => e.preventDefault()));
})();
