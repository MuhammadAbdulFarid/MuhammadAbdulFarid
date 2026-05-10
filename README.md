<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Muhammad Abdul Farid — Profile</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Outfit:wght@300;400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/tabler-icons.min.css">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #0d1117;
    --bg2: #161b22;
    --bg3: #21262d;
    --border: #30363d;
    --border2: #484f58;
    --text: #e6edf3;
    --text2: #8b949e;
    --text3: #656d76;
    --green: #3fb950;
    --green-dim: #4ade8044;
    --green-bg: #0d2b1a;
    --radius: 10px;
    --radius-sm: 6px;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Outfit', sans-serif;
    min-height: 100vh;
    display: flex;
    align-items: flex-start;
    justify-content: center;
    padding: 2rem 1rem;
  }

  .card {
    width: 100%;
    max-width: 580px;
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.75rem;
  }

  /* ── HERO ── */
  .hero {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 1.25rem;
    align-items: center;
    margin-bottom: 1.5rem;
  }

  .avatar-wrap { position: relative; flex-shrink: 0; }

  .avatar {
    width: 80px; height: 80px;
    border-radius: 50%;
    background: #0d2b1a;
    border: 2px solid #238636;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Space Mono', monospace;
    font-size: 20px; font-weight: 700;
    color: #4ade80;
    position: relative; overflow: hidden;
  }

  .avatar img {
    width: 100%; height: 100%;
    object-fit: cover; border-radius: 50%;
    position: absolute; top: 0; left: 0;
  }

  .status-dot {
    position: absolute; bottom: 4px; right: 4px;
    width: 14px; height: 14px;
    border-radius: 50%; background: #3fb950;
    border: 2.5px solid var(--bg2);
    z-index: 2;
  }

  .pulse {
    position: absolute; bottom: 4px; right: 4px;
    width: 14px; height: 14px;
    border-radius: 50%; background: #3fb95066;
    animation: ripple 2s infinite;
    z-index: 1;
  }

  @keyframes ripple {
    0% { transform: scale(1); opacity: .6; }
    100% { transform: scale(2.8); opacity: 0; }
  }

  .hero-name {
    font-family: 'Space Mono', monospace;
    font-size: 18px; font-weight: 700;
    color: var(--text); letter-spacing: -.4px; line-height: 1.2;
  }

  .hero-handle {
    font-family: 'Space Mono', monospace;
    font-size: 12px; color: var(--green);
    margin: 3px 0 6px;
  }

  .hero-bio {
    font-size: 13px; color: var(--text2);
    margin-bottom: 8px; line-height: 1.5;
  }

  .hero-meta {
    display: flex; gap: 10px; flex-wrap: wrap; align-items: center;
  }

  .meta-chip {
    display: flex; align-items: center; gap: 4px;
    font-size: 11px; color: var(--text3);
  }

  .meta-chip i { font-size: 13px; }

  /* ── STATS ── */
  .stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin-bottom: 1.5rem;
  }

  .stat-box {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 12px 10px;
    text-align: center;
    cursor: default;
    transition: border-color .2s, transform .2s;
  }

  .stat-box:hover {
    border-color: #238636;
    transform: translateY(-2px);
  }

  .stat-n {
    font-family: 'Space Mono', monospace;
    font-size: 20px; font-weight: 700;
    color: var(--text); display: block;
  }

  .stat-l {
    font-size: 10px; color: var(--text3);
    text-transform: uppercase; letter-spacing: 1px;
    margin-top: 2px; display: block;
  }

  /* ── TABS ── */
  .tabs {
    display: flex; gap: 4px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.25rem;
  }

  .tab-btn {
    padding: 7px 14px;
    font-size: 12px;
    font-family: 'Space Mono', monospace;
    border: none; background: none;
    color: var(--text3); cursor: pointer;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
    transition: color .15s;
  }

  .tab-btn.active { color: var(--green); border-bottom-color: var(--green); }
  .tab-btn:hover:not(.active) { color: var(--text); }

  .tab-panel { display: none; }
  .tab-panel.active { display: block; }

  /* ── STACK ── */
  .stack-grid {
    display: flex; flex-wrap: wrap; gap: 7px;
    margin-bottom: 1.25rem;
  }

  .tech-pill {
    display: flex; align-items: center; gap: 6px;
    padding: 5px 12px;
    border-radius: 20px;
    border: 1px solid var(--border);
    background: var(--bg);
    font-size: 12px; font-weight: 500;
    color: var(--text);
    cursor: default;
    transition: border-color .15s, transform .15s, background .15s;
  }

  .tech-pill:hover {
    border-color: #238636;
    background: #0d2b1a33;
    transform: translateY(-1px);
  }

  .tech-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }

  /* ── LANG BARS ── */
  .section-label {
    font-family: 'Space Mono', monospace;
    font-size: 10px; letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text3); margin-bottom: 12px;
  }

  .lang-row {
    display: flex; align-items: center;
    gap: 10px; margin-bottom: 9px;
  }

  .lang-name {
    font-family: 'Space Mono', monospace;
    font-size: 11px; color: var(--text2);
    width: 86px; flex-shrink: 0;
  }

  .lang-track {
    flex: 1; height: 6px;
    background: var(--bg3); border-radius: 4px; overflow: hidden;
  }

  .lang-fill {
    height: 100%; border-radius: 4px; width: 0%;
    transition: width .9s cubic-bezier(.4,0,.2,1);
  }

  .lang-pct {
    font-family: 'Space Mono', monospace;
    font-size: 11px; color: var(--text3);
    width: 30px; text-align: right; flex-shrink: 0;
  }

  /* ── REPOS ── */
  .repos-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }

  .repo-card {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 12px 14px;
    cursor: pointer;
    transition: border-color .15s, transform .15s;
    text-decoration: none; display: block;
  }

  .repo-card:hover { border-color: #238636; transform: translateY(-2px); }

  .repo-name {
    font-family: 'Space Mono', monospace;
    font-size: 12px; font-weight: 700; color: var(--green);
    margin-bottom: 4px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }

  .repo-desc {
    font-size: 11px; color: var(--text3);
    line-height: 1.4; margin-bottom: 10px; min-height: 28px;
    display: -webkit-box;
    -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
  }

  .repo-meta { display: flex; align-items: center; gap: 8px; }
  .repo-lang { display: flex; align-items: center; gap: 4px; font-size: 10px; color: var(--text3); }
  .repo-dot { width: 8px; height: 8px; border-radius: 50%; }

  .see-all {
    margin-top: 10px; text-align: right;
    font-family: 'Space Mono', monospace;
    font-size: 11px; color: var(--green);
    text-decoration: none; display: block;
  }

  .see-all:hover { text-decoration: underline; }

  /* ── CONNECT ── */
  .connect-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }

  .connect-card {
    display: flex; align-items: center; gap: 10px;
    padding: 12px 14px;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    background: var(--bg);
    text-decoration: none; color: var(--text);
    transition: border-color .15s, transform .15s;
  }

  .connect-card:hover { border-color: var(--border2); transform: translateY(-2px); }

  .connect-icon-box {
    width: 34px; height: 34px;
    border-radius: var(--radius-sm);
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
  }

  .connect-label { font-size: 13px; font-weight: 500; color: var(--text); }
  .connect-handle { font-size: 10px; color: var(--text3); font-family: 'Space Mono', monospace; }

  /* ── FOOTER ── */
  .footer {
    margin-top: 1.5rem; padding-top: 1rem;
    border-top: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
  }

  .views-badge {
    display: flex; align-items: center; gap: 6px;
    font-family: 'Space Mono', monospace;
    font-size: 10px; color: var(--text3);
  }

  .live-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--green); animation: blink 2s infinite;
  }

  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }

  .ver-tag { font-family: 'Space Mono', monospace; font-size: 10px; color: var(--text3); }
  .ver-green { color: var(--green); }

  /* ── RESPONSIVE ── */
  @media (max-width: 480px) {
    .stats-row { grid-template-columns: repeat(2, 1fr); }
    .repos-grid { grid-template-columns: 1fr; }
    .connect-grid { grid-template-columns: 1fr; }
    .hero { gap: .75rem; }
    .hero-name { font-size: 15px; }
    .card { padding: 1.25rem; }
  }
</style>
</head>
<body>

<div class="card">

  <!-- HERO -->
  <div class="hero">
    <div class="avatar-wrap">
      <div class="avatar">
        <img src="https://avatars.githubusercontent.com/u/189076926?v=4" alt="Muhammad Abdul Farid" onerror="this.style.display='none'">
        MAF
      </div>
      <div class="pulse"></div>
      <div class="status-dot"></div>
    </div>
    <div class="hero-info">
      <div class="hero-name">Muhammad Abdul Farid</div>
      <div class="hero-handle">@cunam_aja</div>
      <div class="hero-bio">Full-Stack Developer &amp; Cybersecurity Student 🔐</div>
      <div class="hero-meta">
        <span class="meta-chip"><i class="ti ti-building"></i> Unismuh Makassar</span>
        <span class="meta-chip"><i class="ti ti-map-pin"></i> Sulawesi Selatan, ID</span>
        <span class="meta-chip"><i class="ti ti-home"></i> Working from home</span>
      </div>
    </div>
  </div>

  <!-- STATS -->
  <div class="stats-row">
    <div class="stat-box">
      <span class="stat-n" id="cRepos">0</span>
      <span class="stat-l">Repos</span>
    </div>
    <div class="stat-box">
      <span class="stat-n" id="cFollow">0</span>
      <span class="stat-l">Followers</span>
    </div>
    <div class="stat-box">
      <span class="stat-n" id="cFollowing">0</span>
      <span class="stat-l">Following</span>
    </div>
    <div class="stat-box">
      <span class="stat-n">B+</span>
      <span class="stat-l">Rank</span>
    </div>
  </div>

  <!-- TABS -->
  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab(this,'stack')">// stack</button>
    <button class="tab-btn" onclick="switchTab(this,'repos')">// repos</button>
    <button class="tab-btn" onclick="switchTab(this,'connect')">// connect</button>
  </div>

  <!-- TAB: STACK -->
  <div class="tab-panel active" id="tab-stack">
    <div class="stack-grid">
      <div class="tech-pill"><div class="tech-dot" style="background:#F7DF1E"></div>JavaScript</div>
      <div class="tech-pill"><div class="tech-dot" style="background:#3776AB"></div>Python</div>
      <div class="tech-pill"><div class="tech-dot" style="background:#E34F26"></div>HTML5</div>
      <div class="tech-pill"><div class="tech-dot" style="background:#1572B6"></div>CSS3</div>
      <div class="tech-pill"><div class="tech-dot" style="background:#777BB4"></div>PHP</div>
      <div class="tech-pill"><div class="tech-dot" style="background:#ED8B00"></div>Java</div>
      <div class="tech-pill"><div class="tech-dot" style="background:#339933"></div>Node.js</div>
      <div class="tech-pill"><div class="tech-dot" style="background:#007ACC"></div>VS Code</div>
    </div>

    <p class="section-label">// top languages</p>
    <div class="lang-row"><span class="lang-name">JavaScript</span><div class="lang-track"><div class="lang-fill" data-w="42" style="background:#F7DF1E"></div></div><span class="lang-pct">42%</span></div>
    <div class="lang-row"><span class="lang-name">Python</span><div class="lang-track"><div class="lang-fill" data-w="26" style="background:#3776AB"></div></div><span class="lang-pct">26%</span></div>
    <div class="lang-row"><span class="lang-name">PHP</span><div class="lang-track"><div class="lang-fill" data-w="16" style="background:#777BB4"></div></div><span class="lang-pct">16%</span></div>
    <div class="lang-row"><span class="lang-name">HTML/CSS</span><div class="lang-track"><div class="lang-fill" data-w="11" style="background:#E34F26"></div></div><span class="lang-pct">11%</span></div>
    <div class="lang-row"><span class="lang-name">Java</span><div class="lang-track"><div class="lang-fill" data-w="5" style="background:#ED8B00"></div></div><span class="lang-pct">5%</span></div>
  </div>

  <!-- TAB: REPOS -->
  <div class="tab-panel" id="tab-repos">
    <div class="repos-grid">
      <a class="repo-card" href="https://github.com/MuhammadAbdulFarid/cunam" target="_blank">
        <div class="repo-name">📁 cunam</div>
        <div class="repo-desc">New student project — learning journey begins here.</div>
        <div class="repo-meta"><div class="repo-lang"><div class="repo-dot" style="background:#E34F26"></div>HTML</div></div>
      </a>
      <a class="repo-card" href="https://github.com/MuhammadAbdulFarid/gamekotak_kotak" target="_blank">
        <div class="repo-name">🎮 gamekotak_kotak</div>
        <div class="repo-desc">A grid-based game built for learning purposes.</div>
        <div class="repo-meta"><div class="repo-lang"><div class="repo-dot" style="background:#F7DF1E"></div>JavaScript</div></div>
      </a>
      <a class="repo-card" href="https://github.com/MuhammadAbdulFarid/belajar" target="_blank">
        <div class="repo-name">📚 belajar</div>
        <div class="repo-desc">Personal Python learning playground &amp; experiments.</div>
        <div class="repo-meta"><div class="repo-lang"><div class="repo-dot" style="background:#3776AB"></div>Python</div></div>
      </a>
      <a class="repo-card" href="https://github.com/MuhammadAbdulFarid/projectdasarjavascript" target="_blank">
        <div class="repo-name">⚡ projectdasarjavascript</div>
        <div class="repo-desc">Fundamental JavaScript exercises &amp; mini-projects.</div>
        <div class="repo-meta"><div class="repo-lang"><div class="repo-dot" style="background:#F7DF1E"></div>JavaScript</div></div>
      </a>
      <a class="repo-card" href="https://github.com/MuhammadAbdulFarid/scanemukadan-audio" target="_blank">
        <div class="repo-name">🎵 scanemukadan-audio</div>
        <div class="repo-desc">Audio scanning utility built with Python.</div>
        <div class="repo-meta"><div class="repo-lang"><div class="repo-dot" style="background:#3776AB"></div>Python</div></div>
      </a>
      <a class="repo-card" href="https://github.com/MuhammadAbdulFarid/web_produk_front" target="_blank">
        <div class="repo-name">🛒 web_produk_front</div>
        <div class="repo-desc">Front-end product showcase webpage.</div>
        <div class="repo-meta"><div class="repo-lang"><div class="repo-dot" style="background:#E34F26"></div>HTML</div></div>
      </a>
    </div>
    <a class="see-all" href="https://github.com/MuhammadAbdulFarid?tab=repositories" target="_blank">→ see all 33 repos</a>
  </div>

  <!-- TAB: CONNECT -->
  <div class="tab-panel" id="tab-connect">
    <div class="connect-grid">
      <a class="connect-card" href="https://github.com/MuhammadAbdulFarid" target="_blank">
        <div class="connect-icon-box" style="background:#0d1117;border:1px solid #30363d;">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="#f0f6fc"><path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"/></svg>
        </div>
        <div>
          <div class="connect-label">GitHub</div>
          <div class="connect-handle">@MuhammadAbdulFarid</div>
        </div>
      </a>
      <a class="connect-card" href="https://www.instagram.com/oi_parid" target="_blank">
        <div class="connect-icon-box" style="background:linear-gradient(135deg,#405de6,#833ab4,#c13584,#e1306c,#fd1d1d);">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><rect x="2" y="2" width="20" height="20" rx="5"/><circle cx="12" cy="12" r="4"/><circle cx="17.5" cy="6.5" r="1.2" fill="white" stroke="none"/></svg>
        </div>
        <div>
          <div class="connect-label">Instagram</div>
          <div class="connect-handle">@oi_parid</div>
        </div>
      </a>
      <a class="connect-card" href="https://www.tiktok.com/@is_it_farid" target="_blank">
        <div class="connect-icon-box" style="background:#010101;border:1px solid #333;">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="white"><path d="M19.59 6.69a4.83 4.83 0 01-3.77-4.25V2h-3.45v13.67a2.89 2.89 0 01-2.88 2.5 2.89 2.89 0 01-2.89-2.89 2.89 2.89 0 012.89-2.89c.28 0 .54.04.79.1V9.01a6.33 6.33 0 00-.79-.05 6.34 6.34 0 00-6.34 6.34 6.34 6.34 0 006.34 6.34 6.34 6.34 0 006.33-6.34V8.98a8.18 8.18 0 004.77 1.52V7.05a4.85 4.85 0 01-1-.36z"/></svg>
        </div>
        <div>
          <div class="connect-label">TikTok</div>
          <div class="connect-handle">@is_it_farid</div>
        </div>
      </a>
      <a class="connect-card" href="mailto:105841100724@student.unismuh.ac.id">
        <div class="connect-icon-box" style="background:#1a0a0a;border:1px solid #D1483655;">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#D14836" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="4" width="20" height="16" rx="2"/><polyline points="2,4 12,13 22,4"/></svg>
        </div>
        <div>
          <div class="connect-label">Email</div>
          <div class="connect-handle">student.unismuh.ac.id</div>
        </div>
      </a>
      <a class="connect-card" href="https://www.youtube.com/@YOUR_YOUTUBE_HANDLE" target="_blank">
        <div class="connect-icon-box" style="background:#1a0000;border:1px solid #FF000044;">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="#FF0000"><path d="M23.498 6.186a3.016 3.016 0 00-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 00.502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 002.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 002.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/></svg>
        </div>
        <div>
          <div class="connect-label">YouTube</div>
          <div class="connect-handle">@YOUR_HANDLE</div>
        </div>
      </a>
    </div>
  </div>

  <!-- FOOTER -->
  <div class="footer">
    <div class="views-badge">
      <div class="live-dot"></div>
      <span>MuhammadAbdulFarid · live</span>
    </div>
    <span class="ver-tag"><span class="ver-green">~/</span> profile v3.0</span>
  </div>

</div><!-- end .card -->

<script>
  function switchTab(btn, id) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + id).classList.add('active');
    if (id === 'stack') animateBars();
  }

  function animateBars() {
    document.querySelectorAll('.lang-fill').forEach((el, i) => {
      el.style.width = '0%';
      setTimeout(() => { el.style.width = el.dataset.w + '%'; }, i * 130 + 100);
    });
  }

  function animateCount(el, target, duration) {
    let start = 0, step = target / (duration / 16);
    const tick = () => {
      start = Math.min(start + step, target);
      el.textContent = Math.round(start);
      if (start < target) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  window.addEventListener('load', () => {
    setTimeout(() => {
      animateBars();
      animateCount(document.getElementById('cRepos'), 33, 800);
      animateCount(document.getElementById('cFollow'), 8, 600);
      animateCount(document.getElementById('cFollowing'), 16, 700);
    }, 300);
  });
</script>

</body>
</html>
