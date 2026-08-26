/* clipFarm front end — thin client over the Flask API + SSE progress stream. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  const state = { jobId: null, source: null, file: null, stages: [], status: null };

  /* ── helpers ─────────────────────────────────────────── */

  const hms = (s) => {
    s = Math.max(0, Math.round(s || 0));
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
    const mm = h ? String(m).padStart(2, "0") : String(m);
    return (h ? `${h}:` : "") + `${mm}:${String(sec).padStart(2, "0")}`;
  };

  const scoreClass = (n) => (n >= 75 ? "score--hot" : n >= 55 ? "score--warm" : "score--cold");

  async function api(path, options = {}) {
    const res = await fetch(path, options);
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.error || `${res.status} ${res.statusText}`);
    return body;
  }

  /* ── health ──────────────────────────────────────────── */

  async function loadHealth() {
    const box = $("health");
    try {
      const data = await api("/api/health");
      const failed = Object.entries(data.checks).filter(([, c]) => !c.ok);
      box.innerHTML = "";
      box.appendChild(el("span", `dot ${data.ok ? "dot--ok" : "dot--bad"}`));
      box.appendChild(el(
        "span",
        "health-text",
        data.ok
          ? `${data.config.whisper_model} · ${data.config.ollama_model} · ${data.config.resolution}`
          : `${failed.length} check(s) failing`
      ));
      box.title = Object.entries(data.checks)
        .map(([k, c]) => `${c.ok ? "✓" : "✗"} ${k}: ${c.detail}`)
        .join("\n");
    } catch (err) {
      box.innerHTML = "";
      box.appendChild(el("span", "dot dot--bad"));
      box.appendChild(el("span", "health-text", "API unreachable"));
      box.title = String(err.message || err);
    }
  }

  /* ── new job form ────────────────────────────────────── */

  function initForm() {
    document.querySelectorAll(".tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach((t) => t.classList.remove("is-active"));
        document.querySelectorAll(".pane").forEach((p) => p.classList.remove("is-active"));
        tab.classList.add("is-active");
        $(`pane-${tab.dataset.tab}`).classList.add("is-active");
      });
    });

    const drop = $("dropzone");
    const input = $("file-input");

    drop.addEventListener("click", () => input.click());
    input.addEventListener("change", () => setFile(input.files[0]));

    ["dragenter", "dragover"].forEach((evt) =>
      drop.addEventListener(evt, (e) => {
        e.preventDefault();
        drop.classList.add("is-over");
      })
    );
    ["dragleave", "drop"].forEach((evt) =>
      drop.addEventListener(evt, (e) => {
        e.preventDefault();
        drop.classList.remove("is-over");
      })
    );
    drop.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

    $("submit").addEventListener("click", submit);
    $("url-input").addEventListener("keydown", (e) => e.key === "Enter" && submit());
    $("cancel").addEventListener("click", cancelJob);
  }

  function setFile(file) {
    if (!file) return;
    state.file = file;
    const label = $("filename");
    label.textContent = `${file.name} · ${(file.size / 1e6).toFixed(1)} MB`;
    label.hidden = false;
  }

  async function submit() {
    const btn = $("submit");
    const errBox = $("submit-error");
    errBox.hidden = true;

    const usingUrl = document.querySelector(".tab.is-active").dataset.tab === "url";
    let options;

    if (usingUrl) {
      const url = $("url-input").value.trim();
      if (!url) return fail("Paste a video URL first.");
      options = {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      };
    } else {
      if (!state.file) return fail("Choose a file first.");
      const form = new FormData();
      form.append("file", state.file);
      options = { method: "POST", body: form };
    }

    btn.disabled = true;
    btn.textContent = usingUrl ? "Starting…" : "Uploading…";
    try {
      const job = await api("/api/jobs", options);
      attachJob(job);
      loadHistory();
    } catch (err) {
      fail(err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = "Generate clips";
    }

    function fail(message) {
      errBox.textContent = message;
      errBox.hidden = false;
    }
  }

  async function cancelJob() {
    if (!state.jobId) return;
    try {
      await api(`/api/jobs/${state.jobId}/cancel`, { method: "POST" });
    } catch (err) {
      console.warn(err);
    }
  }

  /* ── live job view ───────────────────────────────────── */

  let source = null;

  function attachJob(job) {
    if (source) source.close();
    state.jobId = job.id;
    state.stages = job.stages || state.stages;
    $("log").textContent = "";
    $("clips").innerHTML = "";
    $("search-results").innerHTML = "";
    $("job-panel").hidden = false;
    $("clips-section").hidden = true;
    $("search-panel").hidden = true;
    renderJob(job);
    $("job-panel").scrollIntoView({ behavior: "smooth", block: "start" });

    source = new EventSource(`/api/jobs/${job.id}/events`);
    source.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      if (payload.type === "state") renderJob(payload.job);
      else if (payload.type === "log") appendLog(payload.message);
      else if (payload.type === "end") {
        source.close();
        source = null;
        loadHistory();
      }
    };
    source.onerror = () => {
      // The stream closes normally when the job ends; only poll if it died early.
      if (source && state.status !== "done") {
        source.close();
        source = null;
        setTimeout(() => refreshJob(job.id), 1500);
      }
    };
  }

  async function refreshJob(jobId) {
    try {
      renderJob(await api(`/api/jobs/${jobId}`));
    } catch (err) {
      console.warn(err);
    }
  }

  function renderJob(job) {
    state.status = job.status;
    state.stages = job.stages || state.stages;

    const title = job.media?.title || job.source_label || job.id;
    $("job-title").textContent = title;

    const bits = [];
    if (job.media?.duration) bits.push(`${hms(job.media.duration)} source`);
    if (job.stats?.words) bits.push(`${job.stats.words} words`);
    if (job.stats?.vectors) bits.push(`${job.stats.vectors} vectors`);
    if (job.stats?.elapsed_seconds) bits.push(`${job.stats.elapsed_seconds}s total`);
    $("job-sub").textContent = bits.join(" · ") || job.source_label;

    const pill = $("job-status");
    pill.textContent = job.status;
    pill.className = `pill pill--${job.status}`;

    $("cancel").hidden = !(job.status === "running" || job.status === "queued");

    $("progress-bar").style.width = `${(job.progress * 100).toFixed(1)}%`;
    $("job-message").textContent = job.message || "";

    const errBox = $("job-error");
    errBox.hidden = !job.error;
    if (job.error) errBox.textContent = job.error;

    renderRail(job);

    if (job.clips?.length) {
      $("clips-section").hidden = false;
      $("clips-meta").textContent =
        `${job.clips.length} clip${job.clips.length > 1 ? "s" : ""}` +
        (job.stats?.candidates_scored ? ` from ${job.stats.candidates_scored} scored candidates` : "");
      renderClips(job);
    }

    if (job.status === "done") {
      $("search-panel").hidden = false;
      initSearch();
    }
  }

  function renderRail(job) {
    const rail = $("rail");
    const stages = state.stages;
    if (!stages.length) return;
    const activeIdx = stages.findIndex((s) => s.key === job.stage);

    rail.innerHTML = "";
    stages.forEach((stage, i) => {
      const done = job.status === "done" || (activeIdx > -1 && i < activeIdx);
      const active = i === activeIdx && job.status === "running";
      const li = el("li", done ? "is-done" : active ? "is-active" : "");
      li.appendChild(el("span", "idx", done ? "✓" : String(i + 1)));
      li.appendChild(el("span", "", stage.label));
      rail.appendChild(li);
    });
  }

  function appendLog(message) {
    const log = $("log");
    log.textContent += `${message}\n`;
    log.scrollTop = log.scrollHeight;
  }

  /* ── clips ───────────────────────────────────────────── */

  const BAR_LABELS = {
    hook_strength: "hook",
    emotional_impact: "emotion",
    standalone_clarity: "clarity",
    shareability: "shareable",
  };

  function renderClips(job) {
    const grid = $("clips");
    grid.innerHTML = "";
    job.clips.forEach((clip) => grid.appendChild(clipCard(clip)));
  }

  function clipCard(clip) {
    const card = el("div", "clip");

    const media = el("div", "clip-media");
    media.appendChild(el("span", "clip-rank", `#${clip.rank}`));
    const score = el("span", `clip-score ${scoreClass(clip.virality_score)}`,
      String(Math.round(clip.virality_score)));
    score.title = "Virality score (0–100), judged by the local LLM";
    media.appendChild(score);
    media.appendChild(el("span", "clip-time",
      `${hms(clip.start)}–${hms(clip.end)} · ${Math.round(clip.duration)}s`));

    if (clip.thumbnail_url) {
      const img = document.createElement("img");
      img.src = clip.thumbnail_url;
      img.alt = "";
      img.loading = "lazy";
      media.appendChild(img);
    }

    const play = el("button", "clip-play", "▶");
    play.setAttribute("aria-label", `Play ${clip.title}`);
    play.addEventListener("click", () => {
      const video = document.createElement("video");
      video.src = clip.video_url;
      video.controls = true;
      video.autoplay = true;
      video.playsInline = true;
      media.querySelector("img")?.remove();
      play.remove();
      media.appendChild(video);
    });
    media.appendChild(play);
    card.appendChild(media);

    const body = el("div", "clip-body");
    body.appendChild(el("div", "clip-title", clip.title));
    if (clip.hook) body.appendChild(el("div", "clip-hook", `“${clip.hook}”`));
    if (clip.reason) body.appendChild(el("div", "clip-reason", clip.reason));

    const breakdown = clip.breakdown || {};
    if (Object.keys(breakdown).length) {
      const bars = el("div", "bars");
      Object.entries(BAR_LABELS).forEach(([key, label]) => {
        if (breakdown[key] === undefined) return;
        const value = Number(breakdown[key]) || 0;
        const bar = el("div", "bar");
        bar.appendChild(el("span", "bar-label", label));
        const track = el("div", "bar-track");
        const fill = el("div", "bar-fill");
        fill.style.width = `${Math.min(100, value * 10)}%`;
        track.appendChild(fill);
        bar.appendChild(track);
        bar.appendChild(el("span", "bar-val", String(value)));
        bars.appendChild(bar);
      });
      body.appendChild(bars);
    }

    if (clip.tags?.length) {
      const tags = el("div", "tags");
      clip.tags.forEach((t) => tags.appendChild(el("span", "tag", `#${t}`)));
      body.appendChild(tags);
    }

    const actions = el("div", "clip-actions");
    actions.appendChild(linkBtn(`${clip.video_url}?download=1`, "Download MP4"));
    actions.appendChild(linkBtn(clip.srt_url, "SRT"));
    body.appendChild(actions);

    card.appendChild(body);
    return card;
  }

  function linkBtn(href, label) {
    const a = document.createElement("a");
    a.className = "btn btn--sm";
    a.href = href;
    a.textContent = label;
    return a;
  }

  /* ── semantic search ─────────────────────────────────── */

  let searchWired = false;

  function initSearch() {
    if (searchWired) return;
    searchWired = true;
    $("search-btn").addEventListener("click", runSearch);
    $("search-input").addEventListener("keydown", (e) => e.key === "Enter" && runSearch());
  }

  async function runSearch() {
    const query = $("search-input").value.trim();
    const box = $("search-results");
    if (!query || !state.jobId) return;
    box.innerHTML = "";
    box.appendChild(el("p", "hint", "Searching…"));
    try {
      const data = await api(`/api/jobs/${state.jobId}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, k: 6 }),
      });
      box.innerHTML = "";
      if (!data.hits.length) {
        box.appendChild(el("p", "hint", "No matches."));
        return;
      }
      data.hits.forEach((hit) => {
        const node = el("div", "hit");
        const meta = el("div", "hit-meta");
        meta.appendChild(el("span", "", `${hms(hit.start)} – ${hms(hit.end)}`));
        meta.appendChild(el("span", "hit-rel", `relevance ${hit.relevance.toFixed(3)}`));
        node.appendChild(meta);
        node.appendChild(el("div", "", hit.text));
        box.appendChild(node);
      });
    } catch (err) {
      box.innerHTML = "";
      box.appendChild(el("p", "err", err.message));
    }
  }

  /* ── history ─────────────────────────────────────────── */

  async function loadHistory() {
    let jobs;
    try {
      ({ jobs } = await api("/api/jobs"));
    } catch {
      return;
    }
    const box = $("history");
    box.innerHTML = "";
    const others = jobs.filter((j) => j.id !== state.jobId);
    $("history-section").hidden = others.length === 0;

    others.forEach((job) => {
      const item = el("div", "hist-item");
      item.appendChild(el("span", `dot dot--${job.status === "done" ? "ok" : job.status === "error" ? "bad" : "idle"}`));
      item.appendChild(el("span", "hist-label", job.media?.title || job.source_label));
      item.appendChild(el("span", "hist-meta",
        `${job.clips?.length || 0} clips · ${new Date(job.created_at * 1000).toLocaleString()}`));

      const del = el("button", "hist-del", "×");
      del.title = "Delete job and its clips";
      del.addEventListener("click", async (e) => {
        e.stopPropagation();
        if (!confirm("Delete this job and all of its rendered clips?")) return;
        await api(`/api/jobs/${job.id}`, { method: "DELETE" }).catch(() => {});
        loadHistory();
      });
      item.appendChild(del);

      item.addEventListener("click", async () => {
        const full = await api(`/api/jobs/${job.id}`).catch(() => null);
        if (full) attachJob(full);
      });
      box.appendChild(item);
    });
  }

  /* ── boot ────────────────────────────────────────────── */

  initForm();
  loadHealth();
  loadHistory();
  setInterval(loadHealth, 30000);
})();
