(() => {
  const root = document.documentElement;
  let stored = null;
  try { stored = localStorage.getItem("paperwatch-theme"); } catch (_) { /* storage unavailable */ }
  const preferredDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = stored || (preferredDark ? "dark" : "light");

  const toggle = document.getElementById("theme-toggle");
  if (toggle) {
    const updateLabel = () => {
      const dark = root.dataset.theme === "dark";
      toggle.textContent = dark ? "☀" : "◐";
      toggle.setAttribute("aria-label", dark ? "Use light mode" : "Use dark mode");
    };
    updateLabel();
    toggle.addEventListener("click", () => {
      root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
      try { localStorage.setItem("paperwatch-theme", root.dataset.theme); } catch (_) { /* storage unavailable */ }
      updateLabel();
    });
  }

  const paperJobs = document.getElementById("paper-jobs");
  if (document.body.dataset.runRunning === "true" && !paperJobs) {
    window.setTimeout(() => window.location.reload(), 5000);
  }

  document.querySelectorAll("form[data-confirm]").forEach((form) => {
    form.addEventListener("submit", (event) => {
      if (!window.confirm(form.dataset.confirm)) event.preventDefault();
    });
  });

  const authorFilter = document.getElementById("author-filter");
  if (authorFilter) {
    const rows = Array.from(document.querySelectorAll("#tracked-author-rows tr[data-author-search]"));
    const applyAuthorFilter = () => {
      const term = authorFilter.value.trim().toLocaleLowerCase();
      rows.forEach((row) => {
        row.hidden = Boolean(term) && !row.dataset.authorSearch.toLocaleLowerCase().includes(term);
      });
    };
    authorFilter.addEventListener("input", applyAuthorFilter);
  }

  if (paperJobs) {
    const paperId = Number(paperJobs.dataset.paperId);
    const stage = document.getElementById("job-stage");
    const message = document.getElementById("job-message");
    const output = document.getElementById("job-output");
    const cancel = document.getElementById("job-cancel");
    const forms = Array.from(document.querySelectorAll("form[data-paper-job]"));
    let activeKind = null;
    let wasRunning = false;
    let pollTimer = null;

    const setFormsDisabled = (disabled) => {
      forms.forEach((form) => {
        form.querySelectorAll("button[type=submit]").forEach((button) => { button.disabled = disabled; });
      });
    };

    const refreshResults = async () => {
      const scrollY = window.scrollY;
      const response = await fetch(window.location.pathname, { headers: { "X-Paperwatch-Fragment": "1" } });
      if (!response.ok) return;
      const documentText = await response.text();
      const parsed = new DOMParser().parseFromString(documentText, "text/html");
      const fresh = parsed.getElementById("paper-results");
      const current = document.getElementById("paper-results");
      if (fresh && current) current.replaceWith(fresh);
      window.scrollTo(0, scrollY);
    };

    const showSnapshot = (kind, snapshot) => {
      activeKind = kind;
      paperJobs.hidden = false;
      stage.textContent = snapshot.stage || (snapshot.running ? "Running" : "Complete");
      message.textContent = snapshot.message || "";
      output.textContent = snapshot.live_output || "";
      output.hidden = !snapshot.live_output;
      cancel.hidden = !snapshot.running;
      cancel.disabled = Boolean(snapshot.cancel_requested);
      cancel.textContent = snapshot.cancel_requested ? "Stopping…" : "Stop";
      setFormsDisabled(Boolean(snapshot.running));
    };

    const poll = async () => {
      try {
        const [analysisResponse, queryResponse] = await Promise.all([
          fetch("/api/analysis-status", { cache: "no-store" }),
          fetch("/api/query-status", { cache: "no-store" }),
        ]);
        const analysis = await analysisResponse.json();
        const query = await queryResponse.json();
        let kind = activeKind;
        let snapshot = kind === "query" ? query : analysis;
        if (analysis.running && Number(analysis.paper_id) === paperId) { kind = "analysis"; snapshot = analysis; }
        else if (query.running && Number(query.paper_id) === paperId) { kind = "query"; snapshot = query; }
        const relevant = Number(snapshot.paper_id) === paperId && Boolean(snapshot.batch_id);
        if (relevant) showSnapshot(kind || "analysis", snapshot);
        const nowRunning = relevant && Boolean(snapshot.running);
        if (wasRunning && !nowRunning) await refreshResults();
        wasRunning = nowRunning;
        if (nowRunning) pollTimer = window.setTimeout(poll, 600);
        else setFormsDisabled(false);
      } catch (_) {
        message.textContent = "Live status is temporarily unavailable.";
        setFormsDisabled(false);
      }
    };

    forms.forEach((form) => {
      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        activeKind = form.dataset.paperJob;
        wasRunning = true;
        paperJobs.hidden = false;
        stage.textContent = activeKind === "query" ? "Starting saved prompt" : "Starting summary";
        message.textContent = "";
        output.textContent = "";
        output.hidden = true;
        cancel.hidden = false;
        cancel.disabled = false;
        setFormsDisabled(true);
        try {
          const response = await fetch(form.action, {
            method: "POST",
            body: new FormData(form),
            headers: { "X-Paperwatch-Async": "1", "Accept": "application/json" },
          });
          const payload = await response.json();
          if (!response.ok || !payload.ok) throw new Error(payload.message || "Could not start the job");
          message.textContent = payload.message || "Started";
          if (pollTimer) window.clearTimeout(pollTimer);
          pollTimer = window.setTimeout(poll, 100);
        } catch (error) {
          stage.textContent = "Could not start";
          message.textContent = error.message || String(error);
          cancel.hidden = true;
          setFormsDisabled(false);
        }
      });
    });

    cancel.addEventListener("click", async () => {
      const endpoint = activeKind === "query" ? "/api/query-cancel" : "/api/analysis-cancel";
      cancel.disabled = true;
      cancel.textContent = "Stopping…";
      await fetch(endpoint, { method: "POST", headers: { "X-Paperwatch-Async": "1" } });
      if (pollTimer) window.clearTimeout(pollTimer);
      pollTimer = window.setTimeout(poll, 100);
    });

    const initialAnalysis = document.body.dataset.analysisRunning === "true";
    const initialQuery = document.body.dataset.queryRunning === "true";
    if (initialAnalysis || initialQuery) {
      activeKind = initialQuery ? "query" : "analysis";
      wasRunning = true;
      pollTimer = window.setTimeout(poll, 100);
    }
  }

  const picker = document.getElementById("author-picker");
  if (picker) {
    const local = document.getElementById("author-local");
    const scholar = document.getElementById("author-scholar");
    const arxiv = document.getElementById("author-arxiv");
    const openalex = document.getElementById("author-openalex");
    const track = document.getElementById("author-track");
    const trackForm = document.getElementById("author-track-form");
    const status = document.getElementById("author-status");
    const update = () => {
      const option = picker.options[picker.selectedIndex];
      local.href = option.dataset.papers || "#";
      scholar.href = option.dataset.scholar || "#";
      arxiv.href = option.dataset.arxiv || "#";
      const hasOpenAlex = Boolean(option.dataset.openalex);
      openalex.href = option.dataset.openalex || "#";
      openalex.hidden = !hasOpenAlex;
      const trackUrl = option.dataset.track || "";
      trackForm.action = trackUrl;
      trackForm.hidden = !trackUrl;
      track.disabled = !trackUrl;
      status.textContent = option.dataset.status || "";
      status.className = `status ${option.dataset.status === "tracked" ? "status-success" : "status-disabled"}`;
      document.querySelectorAll("[data-author-index]").forEach((button) => {
        button.setAttribute("aria-pressed", String(Number(button.dataset.authorIndex) === picker.selectedIndex));
      });
    };
    picker.addEventListener("change", update);
    document.querySelectorAll("[data-author-index]").forEach((button) => {
      button.addEventListener("click", () => {
        picker.selectedIndex = Number(button.dataset.authorIndex);
        update();
        picker.focus();
      });
    });
    update();
  }
})();
