(() => {
  const storageGet = (key) => {
    try { return window.localStorage.getItem(key); } catch (_) { return null; }
  };
  const storageSet = (key, value) => {
    try { window.localStorage.setItem(key, value); } catch (_) { /* Browser storage is optional. */ }
  };
  const storageRemove = (key) => {
    try { window.localStorage.removeItem(key); } catch (_) { /* Browser storage is optional. */ }
  };
  const themeToggle = document.getElementById("theme-toggle");
  const themeKey = "paperwatch-theme";
  const defaultTheme = themeToggle?.dataset.defaultTheme || "system";
  const storedTheme = storageGet(themeKey);
  if (storedTheme === "light" || storedTheme === "dark") {
    document.documentElement.dataset.theme = storedTheme;
  } else {
    delete document.documentElement.dataset.theme;
    if (storedTheme !== null) storageRemove(themeKey);
  }
  const themePreference = () => document.documentElement.dataset.theme || defaultTheme;
  const effectiveTheme = () => document.documentElement.dataset.theme ||
    (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const updateThemeButton = () => {
    if (!themeToggle) return;
    const preference = themePreference();
    const next = preference === "system" ? "light" : preference === "light" ? "dark" : "system";
    const labels = { system: "follow system", light: "light", dark: "dark" };
    const icons = { system: "◐", light: "☀", dark: "☾" };
    themeToggle.textContent = icons[preference];
    themeToggle.setAttribute("aria-label", `Theme: ${labels[preference]}. Switch to ${labels[next]}`);
    themeToggle.title = `Theme: ${labels[preference]}. Click for ${labels[next]}`;
  };
  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      const preference = themePreference();
      const next = preference === "system" ? "light" : preference === "light" ? "dark" : "system";
      if (next === "system") {
        delete document.documentElement.dataset.theme;
        storageRemove(themeKey);
      } else {
        document.documentElement.dataset.theme = next;
        storageSet(themeKey, next);
      }
      updateThemeButton();
    });
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", updateThemeButton);
    updateThemeButton();
  }

  document.querySelectorAll("form").forEach((form) => {
    const source = form.querySelector("[data-source-scope]");
    const strategy = form.querySelector("[data-pdf-strategy]");
    if (!source || !strategy) return;
    const syncSourceOptions = () => {
      const usesPdf = source.value === "fulltext";
      strategy.disabled = !usesPdf;
      if (!usesPdf) strategy.value = "auto";
      const label = strategy.closest("label");
      if (label) label.classList.toggle("control-disabled", !usesPdf);
    };
    source.addEventListener("change", syncSourceOptions);
    syncSourceOptions();
  });

  const paperTableBody = document.getElementById("paper-table-body");
  if (paperTableBody) {
    const selectionKey = "paperwatch-selected-papers";
    const hiddenInput = document.getElementById("selected-paper-ids");
    const status = document.getElementById("paper-selection-status");
    const runLabel = document.getElementById("collection-run-label");
    const clear = document.getElementById("clear-paper-selection");
    let selected = new Set();
    try {
      const saved = JSON.parse(storageGet(selectionKey) || "[]");
      if (Array.isArray(saved)) selected = new Set(saved.map(String));
    } catch (_) {
      storageRemove(selectionKey);
    }
    const rows = Array.from(paperTableBody.querySelectorAll("tr[data-paper-id]"));
    const updateSelection = () => {
      rows.forEach((row) => {
        const id = String(row.dataset.paperId);
        const checked = selected.has(id);
        const checkbox = row.querySelector(".paper-select-toggle");
        if (checkbox) checkbox.checked = checked;
        row.classList.toggle("paper-selected", checked);
      });
      const ordered = [...rows.filter((row) => selected.has(String(row.dataset.paperId))), ...rows.filter((row) => !selected.has(String(row.dataset.paperId)))];
      ordered.forEach((row) => paperTableBody.appendChild(row));
      const ids = Array.from(selected);
      storageSet(selectionKey, JSON.stringify(ids));
      if (hiddenInput) hiddenInput.value = ids.join(",");
      if (clear) clear.hidden = ids.length === 0;
      if (runLabel) runLabel.textContent = ids.length ? `Run on ${ids.length} selected paper${ids.length === 1 ? "" : "s"}` : runLabel.dataset.defaultLabel || runLabel.textContent;
      if (status) status.textContent = ids.length
        ? "Selected papers stay active across searches and are sorted to the top."
        : "Select rows to run on only those papers. Selections persist across searches.";
    };
    if (runLabel) runLabel.dataset.defaultLabel = runLabel.textContent;
    paperTableBody.querySelectorAll(".paper-select-toggle").forEach((checkbox) => {
      checkbox.addEventListener("change", () => {
        const id = String(checkbox.value);
        if (checkbox.checked) selected.add(id); else selected.delete(id);
        updateSelection();
      });
    });
    if (clear) clear.addEventListener("click", () => { selected.clear(); updateSelection(); });
    updateSelection();
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
