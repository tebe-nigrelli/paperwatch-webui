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

  document.querySelectorAll("details[data-persist-open]").forEach((details) => {
    const key = details.dataset.persistOpen;
    const saved = storageGet(key);
    if (saved === "true") details.open = true;
    else if (saved === "false") details.open = false;
    details.addEventListener("toggle", () => storageSet(key, String(details.open)));
  });

  document.querySelectorAll("[data-author-tag-combobox]").forEach((fieldset) => {
    const input = fieldset.querySelector("[data-author-tag-input]");
    const selection = fieldset.querySelector("[data-author-tag-selection]");
    const dropdown = fieldset.querySelector("[data-author-tag-options]");
    const options = Array.from(fieldset.querySelectorAll("[data-author-tag-option]"));
    const noMatch = fieldset.querySelector("[data-author-tag-no-match]");
    if (!input || !selection || !dropdown) return;

    let activeOption = null;
    const normalized = (value) => value.trim().toLocaleLowerCase();
    const selectedValues = () => new Set(
      Array.from(selection.querySelectorAll('input[name="author_tag"]')).map((item) => normalized(item.value))
    );
    const visibleOptions = () => options.filter((option) => !option.hidden);
    const syncExternalInputs = () => {
      const values = Array.from(selection.querySelectorAll('input[name="author_tag"]')).map((item) => item.value);
      document.querySelectorAll("form").forEach((form) => {
        if (form.contains(fieldset)) return;
        form.querySelectorAll('input[type="hidden"][name="author_tag"]').forEach((item) => item.remove());
        values.forEach((value) => {
          const hidden = document.createElement("input");
          hidden.type = "hidden";
          hidden.name = "author_tag";
          hidden.value = value;
          form.appendChild(hidden);
        });
      });
    };
    const setActive = (option) => {
      options.forEach((candidate) => candidate.classList.remove("active"));
      activeOption = option || null;
      if (activeOption) {
        activeOption.classList.add("active");
        input.setAttribute("aria-activedescendant", activeOption.id);
        activeOption.scrollIntoView({ block: "nearest" });
      } else {
        input.removeAttribute("aria-activedescendant");
      }
    };
    const closeDropdown = () => {
      dropdown.hidden = true;
      input.setAttribute("aria-expanded", "false");
      setActive(null);
    };
    const refreshOptions = ({ open = true } = {}) => {
      const term = normalized(input.value);
      const selected = selectedValues();
      options.forEach((option) => {
        const isSelected = selected.has(normalized(option.dataset.value || ""));
        const matches = !term || normalized(option.dataset.value || "").includes(term);
        option.hidden = isSelected || !matches;
        option.setAttribute("aria-selected", String(isSelected));
      });
      const visible = visibleOptions();
      if (noMatch) {
        noMatch.textContent = term ? "No matching tags." : "All tags are selected.";
        noMatch.hidden = options.length === 0 || visible.length > 0;
      }
      if (open) {
        dropdown.hidden = false;
        input.setAttribute("aria-expanded", "true");
      }
      setActive(visible[0] || null);
    };
    const createChip = (value) => {
      const chip = document.createElement("span");
      chip.className = "author-tag-chip";
      chip.dataset.authorTagChip = value;
      const label = document.createElement("span");
      label.textContent = value;
      const remove = document.createElement("button");
      remove.type = "button";
      remove.dataset.removeAuthorTag = value;
      remove.setAttribute("aria-label", `Remove author tag ${value}`);
      remove.textContent = "×";
      const hidden = document.createElement("input");
      hidden.type = "hidden";
      hidden.name = "author_tag";
      hidden.value = value;
      chip.append(label, remove, hidden);
      selection.insertBefore(chip, input);
    };
    const addTag = (value) => {
      const clean = value.trim();
      if (!clean || selectedValues().has(normalized(clean))) return;
      createChip(clean);
      input.value = "";
      syncExternalInputs();
      refreshOptions();
      input.focus();
      const form = fieldset.closest("form");
      if (form && (form.method || "get").toLocaleLowerCase() === "get") form.requestSubmit();
    };
    const removeTag = (value) => {
      const target = normalized(value);
      const chip = Array.from(selection.querySelectorAll("[data-author-tag-chip]")).find(
        (candidate) => normalized(candidate.dataset.authorTagChip || "") === target
      );
      if (chip) chip.remove();
      syncExternalInputs();
      refreshOptions();
      input.focus();
      const form = fieldset.closest("form");
      if (form && (form.method || "get").toLocaleLowerCase() === "get") form.requestSubmit();
    };

    input.addEventListener("focus", () => refreshOptions());
    input.addEventListener("input", () => refreshOptions());
    input.addEventListener("keydown", (event) => {
      const visible = visibleOptions();
      const index = activeOption ? visible.indexOf(activeOption) : -1;
      if (event.key === "ArrowDown") {
        event.preventDefault();
        if (dropdown.hidden) refreshOptions();
        else setActive(visible[(index + 1) % Math.max(visible.length, 1)] || null);
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        if (dropdown.hidden) refreshOptions();
        else setActive(visible[(index <= 0 ? visible.length : index) - 1] || null);
      } else if (event.key === "Enter") {
        event.preventDefault();
        if (dropdown.hidden) refreshOptions();
        if (activeOption) addTag(activeOption.dataset.value || "");
      } else if (event.key === "Escape") {
        event.preventDefault();
        closeDropdown();
      }
    });
    options.forEach((option) => {
      option.addEventListener("mousedown", (event) => event.preventDefault());
      option.addEventListener("click", () => addTag(option.dataset.value || ""));
      option.addEventListener("mousemove", () => setActive(option));
    });
    selection.addEventListener("click", (event) => {
      const remove = event.target.closest("[data-remove-author-tag]");
      if (remove) removeTag(remove.dataset.removeAuthorTag || "");
      else if (event.target !== input) input.focus();
    });
    document.addEventListener("click", (event) => {
      if (!fieldset.contains(event.target)) closeDropdown();
    });
    syncExternalInputs();
    refreshOptions({ open: false });
    closeDropdown();
  });

  document.querySelectorAll("form").forEach((form) => {
    const source = form.querySelector("select[data-source-scope]");
    const strategy = form.querySelector("select[data-pdf-strategy]");
    const context = form.querySelector("input[data-context-window]");
    const models = form.querySelector("select[data-model-picker]");
    if (!models && !context && (!source || !strategy)) return;
    const setLocked = (control, locked) => {
      if (!control) return;
      control.dataset.profileLocked = locked ? "true" : "false";
      control.setAttribute("aria-disabled", String(locked));
      if (control instanceof HTMLInputElement) {
        if (control === context) control.disabled = locked;
        else control.readOnly = locked;
      }
      const wrapper = control.previousElementSibling;
      if (wrapper?.classList.contains("search-select")) {
        wrapper.classList.toggle("control-disabled", locked);
        const input = wrapper.querySelector(".search-select-input");
        if (input) input.disabled = locked;
      }
      control.closest("label")?.classList.toggle("control-disabled", locked);
    };
    const syncSourceOptions = () => {
      if (!source || !strategy) return;
      const locked = source.dataset.profileLocked === "true";
      const usesPdf = source.value === "fulltext";
      if (!usesPdf) strategy.value = "auto";
      setLocked(strategy, locked || !usesPdf);
      strategy.dispatchEvent(new Event("change", { bubbles: true }));
    };
    const syncProfileOptions = () => {
      if (!models) {
        syncSourceOptions();
        return;
      }
      let selected = Array.from(models.selectedOptions);
      const nativeProfile = selected.find((candidate) =>
        ["codex", "claude-code"].includes(candidate.dataset.provider || "")
      );
      if (nativeProfile && selected.length > 1) {
        Array.from(models.options).forEach((candidate) => {
          candidate.selected = candidate === nativeProfile;
        });
        selected = [nativeProfile];
      }
      const native = Boolean(nativeProfile);
      if (native && source && strategy) {
        const fulltext = Array.from(source.options).find((option) => option.value === "fulltext");
        if (fulltext) source.value = "fulltext";
        strategy.value = nativeProfile.dataset.pdfStrategy || "direct";
      }
      if (native && context && nativeProfile.dataset.contextWindow) {
        context.value = nativeProfile.dataset.contextWindow;
      }
      setLocked(source, Boolean(native));
      setLocked(strategy, Boolean(native));
      setLocked(context, Boolean(native));
      source?.dispatchEvent(new Event("change", { bubbles: true }));
      strategy?.dispatchEvent(new Event("change", { bubbles: true }));
    };
    source?.addEventListener("change", syncSourceOptions);
    models?.addEventListener("change", syncProfileOptions);
    syncProfileOptions();
  });

  const automaticQueryForm = document.querySelector('form[data-auto-save="query-selections"]');
  if (automaticQueryForm) {
    const status = document.getElementById("automatic-query-status");
    let saveTimer = null;
    let saving = false;
    let dirty = false;

    const setStatus = (message, failed = false) => {
      if (!status) return;
      status.textContent = message;
      status.classList.toggle("task-error", failed);
      status.classList.toggle("muted", !failed);
    };
    const saveSelections = async () => {
      if (saving) {
        dirty = true;
        return;
      }
      saving = true;
      dirty = false;
      setStatus("Saving…");
      try {
        const response = await fetch(automaticQueryForm.action, {
          method: "POST",
          body: new FormData(automaticQueryForm),
          headers: { "X-Paperwatch-Async": "1", "Accept": "application/json" },
        });
        const payload = await response.json();
        if (!response.ok || !payload.ok) throw new Error(payload.message || "Could not save selections");
        setStatus("Saved");
      } catch (error) {
        setStatus(error.message || "Could not save selections", true);
      } finally {
        saving = false;
        if (dirty) saveSelections();
      }
    };
    Array.from(automaticQueryForm.elements)
      .filter((element) => element.type === "checkbox")
      .forEach((checkbox) => checkbox.addEventListener("change", () => {
        setStatus("Unsaved changes");
        if (saveTimer) window.clearTimeout(saveTimer);
        saveTimer = window.setTimeout(saveSelections, 180);
      }));
  }

  const paperTableBody = document.getElementById("paper-table-body");
  if (paperTableBody) {
    const hiddenInput = document.getElementById("selected-paper-ids");
    const status = document.getElementById("paper-selection-status");
    const runLabel = document.getElementById("collection-run-label");
    const clear = document.getElementById("clear-paper-selection");
    const persistentDock = document.querySelector(".selected-content-dock");
    if (persistentDock) {
      const selectedIds = Array.from(
        persistentDock.querySelectorAll(".paper-select-toggle:checked")
      ).map((checkbox) => String(checkbox.value));
      if (hiddenInput) hiddenInput.value = selectedIds.join(",");
      if (clear) clear.hidden = true;
      if (runLabel && selectedIds.length) {
        runLabel.textContent = `Run on ${selectedIds.length} selected paper${selectedIds.length === 1 ? "" : "s"}`;
      }
      if (status) {
        status.textContent = selectedIds.length
          ? "The persistent selection is used. Change it with the first column or the selected-content panel."
          : "No papers are persistently selected; the run uses the visible paper results.";
      }
    } else {
    const selectionKey = "paperwatch-selected-papers";
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
        const hidden = Boolean(term) && !row.dataset.authorSearch.toLocaleLowerCase().includes(term);
        row.hidden = hidden;
        if (row.nextElementSibling?.matches("[data-author-notes-row]")) {
          row.nextElementSibling.hidden = hidden;
        }
      });
    };
    authorFilter.addEventListener("input", applyAuthorFilter);
  }

  if (paperJobs) {
    const paperId = Number(paperJobs.dataset.paperId || 0);
    const blogEntryId = Number(paperJobs.dataset.blogEntryId || 0);
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
        if (paperId && analysis.running && Number(analysis.paper_id) === paperId) { kind = "analysis"; snapshot = analysis; }
        else if (query.running && (
          (paperId && Number(query.paper_id) === paperId) ||
          (blogEntryId && Number(query.blog_entry_id) === blogEntryId)
        )) { kind = "query"; snapshot = query; }
        const relevant = Boolean(snapshot.batch_id) && (
          (paperId && Number(snapshot.paper_id) === paperId) ||
          (blogEntryId && kind === "query" && Number(snapshot.blog_entry_id) === blogEntryId)
        );
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

  const runningJobsBody = document.getElementById("running-jobs-body");
  if (runningJobsBody) {
    const liveRows = Array.from(runningJobsBody.querySelectorAll("tr[data-live-kind]"))
      .filter((row) => row.dataset.liveKind);
    if (liveRows.length) {
      const pollRunning = async () => {
        try {
          const response = await fetch("/api/running-status", { cache: "no-store" });
          if (!response.ok) throw new Error("status unavailable");
          const payload = await response.json();
          let anyRunning = false;
          liveRows.forEach((row) => {
            const snapshot = payload[row.dataset.liveKind] || {};
            anyRunning = anyRunning || Boolean(snapshot.running);
            const progress = row.querySelector("[data-live-progress]");
            const status = row.querySelector("[data-live-status]");
            let message = row.querySelector("[data-live-message]");
            if (progress) {
              progress.textContent = snapshot.progress_total
                ? `${snapshot.progress_current || 0}/${snapshot.progress_total}`
                : "—";
            }
            if (status) {
              const value = snapshot.cancel_requested
                ? "stopping"
                : (snapshot.running ? "running" : (snapshot.status || "complete"));
              status.textContent = value;
              status.className = `status status-${value}`;
            }
            if (!message && snapshot.message) {
              message = document.createElement("small");
              message.dataset.liveMessage = "";
              row.cells[3]?.appendChild(message);
            }
            if (message) message.textContent = snapshot.message || "";
          });
          if (anyRunning) window.setTimeout(pollRunning, 600);
          else window.location.reload();
        } catch (_) {
          window.setTimeout(pollRunning, 1500);
        }
      };
      window.setTimeout(pollRunning, 200);
    }
  }

  document.querySelectorAll("[data-copy-source]").forEach((button) => {
    button.addEventListener("click", async () => {
      const source = document.getElementById(button.dataset.copySource || "");
      if (!source) return;
      const text = "value" in source ? source.value : source.textContent || "";
      const originalHtml = button.innerHTML;
      const originalLabel = button.getAttribute("aria-label") || "Copy text";
      const showState = (label, failed = false) => {
        button.classList.toggle("copy-failed", failed);
        if (button.hasAttribute("data-copy-icon")) {
          button.setAttribute("aria-label", label);
          button.title = label;
          button.dataset.copyState = label;
        } else {
          button.textContent = label;
        }
      };
      try {
        if (navigator.clipboard && window.isSecureContext) {
          await navigator.clipboard.writeText(text);
        } else {
          const temporary = document.createElement("textarea");
          temporary.value = text;
          temporary.setAttribute("readonly", "");
          temporary.style.position = "fixed";
          temporary.style.opacity = "0";
          document.body.appendChild(temporary);
          temporary.select();
          document.execCommand("copy");
          temporary.remove();
        }
        showState("Copied");
      } catch (_) {
        showState("Copy failed", true);
      }
      window.setTimeout(() => {
        button.innerHTML = originalHtml;
        button.setAttribute("aria-label", originalLabel);
        button.title = originalLabel;
        delete button.dataset.copyState;
        button.classList.remove("copy-failed");
      }, 1600);
    });
  });

  const enhanceSelect = (select, index) => {
    if (select.dataset.searchSelectReady === "true") return;
    select.dataset.searchSelectReady = "true";
    const multiple = select.multiple;
    const enabledOptions = Array.from(select.options).filter((option) => !option.disabled);
    const cycle = !multiple && enabledOptions.length > 0 && enabledOptions.length <= 5;
    const form = select.form || select.closest("form");
    const autoApply = () => {
      if (form && (form.method || "get").toLocaleLowerCase() === "get") form.requestSubmit();
    };
    const wrapper = document.createElement("div");
    wrapper.className = `search-select${multiple ? " search-select-multiple" : ""}${cycle ? " search-select-cycle" : ""}`;
    const input = document.createElement("input");
    input.type = "search";
    input.className = "search-select-input";
    input.autocomplete = "off";
    input.setAttribute("role", "combobox");
    input.setAttribute("aria-autocomplete", "list");
    input.setAttribute("aria-expanded", "false");
    input.placeholder = select.dataset.searchPlaceholder || (multiple ? "Type to filter choices" : "Type to filter");
    if (cycle) input.readOnly = true;
    const list = document.createElement("div");
    list.className = "search-select-options";
    list.id = `search-select-options-${index}`;
    list.setAttribute("role", "listbox");
    if (multiple) list.setAttribute("aria-multiselectable", "true");
    list.hidden = true;
    input.setAttribute("aria-controls", list.id);
    const chips = document.createElement("div");
    chips.className = "search-select-chips";
    if (!multiple) chips.hidden = true;
    wrapper.append(input, chips, list);
    select.before(wrapper);
    select.classList.add("search-select-native");
    select.setAttribute("aria-hidden", "true");
    select.tabIndex = -1;

    let active = -1;
    const normalized = (value) => String(value || "").trim().toLocaleLowerCase();
    const maxRenderedOptions = 80;
    let optionRows = [];
    let renderOptionRows = () => {};
    const visibleRows = () => optionRows;
    const setActive = (rowIndex) => {
      const visible = visibleRows();
      optionRows.forEach((row) => row.classList.remove("active"));
      if (!visible.length) {
        active = -1;
        input.removeAttribute("aria-activedescendant");
        return;
      }
      active = Math.max(0, Math.min(rowIndex, visible.length - 1));
      const row = visible[active];
      if (!row.id) row.id = `${list.id}-option-${row.dataset.optionIndex}`;
      row.classList.add("active");
      input.setAttribute("aria-activedescendant", row.id);
      row.scrollIntoView({ block: "nearest" });
    };
    const selectedOptions = () => Array.from(select.selectedOptions);
    const syncInput = (force = false) => {
      if (!multiple && (force || cycle || document.activeElement !== input)) {
        input.value = select.selectedIndex >= 0 ? select.options[select.selectedIndex].textContent || "" : "";
      }
      chips.replaceChildren();
      if (multiple) {
        selectedOptions().forEach((option) => {
          const chip = document.createElement("button");
          chip.type = "button";
          chip.className = "search-select-chip";
          chip.textContent = `${option.textContent || option.value} ×`;
          chip.setAttribute("aria-label", `Remove ${option.textContent || option.value}`);
          chip.addEventListener("click", () => {
            option.selected = false;
            select.dispatchEvent(new Event("change", { bubbles: true }));
            syncInput();
            refresh(true);
            autoApply();
          });
          chips.appendChild(chip);
        });
      }
      optionRows.forEach((row) => {
        const selected = select.options[Number(row.dataset.optionIndex)]?.selected || false;
        row.classList.toggle("selected", selected);
        row.setAttribute("aria-selected", String(selected));
      });
      input.disabled = select.disabled;
      wrapper.classList.toggle("control-disabled", select.disabled);
    };
    const refresh = (open = true) => {
      renderOptionRows();
      if (open && !select.disabled) {
        list.hidden = false;
        input.setAttribute("aria-expanded", "true");
      }
      setActive(0);
    };
    const close = () => {
      list.hidden = true;
      input.setAttribute("aria-expanded", "false");
      active = -1;
      optionRows.forEach((row) => row.classList.remove("active"));
      if (!multiple) syncInput();
      else input.value = "";
    };
    const choose = (row) => {
      const optionIndex = Number(row.dataset.optionIndex);
      const option = select.options[optionIndex];
      if (!option || option.disabled) return;
      if (multiple) {
        if (select.matches("[data-model-picker]") && !option.selected) {
          const nativeProviders = new Set(["codex", "claude-code"]);
          const choosingNative = nativeProviders.has(option.dataset.provider || "");
          const hasNative = Array.from(select.selectedOptions).some((candidate) =>
            nativeProviders.has(candidate.dataset.provider || "")
          );
          if (choosingNative || hasNative) {
            Array.from(select.options).forEach((candidate) => {
              candidate.selected = false;
            });
          }
        }
        option.selected = !option.selected;
      } else if (option.selected) {
        const emptyOption = Array.from(select.options).find((candidate) => candidate.value === "");
        if (emptyOption) emptyOption.selected = true;
        else select.selectedIndex = -1;
        input.value = "";
      } else {
        select.selectedIndex = optionIndex;
        input.value = option.textContent || option.value;
      }
      select.dispatchEvent(new Event("change", { bubbles: true }));
      syncInput(true);
      if (multiple) {
        input.value = "";
        refresh(true);
        input.focus();
      } else {
        close();
        input.focus();
      }
      autoApply();
    };
    renderOptionRows = () => {
      const term = normalized(input.value);
      const fragment = document.createDocumentFragment();
      const rows = [];
      let matches = 0;
      Array.from(select.options).forEach((option, optionIndex) => {
        if (option.disabled) return;
        const label = option.textContent || option.value;
        if (term && !normalized(option?.textContent).includes(term) && !normalized(option?.value).includes(term)) return;
        matches += 1;
        if (rows.length >= maxRenderedOptions) return;
        const button = document.createElement("button");
        button.type = "button";
        button.className = "search-select-option";
        button.dataset.optionIndex = String(optionIndex);
        button.setAttribute("role", "option");
        button.textContent = label;
        button.addEventListener("mousedown", (event) => event.preventDefault());
        button.addEventListener("mousemove", () => setActive(optionRows.indexOf(button)));
        button.addEventListener("click", () => choose(button));
        rows.push(button);
        fragment.appendChild(button);
      });
      if (matches > maxRenderedOptions) {
        const message = document.createElement("div");
        message.className = "search-select-limit";
        message.textContent = `${matches - maxRenderedOptions} more options. Keep typing to narrow the list.`;
        fragment.appendChild(message);
      }
      optionRows = rows;
      list.replaceChildren(fragment);
      syncInput();
    };
    input.addEventListener("focus", () => {
      if (cycle) return;
      if (!multiple) input.select();
      refresh(true);
    });
    input.addEventListener("click", (event) => {
      if (cycle) {
        event.preventDefault();
        const current = enabledOptions.indexOf(select.options[select.selectedIndex]);
        const next = enabledOptions[(current + 1 + enabledOptions.length) % enabledOptions.length];
        if (!next) return;
        select.value = next.value;
        select.dispatchEvent(new Event("change", { bubbles: true }));
        syncInput(true);
        autoApply();
        return;
      }
      if (!multiple && select.selectedIndex >= 0) {
        const selectedText = select.options[select.selectedIndex].textContent || "";
        if (input.value === selectedText) {
          input.value = "";
          refresh(true);
        }
      }
    });
    input.addEventListener("input", () => refresh(true));
    input.addEventListener("keydown", (event) => {
      const visible = visibleRows();
      if (cycle && ["ArrowDown", "ArrowUp", "Enter", " "].includes(event.key)) {
        event.preventDefault();
        const current = enabledOptions.indexOf(select.options[select.selectedIndex]);
        const direction = event.key === "ArrowUp" ? -1 : 1;
        const next = enabledOptions[(current + direction + enabledOptions.length) % enabledOptions.length];
        if (next) {
          select.value = next.value;
          select.dispatchEvent(new Event("change", { bubbles: true }));
          syncInput(true);
          autoApply();
        }
        return;
      }
      if (event.key === "ArrowDown") {
        event.preventDefault();
        if (list.hidden) refresh(true);
        else setActive((active + 1) % Math.max(visible.length, 1));
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        if (list.hidden) refresh(true);
        else setActive(active <= 0 ? visible.length - 1 : active - 1);
      } else if (event.key === "Enter") {
        event.preventDefault();
        if (list.hidden) refresh(true);
        const row = visibleRows()[active >= 0 ? active : 0];
        if (row) {
          const option = select.options[Number(row.dataset.optionIndex)];
          if (option && !multiple && !option.selected) input.value = option.textContent || option.value;
          choose(row);
        }
      } else if (event.key === "Escape") {
        event.preventDefault();
        close();
      }
    });
    document.addEventListener("click", (event) => {
      if (!wrapper.contains(event.target)) close();
    });
    const observer = new MutationObserver(syncInput);
    observer.observe(select, { attributes: true, attributeFilter: ["disabled"] });
    select.addEventListener("change", () => syncInput(true));
    syncInput();
  };

  document.querySelectorAll("select").forEach((select, index) => enhanceSelect(select, index));

  document.querySelectorAll("[data-display-stubs]").forEach((toggle) => {
    toggle.addEventListener("change", () => {
      const form = toggle.form || toggle.closest("form");
      if (form && (form.method || "get").toLocaleLowerCase() === "get") form.requestSubmit();
    });
  });
  document.querySelectorAll("[data-model-picker]").forEach((select) => {
    select.dispatchEvent(new Event("change", { bubbles: true }));
  });

  document.querySelectorAll("[data-cycle-toggle]").forEach((button) => {
    const input = button.parentElement?.querySelector("[data-cycle-value]");
    if (!input) return;
    let options = [];
    try {
      options = JSON.parse(button.dataset.cycleOptions || "[]");
    } catch (_error) {
      options = [];
    }
    if (!Array.isArray(options) || options.length === 0) return;
    const render = () => {
      const option = options.find((candidate) => candidate.value === input.value) || options[0];
      input.value = option.value;
      button.textContent = option.label;
      button.setAttribute("aria-label", `Supported content: ${option.label}. Click to change.`);
    };
    button.addEventListener("click", () => {
      const current = Math.max(0, options.findIndex((candidate) => candidate.value === input.value));
      const next = options[(current + 1) % options.length];
      input.value = next.value;
      render();
      input.dispatchEvent(new Event("change", { bubbles: true }));
    });
    render();
  });

  const queryKindInput = document.querySelector('input[name="query_kind"][data-cycle-value]');
  const classificationSettings = document.querySelector("[data-classification-settings]");
  if (queryKindInput && classificationSettings) {
    const updateClassificationSettings = () => {
      classificationSettings.hidden = queryKindInput.value !== "classification";
    };
    queryKindInput.addEventListener("change", updateClassificationSettings);
    updateClassificationSettings();
  }

  document.querySelectorAll("[data-dialog-open]").forEach((button) => {
    button.addEventListener("click", () => {
      const dialog = document.getElementById(button.dataset.dialogOpen || "");
      if (dialog?.showModal) dialog.showModal();
    });
  });
  document.querySelectorAll("dialog [data-dialog-close]").forEach((button) => {
    button.addEventListener("click", () => button.closest("dialog")?.close());
  });
  document.querySelectorAll("dialog").forEach((dialog) => {
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) dialog.close();
    });
  });

  document.querySelectorAll("[data-auto-read]").forEach((element) => {
    const contentType = element.dataset.contentType;
    const contentId = element.dataset.contentId;
    if (!contentType || !contentId) return;
    let timer = window.setTimeout(async () => {
      timer = null;
      const body = new FormData();
      body.set("read", "1");
      try {
        await fetch(`/content/${encodeURIComponent(contentType)}/${encodeURIComponent(contentId)}/read`, {
          method: "POST",
          body,
          headers: { "X-Paperwatch-Async": "1", "Accept": "application/json" },
        });
      } catch (_) {
        // Reading state is a convenience signal; a later visit can retry it.
      }
    }, 5000);
    window.addEventListener("pagehide", () => {
      if (timer !== null) window.clearTimeout(timer);
    }, { once: true });
  });

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


  let dynamicSelectIndex = 10000;
  const initializeDynamicSelects = (root = document) => {
    root.querySelectorAll("select").forEach((select) => enhanceSelect(select, dynamicSelectIndex++));
  };

  document.addEventListener("click", (event) => {
    const button = event.target.closest("[data-filter-cycle]");
    if (!button) return;
    event.preventDefault();
    const form = document.getElementById(button.dataset.filterForm || "");
    const name = button.dataset.filterName || "";
    const values = (button.dataset.filterValues || "").split("|").filter(Boolean);
    const labels = (button.dataset.filterLabels || "").split("|");
    const input = form ? Array.from(form.elements).find((control) => control.name === name) : null;
    if (!form || !input || values.length === 0) return;
    const currentValue = String(input.value || "");
    const currentState = currentValue || "any";
    const currentIndex = Math.max(0, values.indexOf(currentState));
    const nextState = values[(currentIndex + 1) % values.length];
    input.value = nextState === "any" ? "" : nextState;
    button.dataset.filterState = nextState;
    const label = labels[values.indexOf(nextState)] || nextState;
    const prefixes = {
      starred: "Star filter",
      content: "Content type",
      read: "Reading filter",
      notes: "Notes filter",
      enabled: "Fetch state",
    };
    const prefix = prefixes[name] || "Filter";
    button.title = `${prefix}: ${label.toLocaleLowerCase()}`;
    button.setAttribute("aria-label", `${prefix}: ${label.toLocaleLowerCase()}. Click to change.`);
    form.requestSubmit();
  });

  const parseHtml = (text) => new DOMParser().parseFromString(text, "text/html");
  const replaceFromResponse = async (response, selector) => {
    if (!response.ok) throw new Error(`Request failed with HTTP ${response.status}`);
    const doc = parseHtml(await response.text());
    const incoming = doc.querySelector(selector);
    const current = document.querySelector(selector);
    if (!incoming || !current) return false;
    current.replaceWith(incoming);
    initializeDynamicSelects(incoming);
    return true;
  };
  const contentUrlKey = "paperwatch-content-url";
  const contentHome = document.querySelector("[data-content-home-link]");
  const storedContentUrl = storageGet(contentUrlKey);
  if (contentHome && storedContentUrl) contentHome.href = storedContentUrl;
  const rememberContentUrl = (url) => {
    const parsed = new URL(url, window.location.href);
    const value = `${parsed.pathname}${parsed.search}`;
    storageSet(contentUrlKey, value);
    if (contentHome) contentHome.href = value;
  };
  if (document.querySelector("#content-page-root")) rememberContentUrl(window.location.href);
  const refreshContentPage = async (url = window.location.href, push = false) => {
    const response = await fetch(url, { headers: { "X-Paperwatch-Partial": "1" } });
    if (!response.ok) throw new Error(`Request failed with HTTP ${response.status}`);
    const doc = parseHtml(await response.text());
    const incomingResults = doc.querySelector("#content-results-root");
    const currentResults = document.querySelector("#content-results-root");
    if (!incomingResults || !currentResults) return false;
    currentResults.replaceWith(incomingResults);
    const incomingCount = doc.querySelector("#content-match-count");
    const currentCount = document.querySelector("#content-match-count");
    if (incomingCount && currentCount) currentCount.replaceWith(incomingCount);
    const incomingForm = doc.querySelector("#content-column-filters");
    const currentForm = document.querySelector("#content-column-filters");
    if (incomingForm && currentForm) currentForm.replaceWith(incomingForm);
    const incomingSearch = doc.querySelector('input[name="q"][form="content-column-filters"]');
    const currentSearch = document.querySelector('input[name="q"][form="content-column-filters"]');
    if (incomingSearch && currentSearch) currentSearch.value = incomingSearch.value;
    initializeDynamicSelects(incomingResults);
    const finalUrl = new URL(url, window.location.href).toString();
    if (push) window.history.pushState({}, "", finalUrl);
    rememberContentUrl(finalUrl);
    return true;
  };
  const refreshSelector = async (selector) => {
    const response = await fetch(window.location.href, { headers: { "X-Paperwatch-Partial": "1" } });
    return replaceFromResponse(response, selector);
  };
  const postAsync = async (form) => {
    const response = await fetch(form.action, {
      method: (form.method || "post").toUpperCase(),
      body: new FormData(form),
      headers: { "X-Paperwatch-Async": "1", "Accept": "application/json" },
    });
    let payload = {};
    try { payload = await response.json(); } catch (_) { payload = {}; }
    if (!response.ok || payload.ok === false) throw new Error(payload.message || `Request failed with HTTP ${response.status}`);
    return payload;
  };

  const selectedDock = () => document.querySelector(".selected-content-dock");
  const selectedBody = () => document.querySelector("[data-selected-content-body]");
  const updateSelectedCount = (delta = null) => {
    const dock = selectedDock();
    const body = selectedBody();
    if (!dock || !body) return;
    const visible = body.querySelectorAll("tr.content-row").length;
    const stored = Number(dock.dataset.selectedCount || visible);
    const count = delta === null ? visible : Math.max(0, stored + delta);
    dock.dataset.selectedCount = String(count);
    const label = dock.querySelector(".selection-count");
    if (label) label.textContent = String(count);
    dock.hidden = count === 0;
  };
  const configureSelectedRow = (row, selected) => {
    row.classList.toggle("content-row-selected", selected);
    const selectForm = row.querySelector('form[action$="/select"]');
    const input = selectForm?.querySelector('input[name="selected"]');
    const button = selectForm?.querySelector(".content-select-button");
    if (input) input.value = selected ? "0" : "1";
    if (button) {
      button.textContent = selected ? "☑" : "☐";
      button.title = selected ? "Remove from selection" : "Keep in selection";
    }
  };
  const syncSelectionRow = (form, selected) => {
    const row = form.closest("tr.content-row");
    const body = selectedBody();
    if (!row || !body) return;
    const key = `${row.dataset.contentType}:${row.dataset.contentId}`;
    const existing = Array.from(body.querySelectorAll("tr.content-row")).find(
      (candidate) => `${candidate.dataset.contentType}:${candidate.dataset.contentId}` === key
    );
    if (selected) {
      const clone = existing || row.cloneNode(true);
      configureSelectedRow(clone, true);
      if (!existing) body.appendChild(clone);
      if (!row.closest("[data-selected-content-body]")) row.remove();
      updateSelectedCount(existing ? 0 : 1);
    } else {
      (existing || row).remove();
      updateSelectedCount(-1);
    }
  };

  const updateRowsForBulkAction = (form) => {
    const action = new URL(form.action).pathname;
    const rows = Array.from(selectedBody()?.querySelectorAll("tr.content-row") || []);
    if (action.endsWith("/unselect")) {
      rows.forEach((row) => row.remove());
      const dock = selectedDock();
      if (dock) dock.dataset.selectedCount = "0";
      updateSelectedCount(0);
      return;
    }
    if (action.endsWith("/star") || action.endsWith("/unstar")) {
      const starred = action.endsWith("/star");
      rows.forEach((row) => {
        const button = row.querySelector(".star-button");
        const input = row.querySelector('input[name="starred"]');
        if (button) { button.classList.toggle("is-starred", starred); button.textContent = starred ? "★" : "☆"; }
        if (input) input.value = starred ? "0" : "1";
      });
      const button = form.querySelector(".star-button");
      form.setAttribute("action", `/content/selected/${starred ? "unstar" : "star"}`);
      if (button) {
        const label = starred ? "Unstar all selected content" : "Star all selected content";
        button.classList.toggle("is-starred", starred);
        button.textContent = starred ? "★" : "☆";
        button.title = label;
        button.setAttribute("aria-label", label);
      }
    }
    if (action.endsWith("/read") || action.endsWith("/unread")) {
      const read = action.endsWith("/read");
      rows.forEach((row) => { row.dataset.read = read ? "true" : "false"; });
      const button = form.querySelector(".selected-read-button");
      form.setAttribute("action", `/content/selected/${read ? "unread" : "read"}`);
      if (button) {
        const label = read ? "Mark all selected content unread" : "Mark all selected content read";
        button.classList.toggle("is-read", read);
        button.textContent = read ? "✓" : "○";
        button.title = label;
        button.setAttribute("aria-label", label);
      }
    }
  };

  document.addEventListener("submit", async (event) => {
    const form = event.target.closest("form");
    if (!form) return;
    if (form.matches("[data-content-filter]")) {
      event.preventDefault();
      const url = new URL(form.action || window.location.href, window.location.href);
      const params = new URLSearchParams(new FormData(form));
      params.delete("page");
      for (const [key, value] of Array.from(params.entries())) if (!String(value).trim()) params.delete(key);
      url.search = params.toString();
      try { await refreshContentPage(url.toString(), true); } catch (error) { console.error(error); }
      return;
    }
    if (form.matches("[data-select-results]")) {
      event.preventDefault();
      try {
        await postAsync(form);
        await refreshContentPage(window.location.href, false);
      } catch (error) { console.error(error); }
      return;
    }
    if (form.matches("[data-content-state]")) {
      event.preventDefault();
      try {
        const payload = await postAsync(form);
        if ("starred" in payload) {
          const input = form.querySelector('input[name="starred"]');
          const button = form.querySelector(".star-button");
          if (input) input.value = payload.starred ? "0" : "1";
          if (button) {
            const active = Boolean(payload.starred);
            const label = active
              ? (button.dataset.starredLabel || "Remove star")
              : (button.dataset.unstarredLabel || "Star content");
            button.classList.toggle("is-starred", active);
            button.textContent = active ? "★" : "☆";
            button.title = label;
            button.setAttribute("aria-label", label);
          }
        } else if ("read" in payload) {
          const input = form.querySelector('input[name="read"]');
          const button = form.querySelector(".read-state-button");
          const row = form.closest(".content-row");
          if (input) input.value = payload.read ? "0" : "1";
          if (button) button.textContent = payload.read ? "Read" : "Unread";
          if (row) row.dataset.read = payload.read ? "true" : "false";
        } else if ("selected" in payload) {
          syncSelectionRow(form, Boolean(payload.selected));
        }
      } catch (error) { console.error(error); }
      return;
    }
    if (form.matches("[data-async-action]")) {
      event.preventDefault();
      try {
        await postAsync(form);
        if (form.closest(".selected-content-dock")) updateRowsForBulkAction(form);
        const selector = form.dataset.refreshSelector;
        if (selector) await refreshSelector(selector);
        else {
          const panel = form.closest(".content-tags-panel, .classification-panel, .object-notes-panel");
          if (panel && panel.classList.length) {
            const className = Array.from(panel.classList).find((name) => name.endsWith("-panel"));
            if (className) await refreshSelector(`.${className}`);
          }
        }
      } catch (error) { console.error(error); }
    }
  });

  document.addEventListener("click", async (event) => {
    const link = event.target.closest("a[data-content-page]");
    if (!link) return;
    event.preventDefault();
    try { await refreshContentPage(link.href, true); } catch (error) { console.error(error); }
  });
  document.addEventListener("click", (event) => {
    const reset = event.target.closest("[data-content-reset]");
    if (!reset || !document.querySelector("#content-page-root")) return;
    event.preventDefault();
    refreshContentPage(reset.href, true).catch(console.error);
  });
  window.addEventListener("popstate", () => {
    if (document.querySelector("#content-page-root")) refreshContentPage(window.location.href, false).catch(console.error);
  });

  document.addEventListener("keydown", (event) => {
    const form = event.target.closest("form[data-enter-submit]");
    if (!form || event.key !== "Enter" || event.shiftKey || event.isComposing) return;
    if (event.target instanceof HTMLTextAreaElement) return;
    event.preventDefault();
    form.requestSubmit();
  });

  document.addEventListener("change", (event) => {
    const input = event.target.closest("[data-auto-upload]");
    if (input?.files?.length) input.form?.requestSubmit();
  });

  document.addEventListener("click", (event) => {
    const edit = event.target.closest("[data-notes-edit]");
    const cancel = event.target.closest("[data-notes-cancel]");
    const control = edit || cancel;
    if (!control) return;
    const panel = control.closest("[data-inline-notes]");
    const display = panel?.querySelector("[data-notes-display]");
    const form = panel?.querySelector("[data-notes-form]");
    if (edit) {
      if (display) display.hidden = true;
      if (form) { form.hidden = false; form.querySelector("textarea")?.focus(); }
    } else {
      if (form) form.hidden = true;
      if (display) display.hidden = false;
    }
  });

  const pdfDropTarget = document.querySelector("[data-pdf-drop-target]");
  const pdfUploadForm = document.querySelector("[data-pdf-upload-form]");
  if (pdfDropTarget && pdfUploadForm) {
    const input = pdfUploadForm.querySelector('input[type="file"]');
    ["dragenter", "dragover"].forEach((name) => pdfDropTarget.addEventListener(name, (event) => {
      event.preventDefault();
      pdfDropTarget.classList.add("pdf-drop-active");
    }));
    ["dragleave", "drop"].forEach((name) => pdfDropTarget.addEventListener(name, (event) => {
      event.preventDefault();
      pdfDropTarget.classList.remove("pdf-drop-active");
    }));
    pdfDropTarget.addEventListener("drop", (event) => {
      const files = Array.from(event.dataTransfer?.files || []);
      const pdf = files.find((file) => file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf"));
      if (!pdf || !input) return;
      const transfer = new DataTransfer();
      transfer.items.add(pdf);
      input.files = transfer.files;
      pdfUploadForm.requestSubmit();
    });
  }

  const escapeHtml = (value) => value.replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
  const compileDebugRegexp = (pattern) => {
    let flags = "g";
    let source = pattern.trim();
    if (source.startsWith("(?i)")) { flags += "i"; source = source.slice(4); }
    return new RegExp(source, flags);
  };
  const highlightMatches = (text, patterns) => {
    const ranges = [];
    patterns.forEach((pattern) => {
      if (!pattern.trim()) return;
      const regexp = compileDebugRegexp(pattern);
      let match;
      let guard = 0;
      while ((match = regexp.exec(text)) && guard++ < 1000) {
        if (!match[0]) { regexp.lastIndex += 1; continue; }
        ranges.push([match.index, match.index + match[0].length]);
      }
    });
    ranges.sort((a, b) => a[0] - b[0]);
    const merged = [];
    ranges.forEach((range) => {
      const previous = merged[merged.length - 1];
      if (previous && range[0] <= previous[1]) previous[1] = Math.max(previous[1], range[1]);
      else merged.push([...range]);
    });
    let cursor = 0;
    return merged.map(([start, end]) => {
      const before = escapeHtml(text.slice(cursor, start));
      const match = `<mark>${escapeHtml(text.slice(start, end))}</mark>`;
      cursor = end;
      return before + match;
    }).join("") + escapeHtml(text.slice(cursor));
  };
  document.querySelectorAll("[data-regexp-debug]").forEach((panel) => {
    const mode = panel.dataset.regexpDebug;
    const source = document.querySelector(`[data-regexp-source="${mode}"]`);
    const input = panel.querySelector("[data-regexp-test-input]");
    const output = panel.querySelector("[data-regexp-test-output]");
    if (!source || !input || !output) return;
    let timer;
    const render = () => {
      const raw = source.value || "";
      const patterns = mode === "classification"
        ? raw.split(/\r?\n/).map((line) => line.includes("=") ? line.slice(line.indexOf("=") + 1).trim() : "").filter(Boolean)
        : [raw];
      if (!patterns.length || !input.value) { output.textContent = "No matches to test."; return; }
      try { output.innerHTML = highlightMatches(input.value, patterns) || escapeHtml(input.value); }
      catch (error) { output.textContent = `Invalid regexp: ${error.message}`; }
    };
    const schedule = () => { window.clearTimeout(timer); timer = window.setTimeout(render, 120); };
    source.addEventListener("input", schedule);
    input.addEventListener("input", schedule);
    render();
  });

  const syncDialogScroll = () => document.body.classList.toggle("dialog-open", Boolean(document.querySelector("dialog[open]")));
  document.querySelectorAll("dialog").forEach((dialog) => {
    dialog.addEventListener("close", syncDialogScroll);
    dialog.addEventListener("cancel", syncDialogScroll);
  });
  document.addEventListener("click", (event) => {
    if (event.target.closest("[data-dialog-open]")) window.setTimeout(syncDialogScroll, 0);
  });

})();
