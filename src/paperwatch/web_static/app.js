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

  document.querySelectorAll("[data-app-uptime]").forEach((display) => {
    const initialSeconds = Number.parseInt(display.dataset.uptimeSeconds || "0", 10);
    const startedAt = Date.now() - Math.max(0, initialSeconds) * 1000;
    const renderUptime = () => {
      const totalMinutes = Math.floor((Date.now() - startedAt) / 60000);
      const days = Math.floor(totalMinutes / 1440);
      const hours = Math.floor((totalMinutes % 1440) / 60);
      const minutes = totalMinutes % 60;
      display.textContent = [days, hours, minutes]
        .map((value) => String(value).padStart(2, "0"))
        .join(":");
    };
    renderUptime();
    window.setInterval(renderUptime, 1000);
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

  document.querySelectorAll("form[data-profile-controls]").forEach((form) => {
    const processing = form.querySelector("select[data-paper-processing]");
    const context = form.querySelector("input[data-context-window]");
    const models = form.querySelector("select[data-model-picker]");
    const defaults = form.querySelector("select[data-query-defaults]");
    const defaultsToggle = form.querySelector("input[data-query-defaults-toggle]");
    const queryKind = form.querySelector('input[name="query_kind"][data-cycle-value]');
    if (!models && !context && !processing) return;

    const setLocked = (control, locked) => {
      if (!control) return;
      control.dataset.profileLocked = locked ? "true" : "false";
      control.setAttribute("aria-disabled", String(locked));
      control.disabled = locked;
      const wrapper = control.previousElementSibling;
      if (wrapper?.classList.contains("search-select")) {
        wrapper.classList.toggle("control-disabled", locked);
        const input = wrapper.querySelector(".search-select-input");
        if (input) input.disabled = locked;
      }
      control.closest("label")?.classList.toggle("control-disabled", locked);
    };

    let syncingProfileOptions = false;
    const selectedModels = () => models ? Array.from(models.selectedOptions) : [];
    const selectedQueries = () => defaults ? Array.from(defaults.selectedOptions) : [];
    const defaultsLocked = () => Boolean(defaultsToggle && !defaultsToggle.checked);
    const localQuery = () => {
      if (queryKind?.value === "regexp") return true;
      const selected = selectedQueries();
      return selected.length > 0 && selected.every((option) => option.dataset.local === "true");
    };

    const syncProfileOptions = () => {
      if (syncingProfileOptions) return;
      syncingProfileOptions = true;
      const selected = selectedModels();
      const single = selected.length === 1 ? selected[0] : null;
      const provider = models
        ? (single?.dataset.provider || "")
        : (form.dataset.defaultProvider || "");
      const isCodex = provider === "codex";
      const local = localQuery();
      processing?.querySelectorAll('option[data-codex-only]').forEach((option) => {
        option.disabled = local || !isCodex;
      });
      if (processing && processing.value !== "chunked") {
        const chosen = processing.selectedOptions[0];
        if (local || chosen?.disabled) processing.value = "chunked";
      }

      const lockedByDefaults = defaultsLocked();
      setLocked(models, lockedByDefaults || local);
      setLocked(processing, lockedByDefaults || local);
      setLocked(context, lockedByDefaults || local || processing?.value !== "chunked");
      syncingProfileOptions = false;
    };

    const applyQueryDefaults = () => {
      const selectedQueriesNow = selectedQueries();
      // A multi-query run may contain queries with different saved defaults.
      // Keep those settings per query on the server; only mirror defaults into
      // the override controls when exactly one query is selected.
      if (defaults?.multiple && selectedQueriesNow.length !== 1) {
        syncProfileOptions();
        return;
      }
      const option = selectedQueriesNow[0];
      if (!option) {
        syncProfileOptions();
        return;
      }
      if (models) {
        const wanted = option.dataset.model || "";
        let matched = false;
        Array.from(models.options).forEach((candidate) => {
          candidate.selected = Boolean(wanted && candidate.value === wanted);
          matched ||= candidate.selected;
        });
        if (!matched && wanted === "" && !localQuery() && models.options.length) {
          models.options[0].selected = true;
        }
        const selected = Array.from(models.selectedOptions);
        if (context && selected.length === 1 && selected[0].dataset.contextWindow) {
          context.value = selected[0].dataset.contextWindow;
        }
      }
      if (processing && option.dataset.processing) processing.value = option.dataset.processing;
      syncProfileOptions();
      models?.dispatchEvent(new Event("change", { bubbles: true }));
    };

    processing?.addEventListener("change", syncProfileOptions);
    models?.addEventListener("change", syncProfileOptions);
    defaults?.addEventListener("change", applyQueryDefaults);
    defaultsToggle?.addEventListener("change", () => {
      if (defaultsLocked()) applyQueryDefaults();
      else syncProfileOptions();
    });
    queryKind?.addEventListener("change", syncProfileOptions);
    if (defaults) applyQueryDefaults();
    else syncProfileOptions();
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
        const row = checkbox.closest("[data-query-filter-row]");
        if (row) {
          const key = checkbox.name === "auto_run_papers" ? "filterNewPapers" : "filterNewBlogs";
          row.dataset[key] = checkbox.checked ? "yes" : "no";
          row.closest("[data-query-filter-table]")?.dispatchEvent(new Event("query-filter-refresh"));
        }
        setStatus("Unsaved changes");
        if (saveTimer) window.clearTimeout(saveTimer);
        saveTimer = window.setTimeout(saveSelections, 180);
      }));
  }

  document.querySelectorAll("[data-query-filter-table]").forEach((table) => {
    const rows = Array.from(table.querySelectorAll("tbody [data-query-filter-row]"));
    const filters = Array.from(table.querySelectorAll("[data-table-filter]"));
    const empty = table.querySelector("[data-table-filter-empty]");
    const panel = table.closest(".query-table-panel");
    const count = panel?.querySelector("[data-table-filter-count]");
    const normalize = (value) => String(value || "").trim().toLocaleLowerCase();
    const datasetKey = (name) => `filter${name.split("-").map((part) => part.charAt(0).toUpperCase() + part.slice(1)).join("")}`;
    const applyFilters = () => {
      let visible = 0;
      rows.forEach((row) => {
        const matches = filters.every((filter) => {
          const needle = normalize(filter.value);
          if (!needle) return true;
          const key = datasetKey(filter.dataset.tableFilter || "");
          return normalize(row.dataset[key]).includes(needle);
        });
        row.hidden = !matches;
        if (matches) visible += 1;
      });
      if (empty) empty.hidden = visible !== 0 || rows.length === 0;
      if (count) count.textContent = String(visible);
    };
    filters.forEach((filter) => {
      filter.addEventListener("input", applyFilters);
      filter.addEventListener("change", applyFilters);
    });
    table.addEventListener("query-filter-refresh", applyFilters);
    applyFilters();
  });

  const contentResultsRunnerForm = document.querySelector("form.content-results-runner-form");
  const syncContentRunTarget = () => {
    if (!contentResultsRunnerForm) return;
    const targetToggle = contentResultsRunnerForm.querySelector("[data-run-target-toggle]");
    const targetValue = contentResultsRunnerForm.querySelector("[data-run-target-value]");
    const targetLabel = contentResultsRunnerForm.querySelector("[data-run-target-label]");
    const runButton = contentResultsRunnerForm.querySelector(".collection-run-button");
    const summary = document.querySelector(".content-results-runner-details #collection-run-label");
    if (!targetToggle || !targetValue || !targetLabel || !runButton) return;
    const resultCount = Number(targetToggle.dataset.resultCount || 0);
    const selectedCount = Number(targetToggle.dataset.selectedCount || 0);
    if (selectedCount === 0) targetValue.value = "results";
    targetToggle.disabled = selectedCount === 0;
    const useSelected = targetValue.value === "selected";
    targetToggle.setAttribute("aria-pressed", String(useSelected));
    const activeCount = useSelected ? selectedCount : resultCount;
    targetLabel.textContent = useSelected
      ? `Selected (${selectedCount})`
      : `Search results (${resultCount})`;
    const running = runButton.dataset.runRunning === "true";
    runButton.disabled = running || activeCount === 0;
    if (summary) {
      summary.textContent = useSelected
        ? `Run saved prompt on ${selectedCount} selected record${selectedCount === 1 ? "" : "s"}`
        : `Run saved prompt on all ${resultCount} search result${resultCount === 1 ? "" : "s"}`;
    }
  };
  contentResultsRunnerForm?.querySelector("[data-run-target-toggle]")?.addEventListener(
    "click", () => {
      const targetValue = contentResultsRunnerForm.querySelector("[data-run-target-value]");
      if (!targetValue) return;
      targetValue.value = targetValue.value === "selected" ? "results" : "selected";
      syncContentRunTarget();
    }
  );
  syncContentRunTarget();

  const syncContentResultsRunnerFromDocument = (doc) => {
    const currentDetails = document.querySelector(".content-results-runner-details");
    const incomingDetails = doc.querySelector(".content-results-runner-details");
    if (currentDetails && incomingDetails) currentDetails.hidden = incomingDetails.hidden;
    if (!contentResultsRunnerForm) return;
    const incomingForm = doc.querySelector("form.content-results-runner-form");
    if (!incomingForm) return;
    contentResultsRunnerForm.querySelectorAll("[data-run-filter]").forEach((input) => input.remove());
    incomingForm.querySelectorAll("[data-run-filter]").forEach((input) => {
      contentResultsRunnerForm.appendChild(input.cloneNode(true));
    });
    const targetToggle = contentResultsRunnerForm.querySelector("[data-run-target-toggle]");
    const incomingToggle = incomingForm.querySelector("[data-run-target-toggle]");
    if (targetToggle && incomingToggle) {
      targetToggle.dataset.resultCount = incomingToggle.dataset.resultCount || "0";
      targetToggle.dataset.selectedCount = incomingToggle.dataset.selectedCount || "0";
    }
    const runButton = contentResultsRunnerForm.querySelector(".collection-run-button");
    const incomingButton = incomingForm.querySelector(".collection-run-button");
    if (runButton && incomingButton) {
      runButton.dataset.runRunning = incomingButton.dataset.runRunning || "false";
    }
    syncContentRunTarget();
  };

  const paperTableBody = document.getElementById("paper-table-body");
  if (paperTableBody && !document.querySelector(".selected-content-dock")) {
    const hiddenInput = document.getElementById("selected-paper-ids");
    const status = document.getElementById("paper-selection-status");
    const runLabel = document.getElementById("collection-run-label");
    const clear = document.getElementById("clear-paper-selection");
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

  const paperJobs = document.getElementById("paper-jobs");
  const discoverPaperImports = Array.from(document.querySelectorAll("form[data-discover-paper-import]"));
  if (document.body.dataset.runRunning === "true" && !paperJobs && discoverPaperImports.length === 0
      && !document.querySelector("#content-page-root") && !document.querySelector(".paper-detail-root")) {
    window.setTimeout(() => window.location.reload(), 5000);
  }

  if (discoverPaperImports.length) {
    const markImported = (form, url) => {
      const button = form.querySelector("button[type=submit]");
      if (!button) return;
      form.dataset.importedUrl = url;
      button.disabled = false;
      button.textContent = "Open imported source";
      button.classList.remove("button-primary");
      const error = form.querySelector("[data-import-error]");
      if (error) error.textContent = "";
    };

    const markIdle = (form, message = "") => {
      const button = form.querySelector("button[type=submit]");
      if (!button) return;
      button.disabled = false;
      button.textContent = "Import";
      const error = form.querySelector("[data-import-error]");
      if (error) error.textContent = message;
    };

    const pollImport = async (form) => {
      const workId = form.dataset.workId || "";
      try {
        const response = await fetch(`/api/discover/paper-import-status?work_id=${encodeURIComponent(workId)}`, { cache: "no-store" });
        const payload = await response.json();
        if (payload.state === "imported" && payload.url) {
          markImported(form, payload.url);
          return;
        }
        if (payload.state === "failed") {
          markIdle(form, payload.error || "Import failed");
          return;
        }
        if (payload.state === "idle") {
          markIdle(form);
          return;
        }
      } catch (_error) {
        markIdle(form, "Could not check import status");
        return;
      }
      window.setTimeout(() => pollImport(form), 700);
    };

    discoverPaperImports.forEach((form) => {
      const button = form.querySelector("button[type=submit]");
      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        if (form.dataset.importedUrl) {
          window.location.assign(form.dataset.importedUrl);
          return;
        }
        if (!button || button.disabled) return;
        button.disabled = true;
        button.textContent = "Importing…";
        const error = form.querySelector("[data-import-error]");
        if (error) error.textContent = "";
        try {
          const response = await fetch(form.action, {
            method: "POST",
            body: new FormData(form),
            headers: { Accept: "application/json" },
          });
          const payload = await response.json();
          if (payload.state === "imported" && payload.url) {
            markImported(form, payload.url);
            return;
          }
          if (!response.ok) {
            markIdle(form, payload.error || "Could not start import");
            return;
          }
          pollImport(form);
        } catch (_error) {
          markIdle(form, "Could not start import");
        }
      });
      if (button?.disabled && button.textContent.includes("Importing")) pollImport(form);
    });
  }

  const blogDiscoverForm = document.querySelector("[data-blog-discover-form]");
  if (blogDiscoverForm) {
    const input = blogDiscoverForm.querySelector("[data-blog-url-input]");
    const importWhole = blogDiscoverForm.querySelector("[data-blog-import-whole]");
    const syncBlogImport = () => {
      if (importWhole) importWhole.disabled = !input?.value.trim();
    };
    input?.addEventListener("input", syncBlogImport);
    importWhole?.addEventListener("click", () => {
      const source = input?.value.trim() || "";
      if (!source) return;
      window.location.assign(`/discover/blog/import?source=${encodeURIComponent(source)}`);
    });
    syncBlogImport();
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
      const updateFetchHistory = (snapshot) => {
        const runId = Number(snapshot.run_id || 0);
        if (!runId) return true;
        const row = document.querySelector(
          `[data-fetch-history-run-id="${runId}"]`,
        );
        if (!row) {
          window.location.reload();
          return false;
        }
        const stats = snapshot.stats || {};
        row.querySelectorAll("[data-fetch-stat]").forEach((element) => {
          const key = element.dataset.fetchStat;
          if (key && Object.prototype.hasOwnProperty.call(stats, key)) {
            element.textContent = String(stats[key] || 0);
          }
        });
        const status = row.querySelector("[data-fetch-status]");
        if (status) {
          const value = snapshot.cancel_requested ? "stopping" : "running";
          status.textContent = value;
          status.className = `status status-${value}`;
        }
        const finished = row.querySelector("[data-fetch-finished]");
        if (finished) finished.textContent = "Running";
        return true;
      };
      const pollRunning = async () => {
        try {
          const response = await fetch("/api/running-status", { cache: "no-store" });
          if (!response.ok) throw new Error("status unavailable");
          const payload = await response.json();
          if (!updateFetchHistory(payload.fetch || {})) return;
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

  document.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-copy-source]");
    if (!button) return;
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


  const enhanceSelect = (select, index) => {
    if (select.dataset.searchSelectReady === "true") return;
    select.dataset.searchSelectReady = "true";
    const multiple = select.multiple;
    const enabledOptions = Array.from(select.options).filter((option) => !option.disabled);
    const currentEnabledOptions = () => Array.from(select.options).filter((option) => !option.disabled);
    const cycle = !select.hasAttribute("data-search-always") && !multiple && enabledOptions.length > 0 && enabledOptions.length <= 5;
    const wrapCycle = cycle && select.hasAttribute("data-wrap-cycle");
    const form = select.form || select.closest("form");
    const autoApply = () => {
      if (form && (form.method || "get").toLocaleLowerCase() === "get") form.requestSubmit();
    };
    const wrapper = document.createElement("div");
    wrapper.className = `search-select${multiple ? " search-select-multiple" : ""}${cycle ? " search-select-cycle" : ""}${wrapCycle ? " search-select-wrap-cycle" : ""}`;
    const floatingOptions = Boolean(
      select.hasAttribute("data-float-options") ||
      select.hasAttribute("data-float-options-up") ||
      (select.matches("[data-model-picker]") && select.closest(".content-results-runner-row"))
    );
    const floatOptionsUp = Boolean(
      select.hasAttribute("data-float-options-up") ||
      (select.matches("[data-model-picker]") && select.closest(".content-results-runner-row"))
    );
    const input = document.createElement(wrapCycle ? "button" : "input");
    if (wrapCycle) input.type = "button";
    else input.type = "search";
    input.className = `search-select-input${wrapCycle ? " search-select-wrap-button" : ""}`;
    if (!wrapCycle) input.autocomplete = "off";
    input.setAttribute("role", "combobox");
    input.setAttribute("aria-autocomplete", cycle ? "none" : "list");
    input.setAttribute("aria-expanded", "false");
    if (!wrapCycle) {
      input.placeholder = select.dataset.searchPlaceholder || (multiple ? "Type to filter choices" : "Type to filter");
      if (cycle) input.readOnly = true;
    }
    const list = document.createElement("div");
    list.className = "search-select-options";
    if (floatingOptions) list.classList.add("search-select-options-floating");
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
    if (floatingOptions) document.body.appendChild(list);
    select.classList.add("search-select-native");
    select.setAttribute("aria-hidden", "true");
    select.tabIndex = -1;

    let active = -1;
    const normalized = (value) => String(value || "").trim().toLocaleLowerCase();
    const displayValue = () => wrapCycle ? input.textContent || "" : input.value;
    const setDisplayValue = (value) => {
      if (wrapCycle) input.textContent = value;
      else input.value = value;
    };
    const maxRenderedOptions = 80;
    let optionRows = [];
    let renderOptionRows = () => {};
    const positionFloatingOptions = () => {
      if (!floatingOptions || list.hidden) return;
      const rect = input.getBoundingClientRect();
      const viewportPadding = 8;
      const gap = 5;
      const width = Math.min(rect.width, window.innerWidth - (viewportPadding * 2));
      const left = Math.max(
        viewportPadding,
        Math.min(rect.left, window.innerWidth - width - viewportPadding)
      );
      list.style.left = `${left}px`;
      list.style.right = "auto";
      list.style.width = `${width}px`;
      const spaceAbove = Math.max(0, rect.top - gap - viewportPadding);
      const spaceBelow = Math.max(0, window.innerHeight - rect.bottom - gap - viewportPadding);
      const openUp = floatOptionsUp || (spaceBelow < 180 && spaceAbove > spaceBelow);
      if (openUp) {
        list.style.top = "auto";
        list.style.bottom = `${window.innerHeight - rect.top + gap}px`;
        list.style.maxHeight = `${Math.max(80, Math.min(280, spaceAbove))}px`;
      } else {
        list.style.top = `${rect.bottom + gap}px`;
        list.style.bottom = "auto";
        list.style.maxHeight = `${Math.max(80, Math.min(280, spaceBelow))}px`;
      }
    };
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
        setDisplayValue(select.selectedIndex >= 0 ? select.options[select.selectedIndex].textContent || "" : "");
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
        positionFloatingOptions();
      }
      setActive(0);
    };
    const close = () => {
      list.hidden = true;
      input.setAttribute("aria-expanded", "false");
      active = -1;
      optionRows.forEach((row) => row.classList.remove("active"));
      if (!multiple) syncInput();
      else setDisplayValue("");
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
        setDisplayValue("");
      } else {
        select.selectedIndex = optionIndex;
        if (wrapCycle) setDisplayValue(option.textContent || option.value);
        else input.value = option.textContent || option.value;
      }
      select.dispatchEvent(new Event("change", { bubbles: true }));
      syncInput(true);
      if (multiple) {
        setDisplayValue("");
        refresh(true);
        input.focus();
      } else {
        close();
        input.focus();
      }
      autoApply();
    };
    renderOptionRows = () => {
      const term = normalized(displayValue());
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
        const options = currentEnabledOptions();
        const current = options.indexOf(select.options[select.selectedIndex]);
        const next = options[(current + 1 + options.length) % options.length];
        if (!next) return;
        select.value = next.value;
        select.dispatchEvent(new Event("change", { bubbles: true }));
        syncInput(true);
        autoApply();
        return;
      }
      if (!multiple && select.selectedIndex >= 0) {
        const selectedText = select.options[select.selectedIndex].textContent || "";
        if (displayValue() === selectedText) {
          setDisplayValue("");
          refresh(true);
        }
      }
    });
    input.addEventListener("input", () => refresh(true));
    input.addEventListener("keydown", (event) => {
      const visible = visibleRows();
      if (cycle && ["ArrowDown", "ArrowUp", "Enter", " "].includes(event.key)) {
        event.preventDefault();
        const options = currentEnabledOptions();
        const current = options.indexOf(select.options[select.selectedIndex]);
        const direction = event.key === "ArrowUp" ? -1 : 1;
        const next = options[(current + direction + options.length) % options.length];
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
          if (option && !multiple && !option.selected) setDisplayValue(option.textContent || option.value);
          choose(row);
        }
      } else if (event.key === "Escape") {
        event.preventDefault();
        close();
      }
    });
    document.addEventListener("click", (event) => {
      if (!wrapper.contains(event.target) && !list.contains(event.target)) close();
    });
    if (floatingOptions) {
      window.addEventListener("resize", positionFloatingOptions);
      window.addEventListener("scroll", positionFloatingOptions, { passive: true });
      select.closest(".content-results-runner-row")?.addEventListener(
        "scroll", positionFloatingOptions, { passive: true }
      );
    }
    const observer = new MutationObserver(syncInput);
    observer.observe(select, { attributes: true, attributeFilter: ["disabled"] });
    select.addEventListener("change", () => syncInput(true));
    syncInput();
  };

  document.querySelectorAll("select").forEach((select, index) => enhanceSelect(select, index));

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
  if (queryKindInput) {
    const updateQueryKindSettings = () => {
      const kind = queryKindInput.value;
      document.querySelectorAll("[data-classification-only]").forEach((item) => {
        item.hidden = kind !== "classification";
        item.querySelectorAll("input, textarea, select, button").forEach((control) => {
          control.disabled = kind !== "classification";
        });
      });
      document.querySelectorAll("[data-regexp-only]").forEach((item) => {
        item.hidden = kind !== "regexp";
        item.querySelectorAll("input, textarea, select, button").forEach((control) => {
          control.disabled = kind !== "regexp";
        });
      });
      document.querySelectorAll("[data-llm-query-setting]").forEach((item) => {
        const hidden = kind === "regexp";
        item.hidden = hidden;
        item.querySelectorAll("input, textarea, select, button").forEach((control) => {
          control.disabled = hidden;
        });
      });
      document.querySelector("select[data-paper-processing]")?.dispatchEvent(
        new Event("change", { bubbles: true })
      );
    };
    queryKindInput.addEventListener("change", updateQueryKindSettings);
    updateQueryKindSettings();
  }

  document.addEventListener("click", (event) => {
    const opener = event.target.closest("[data-dialog-open]");
    if (opener) {
      const dialog = document.getElementById(opener.dataset.dialogOpen || "");
      if (dialog?.showModal) dialog.showModal();
      return;
    }
    const closer = event.target.closest("dialog [data-dialog-close]");
    if (closer) {
      closer.closest("dialog")?.close();
      document.body.classList.remove("dialog-open");
      return;
    }
    if (event.target.matches("dialog")) {
      event.target.close();
      document.body.classList.remove("dialog-open");
    }
  });

  document.addEventListener("submit", async (event) => {
    const selectedExportForm = event.target.closest?.("[data-selected-export-form]");
    if (!selectedExportForm) return;
    const folderInput = selectedExportForm.querySelector("[data-export-folder]");
    const folderStatus = selectedExportForm.querySelector("[data-export-folder-status]");
    if (!folderInput || folderInput.value.trim() || selectedExportForm.dataset.choosingFolder === "true") return;
    event.preventDefault();
    selectedExportForm.dataset.choosingFolder = "true";
    const setFolderStatus = (message, failed = false) => {
      if (!folderStatus) return;
      folderStatus.textContent = message;
      folderStatus.classList.toggle("task-error", failed);
      folderStatus.classList.toggle("muted", !failed);
    };
    const submitter = event.submitter || selectedExportForm.querySelector('button[type="submit"]');
    if (submitter) submitter.disabled = true;
    setFolderStatus("Opening system folder picker…");
    try {
      const response = await fetch("/content/selected/export/folder-picker", {
        method: "POST",
        headers: { "Accept": "application/json" },
      });
      const payload = await response.json();
      if (!response.ok || (!payload.ok && !payload.cancelled)) {
        throw new Error(payload.message || "Could not open the system folder picker.");
      }
      if (payload.cancelled) {
        setFolderStatus("Folder selection cancelled.");
        return;
      }
      folderInput.value = payload.folder || "";
      if (!folderInput.value.trim()) throw new Error("No export folder was selected.");
      setFolderStatus(`Destination: ${folderInput.value}`);
      selectedExportForm.requestSubmit(submitter || undefined);
    } catch (error) {
      setFolderStatus(error.message || "Could not choose an export folder.", true);
    } finally {
      selectedExportForm.dataset.choosingFolder = "false";
      if (submitter) submitter.disabled = false;
    }
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
    const localPapers = document.getElementById("author-local-papers");
    const localBlogs = document.getElementById("author-local-blogs");
    const authorPage = document.getElementById("author-page");
    const scholar = document.getElementById("author-scholar");
    const arxiv = document.getElementById("author-arxiv");
    const openalex = document.getElementById("author-openalex");
    const track = document.getElementById("author-track");
    const trackForm = document.getElementById("author-track-form");
    const status = document.getElementById("author-status");
    const update = () => {
      const option = picker.options[picker.selectedIndex];
      localPapers.href = option.dataset.papers || "#";
      localBlogs.href = option.dataset.blogs || "#";
      authorPage.href = `/authors?q=${encodeURIComponent(option.dataset.authorName || "")}`;
      scholar.href = option.dataset.scholar || "#";
      arxiv.href = option.dataset.arxiv || "#";
      const hasOpenAlex = Boolean(option.dataset.openalex);
      openalex.href = option.dataset.openalex || "#";
      openalex.hidden = !hasOpenAlex;
      const trackUrl = option.dataset.track || "";
      trackForm.action = trackUrl;
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
      stub: "Stub filter",
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
  const contentReloadKey = "paperwatch-content-reload";
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
  let contentRefreshSequence = 0;
  let contentRefreshController = null;
  const refreshContentPage = async (url = window.location.href, push = false) => {
    const refreshSequence = ++contentRefreshSequence;
    contentRefreshController?.abort();
    const refreshController = new AbortController();
    contentRefreshController = refreshController;
    let response;
    let responseText;
    try {
      response = await fetch(url, {
        cache: "no-store",
        headers: { "X-Paperwatch-Partial": "1" },
        signal: refreshController.signal,
      });
      if (!response.ok) throw new Error(`Request failed with HTTP ${response.status}`);
      responseText = await response.text();
    } catch (error) {
      if (error?.name === "AbortError") return false;
      throw error;
    } finally {
      if (contentRefreshController === refreshController) contentRefreshController = null;
    }
    const doc = parseHtml(responseText);
    if (refreshSequence !== contentRefreshSequence) return false;
    syncContentResultsRunnerFromDocument(doc);
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
    const finalUrl = new URL(response.url || url, window.location.href).toString();
    if (push) window.history.pushState({}, "", finalUrl);
    else if (response.redirected) window.history.replaceState({}, "", finalUrl);
    rememberContentUrl(finalUrl);
    return true;
  };
  contentHome?.addEventListener("click", (event) => {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    if (document.querySelector("#content-page-root")) {
      event.preventDefault();
      refreshContentPage(contentHome.href, false).catch(console.error);
      return;
    }
    storageSet(contentReloadKey, "1");
  });
  if (document.querySelector("#content-page-root") && storageGet(contentReloadKey) === "1") {
    storageRemove(contentReloadKey);
    window.setTimeout(() => refreshContentPage(window.location.href, false).catch(console.error), 0);
  }
  window.addEventListener("pageshow", (event) => {
    if (event.persisted && document.querySelector("#content-page-root")) {
      refreshContentPage(window.location.href, false).catch(console.error);
    }
  });
  const runningKindsFromBody = () => ({
    fetch: document.body.dataset.runRunning === "true",
    analysis: document.body.dataset.analysisRunning === "true",
    query: document.body.dataset.queryRunning === "true",
  });
  const fetchRunningStatus = async () => {
    const response = await fetch("/api/running-status", { cache: "no-store" });
    if (!response.ok) throw new Error(`Request failed with HTTP ${response.status}`);
    return response.json();
  };
  const initialRunningKinds = runningKindsFromBody();
  if (document.querySelector("#content-page-root")
      && Object.values(initialRunningKinds).some(Boolean)) {
    const pollContentCompletion = async () => {
      try {
        const status = await fetchRunningStatus();
        const stillRunning = Object.entries(initialRunningKinds).some(
          ([kind, watched]) => watched && Boolean(status[kind]?.running)
        );
        if (stillRunning) {
          await refreshContentPage(window.location.href, false);
          window.setTimeout(pollContentCompletion, 1500);
          return;
        }
        await refreshContentPage(window.location.href, false);
      } catch (_) {
        window.setTimeout(pollContentCompletion, 1500);
      }
    };
    window.setTimeout(pollContentCompletion, 250);
  }

  const queryRunDetail = document.querySelector("[data-query-run-detail]");
  if (queryRunDetail && ["queued", "running"].includes(queryRunDetail.dataset.queryRunStatus || "")) {
    window.setTimeout(() => window.location.reload(), 1500);
  }

  const paperDetailRoot = document.querySelector(".paper-detail-root[data-paper-id]");
  if (paperDetailRoot && (initialRunningKinds.analysis || initialRunningKinds.query)) {
    const paperId = Number(paperDetailRoot.dataset.paperId || 0);
    let sawRelevantJob = false;
    const refreshPaperResults = async () => {
      const response = await fetch(window.location.pathname, {
        cache: "no-store",
        headers: { "X-Paperwatch-Fragment": "1" },
      });
      if (!response.ok) return false;
      const doc = parseHtml(await response.text());
      const incoming = doc.querySelector("#paper-results");
      const current = document.querySelector("#paper-results");
      if (!incoming || !current) return false;
      current.replaceWith(incoming);
      return true;
    };
    const pollPaperCompletion = async () => {
      try {
        const status = await fetchRunningStatus();
        const analysisRelevant = Boolean(status.analysis?.batch_id)
          && Number(status.analysis?.paper_id || 0) === paperId;
        const queryRelevant = Boolean(status.query?.batch_id)
          && Number(status.query?.paper_id || 0) === paperId;
        sawRelevantJob ||= analysisRelevant || queryRelevant;
        const stillRunning = (analysisRelevant && Boolean(status.analysis?.running))
          || (queryRelevant && Boolean(status.query?.running));
        if (stillRunning) {
          await refreshPaperResults();
          window.setTimeout(pollPaperCompletion, 1500);
          return;
        }
        if (sawRelevantJob) await refreshPaperResults();
      } catch (_) {
        window.setTimeout(pollPaperCompletion, 1500);
      }
    };
    window.setTimeout(pollPaperCompletion, 250);
  }

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
    if (!form || event.defaultPrevented) return;
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
        if (document.querySelector("#content-page-root")) {
          await refreshContentPage(window.location.href, false);
          return;
        }
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
      const refreshContentResults = Boolean(
        document.querySelector("#content-page-root") && form.closest("#content-results-root")
      );
      try {
        await postAsync(form);
        if (refreshContentResults) {
          await refreshContentPage(window.location.href, false);
          return;
        }
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
