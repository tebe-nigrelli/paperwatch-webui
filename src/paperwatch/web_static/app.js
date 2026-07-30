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

  const running = ["runRunning", "analysisRunning", "queryRunning"]
    .some((key) => document.body.dataset[key] === "true");
  if (running) window.setTimeout(() => window.location.reload(), 5000);

  document.querySelectorAll("form[data-confirm]").forEach((form) => {
    form.addEventListener("submit", (event) => {
      if (!window.confirm(form.dataset.confirm)) event.preventDefault();
    });
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
