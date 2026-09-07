const data = window.PROJECT_DATA;

function renderSnapshot() {
  if (!data?.snapshot) return;
  const { status, checkpoint, lastCuratedStep, targetStep, description } = data.snapshot;
  document.querySelector("#current-status").textContent = status;
  document.querySelector("#current-run-description").textContent = description;
  document.querySelector("#checkpoint-label").textContent = `checkpoint ${checkpoint}`;
  document.querySelector("#target-label").textContent = `target ${targetStep}`;
  document.querySelector("#run-progress").style.width = `${Math.min(100, (lastCuratedStep / targetStep) * 100)}%`;
}

function renderExperimentTable() {
  const table = document.querySelector("#experiment-table");
  if (!table || !data) return;

  const statusLabels = {
    baseline: "基线",
    complete: "完成",
    active: "进行中",
  };

  table.innerHTML = `
    <div class="experiment-row header">
      <span>阶段</span><span>模型</span><span>评测设计</span><span>结果</span><span>状态</span>
    </div>
    ${data.experiments
      .map(
        (run) => `
          <div class="experiment-row">
            <strong>${run.stage}</strong>
            <span>${run.model}<br><small>${run.note}</small></span>
            <small>${run.protocol}</small>
            <span class="experiment-result">${run.result}</span>
            <span class="run-status ${run.status}">${statusLabels[run.status]}</span>
          </div>`,
      )
      .join("")}
  `;
}

function renderPassK() {
  const chart = document.querySelector("#passk-chart");
  if (!chart || !data) return;

  chart.innerHTML = data.passk
    .map(
      (item) => `
        <div class="bar-row">
          <span>${item.label}</span>
          <div class="bar-track"><i style="width:${item.value}%"></i></div>
          <strong>${item.value.toFixed(item.value % 1 === 0 ? 1 : 3)}%</strong>
        </div>`,
    )
    .join("");
}

function renderFlow(activeIndex = 0) {
  const selector = document.querySelector("#flow-selector");
  const detail = document.querySelector("#flow-detail");
  if (!selector || !detail || !data) return;

  selector.innerHTML = data.stepFlow
    .map(
      (step, index) => `
        <button class="flow-button ${index === activeIndex ? "active" : ""}" type="button" data-flow-index="${index}">
          <span>${String(index + 1).padStart(2, "0")}</span>
          <strong>${step.title}</strong>
        </button>`,
    )
    .join("");

  const step = data.stepFlow[activeIndex];
  detail.innerHTML = `
    <span class="flow-count">STEP ${String(activeIndex + 1).padStart(2, "0")} / ${data.stepFlow.length}</span>
    <h3>${step.title}</h3>
    <span class="flow-owner">负责组件 · ${step.owner}</span>
    <span class="flow-shape">${step.shape}</span>
    <p>${step.detail}</p>
    <div class="flow-why"><span>为什么必要</span><strong>${step.why}</strong></div>
  `;

  selector.querySelectorAll("[data-flow-index]").forEach((button) => {
    button.addEventListener("click", () => renderFlow(Number(button.dataset.flowIndex)));
  });
}

function setupNavigation() {
  const sections = [...document.querySelectorAll("main section[id]")];
  const navLinks = [...document.querySelectorAll(".side-nav a")];
  const observer = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!visible) return;
      navLinks.forEach((link) => {
        link.classList.toggle("active", link.hash === `#${visible.target.id}`);
      });
    },
    { rootMargin: "-20% 0px -65%", threshold: [0.05, 0.25, 0.6] },
  );
  sections.forEach((section) => observer.observe(section));
}

function setupInterviewMode() {
  const button = document.querySelector("#interview-toggle");
  button?.addEventListener("click", () => {
    document.body.classList.add("interview-mode");
    document.querySelector("#interview")?.scrollIntoView({ behavior: "smooth" });
    button.textContent = "面试模式已开启";
  });
}

function setupThemeToggle() {
  const root = document.documentElement;
  const button = document.querySelector("#theme-toggle");
  const icon = document.querySelector("#theme-icon");
  const label = document.querySelector("#theme-label");
  if (!button || !icon || !label) return;

  const updateControl = () => {
    const isLight = root.dataset.theme === "light";
    icon.textContent = isLight ? "☀" : "☾";
    label.textContent = isLight ? "浅色模式" : "深色模式";
    button.setAttribute("aria-pressed", String(isLight));
    button.setAttribute("title", isLight ? "切换为深色模式" : "切换为浅色模式");
  };

  button.addEventListener("click", () => {
    root.dataset.theme = root.dataset.theme === "light" ? "dark" : "light";
    localStorage.setItem("agentic-rl-theme", root.dataset.theme);
    updateControl();
  });

  updateControl();
}

function setupCopyButtons() {
  document.querySelectorAll("[data-copy-target]").forEach((button) => {
    button.addEventListener("click", async () => {
      const target = document.getElementById(button.dataset.copyTarget);
      if (!target) return;
      const text = target.textContent.trim();
      try {
        await navigator.clipboard.writeText(text);
      } catch {
        const area = document.createElement("textarea");
        area.value = text;
        document.body.appendChild(area);
        area.select();
        document.execCommand("copy");
        area.remove();
      }
      const original = button.textContent;
      button.textContent = "已复制";
      window.setTimeout(() => { button.textContent = original; }, 1600);
    });
  });
}

renderSnapshot();
renderExperimentTable();
renderPassK();
renderFlow();
setupNavigation();
setupInterviewMode();
setupThemeToggle();
setupCopyButtons();
