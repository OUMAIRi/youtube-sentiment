const API_URL = "https://supreme-telegram-g469prxwq6x52vqvq-7860.app.github.dev";
let latestAnalysis = null;

const status = document.getElementById("status");
const summary = document.getElementById("summary");
const results = document.getElementById("results");
const checkButton = document.getElementById("check-api");
const analyzeButton = document.getElementById("analyze");

async function callApi(path, options = {}) {
  const response = await fetch(`${API_URL}${path}`, {
    ...options,
    signal: AbortSignal.timeout(20000)
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(
      `HTTP ${response.status} : ${JSON.stringify(data.detail)}`
    );
  }

  return data;
}

async function runAction(action) {
  checkButton.disabled = true;
  analyzeButton.disabled = true;

  try {
    await action();
  } catch (error) {
    status.textContent = `Erreur : ${error.message}`;
  } finally {
    checkButton.disabled = false;
    analyzeButton.disabled = false;
  }
}

checkButton.addEventListener("click", () => runAction(async () => {
  status.textContent = "Connexion en cours…";
  const data = await callApi("/health");

  if (data.status !== "ok" || data.model_loaded !== true) {
    throw new Error("Le modèle n’est pas prêt.");
  }

  status.textContent = "API connectée — modèle prêt.";
}));

analyzeButton.addEventListener("click", () => runAction(async () => {
  results.replaceChildren();
latestAnalysis = null;
document.getElementById("chart").replaceChildren();
  summary.textContent = "";
  status.textContent = "Extraction des commentaires…";

  const [tab] = await chrome.tabs.query({
    active: true,
    currentWindow: true
  });

  if (!tab?.url) {
    throw new Error("Aucun onglet accessible.");
  }

  const url = new URL(tab.url);

  if (url.hostname !== "www.youtube.com" || url.pathname !== "/watch") {
    throw new Error("Ouvre une vidéo sur www.youtube.com.");
  }

  let extracted;

  try {
    extracted = await chrome.tabs.sendMessage(tab.id, {
      action: "extract-comments"
    });
  } catch {
    throw new Error("Recharge la page YouTube, puis réessaie.");
  }

  if (!extracted?.comments?.length) {
    throw new Error(
      "Aucun commentaire trouvé. Descends dans la page pour les charger."
    );
  }

  status.textContent =
    `Analyse de ${extracted.comments.length} commentaires…`;

  const data = await callApi("/predict_batch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ comments: extracted.comments })
  });
latestAnalysis = data;
renderChart(data.statistics);
  const stats = data.statistics;
  const percentages = stats.percentages;

  summary.textContent =
    `Total : ${stats.total}\n` +
    `Positifs : ${percentages.positif} %\n` +
    `Neutres : ${percentages.neutre} %\n` +
    `Négatifs : ${percentages.negatif} %`;

  for (const result of data.results) {
    const card = document.createElement("div");
    card.className = `comment ${result.sentiment}`;

    const heading = document.createElement("strong");
    heading.textContent =
      `${result.sentiment} — confiance : ` +
      `${(result.confidence * 100).toFixed(1)} %`;

    const text = document.createElement("p");
    text.textContent = result.text;

    card.append(heading, text);
    results.append(card);
  }

  status.textContent = "Analyse terminée.";
}));
const filter = document.getElementById("sentiment-filter");
const themeButton = document.getElementById("toggle-theme");
const copyButton = document.getElementById("copy-results");

document.body.classList.toggle(
  "dark",
  localStorage.getItem("theme") === "dark"
);

themeButton.addEventListener("click", () => {
  const isDark = document.body.classList.toggle("dark");
  localStorage.setItem("theme", isDark ? "dark" : "light");
});

filter.addEventListener("change", () => {
  for (const card of results.querySelectorAll(".comment")) {
    card.hidden =
      filter.value !== "all" &&
      !card.classList.contains(filter.value);
  }
});

analyzeButton.addEventListener("click", () => {
  filter.value = "all";
});

copyButton.addEventListener("click", async () => {
  const cards = Array.from(results.querySelectorAll(".comment"))
    .filter(card => !card.hidden);

  if (cards.length === 0) {
    status.textContent = "Aucun commentaire affiché à copier.";
    return;
  }

  const text = cards.map(card => {
    const heading = card.querySelector("strong").textContent;
    const comment = card.querySelector("p").textContent;
    return `${heading}\n${comment}`;
  }).join("\n\n");

  try {
    await navigator.clipboard.writeText(text);
    status.textContent = `${cards.length} commentaires copiés.`;
  } catch (error) {
    status.textContent = `Copie impossible : ${error.message}`;
  }
});
function renderChart(statistics) {
  const chart = document.getElementById("chart");
  chart.replaceChildren();

  const categories = [
    ["positif", "Positifs", "#16a34a"],
    ["neutre", "Neutres", "#64748b"],
    ["negatif", "Négatifs", "#dc2626"]
  ];

  for (const [key, name, color] of categories) {
    const percentage = statistics.percentages[key];

    const row = document.createElement("div");
    row.className = "chart-row";

    const label = document.createElement("label");
    label.htmlFor = `bar-${key}`;
    label.textContent =
      `${name} : ${statistics.counts[key]} (${percentage} %)`;

    const bar = document.createElement("progress");
    bar.id = `bar-${key}`;
    bar.max = 100;
    bar.value = percentage;
    bar.style.accentColor = color;
    bar.textContent = `${percentage} %`;

    row.append(label, bar);
    chart.append(row);
  }
}

document.getElementById("export-results")
  .addEventListener("click", () => {
    if (!latestAnalysis) {
      status.textContent = "Lance une analyse avant d’exporter.";
      return;
    }

    const exportData = {
      exported_at: new Date().toISOString(),
      ...latestAnalysis
    };

    const blob = new Blob(
      [JSON.stringify(exportData, null, 2)],
      { type: "application/json;charset=utf-8" }
    );

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "youtube-sentiment-results.json";

    document.body.append(link);
    link.click();
    link.remove();

    setTimeout(() => URL.revokeObjectURL(url), 1000);
    status.textContent = "Export JSON demandé.";
  });