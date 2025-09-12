const els = {
  select: document.getElementById('playerSelect'),
  refreshBtn: document.getElementById('refreshBtn'),
  status: document.getElementById('status'),
  predictionValue: document.getElementById('predictionValue'),
  asOf: document.getElementById('asOf'),
  featureList: document.getElementById('featureList'),
  downloadCsvBtn: document.getElementById('downloadCsvBtn')
};

let players = [];
let historyData = [];

async function loadPlayers() {
  const res = await fetch('./data/players.json');
  const json = await res.json();
  players = json.players || [];
  els.select.innerHTML = '';
  players.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p.id;
    opt.textContent = `${p.name} (${p.model})`;
    els.select.appendChild(opt);
  });
}

function setStatus(msg) { els.status.textContent = msg || ''; }

async function loadNextGame(pid) {
  const res = await fetch(`./data/${pid}/next_game.json?_=${Date.now()}`);
  const json = await res.json();
  els.predictionValue.textContent = (json.predicted_points ?? '—');
  els.asOf.textContent = `As of: ${json.as_of || '—'}`;

  els.featureList.innerHTML = '';
  const expl = json.explanation || null;
  if (expl) {
    Object.entries(expl).forEach(([k,v]) => {
      const li = document.createElement('li');
      li.innerHTML = `<span>${k}</span><span>${Number(v).toFixed(3)}</span>`;
      els.featureList.appendChild(li);
    });
  } else {
    const li = document.createElement('li');
    li.textContent = 'No feature explanation available.';
    els.featureList.appendChild(li);
  }
}

async function loadHistory(pid) {
  const res = await fetch(`./data/${pid}/history.json?_=${Date.now()}`);
  const json = await res.json();
  historyData = (json.series || []);
  renderChart(historyData);
}

let chart;
function renderChart(series) {
  const ctx = document.getElementById('historyChart').getContext('2d');
  const labels = series.map(s => s.game_date);
  const actuals = series.map(s => s.actual);
  const preds = series.map(s => s.predicted);

  if (chart) chart.destroy();
  chart = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets: [{label:'Actual', data: actuals}, {label:'Predicted', data: preds}] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { ticks: { color: '#9aa7b6' } },
        y: { ticks: { color: '#9aa7b6' } }
      },
      plugins: { legend: { labels: { color: '#e8edf2' } } }
    }
  });
}

function downloadCSV(series) {
  const hdr = 'game_date,actual,predicted\n';
  const rows = series.map(s => `${s.game_date},${s.actual},${s.predicted}`).join('\n');
  const blob = new Blob([hdr + rows], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `history_${els.select.value}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

async function refresh() {
  setStatus('Loading...');
  const pid = els.select.value;
  await Promise.all([loadNextGame(pid), loadHistory(pid)]);
  setStatus('');
}

async function init() {
  await loadPlayers();
  els.select.addEventListener('change', refresh);
  els.refreshBtn.addEventListener('click', refresh);
  els.downloadCsvBtn.addEventListener('click', () => downloadCSV(historyData));
  if (els.select.value) await refresh();
}
init();
