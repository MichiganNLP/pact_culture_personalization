const DATA = window.PACT_DATA;
const maxOf = (arr, key) => Math.max(...arr.map(d => Math.abs(d[key])));


const modelViewNotes = {
  models: {
    title: 'Allow-preference rate',
    text: 'Percent of valid model decisions selecting Allow Preference rather than Follow Culture.'
  },
  regions: {
    title: 'Country-context effect',
    text: 'Shift in Allow Preference rate by scenario region, shown in percentage points relative to the overall model average.'
  },
  significance: {
    title: 'Significance tests',
    text: 'Concise regression-style contrasts for demographic and interaction effects in the model-behavior analysis.'
  }
};

function setModelViewNote(view) {
  const note = modelViewNotes[view];
  const el = document.querySelector('#model-chart-note');
  if (!note || !el) return;
  el.innerHTML = `<strong>${note.title}</strong><p>${note.text}</p>`;
}

const alignmentNotes = {
  majority: {
    title: 'Majority alignment',
    text: 'Agreement with the human-majority option, averaged across personal-choice and norm-judgment frames for each model.'
  },
  mae: {
    title: 'Rate-alignment MAE',
    text: 'Absolute difference between model and human culture-following rates. Lower values mean the model better matches the human response distribution.'
  },
  gap: {
    title: 'Signed culture-rate gap',
    text: 'Model minus human culture-following rate. Positive values indicate more culture-following than humans; negative values indicate more preference-allowing.'
  },
  uncertainty: {
    title: 'Uncertainty correlation',
    text: 'Correlation between human agreement and persona-conditioned model agreement on the same items. Higher values mean model variation better tracks human disagreement.'
  }
};

function barChart(el, rows, options = {}) {
  const max = options.max ?? maxOf(rows, 'value');
  el.innerHTML = rows.map(row => {
    const v = Number(row.value);
    const pct = Math.min(100, Math.abs(v) / max * 100);
    const cls = v < 0 ? 'negative' : '';
    const suffix = options.suffix ?? '%';
    return `<div class="bar-row">
      <div class="bar-label" title="${row.label}">${row.label}</div>
      <div class="bar-track"><div class="bar-fill ${cls}" style="width:${pct}%"></div></div>
      <div class="bar-value">${v}${suffix}</div>
    </div>`;
  }).join('');
}

function renderModelCharts() {
  setModelViewNote('models');
  barChart(document.querySelector('#model-chart'), DATA.modelBehavior.map(d => ({label: d.model, value: d.allow})), {max: 40});
  barChart(document.querySelector('#region-chart'), DATA.regionEffects.map(d => ({label: d.region, value: d.delta})), {max: 6, suffix: ' pp'});
  const sig = document.querySelector('#significance-table');
  sig.innerHTML = `<table><thead><tr><th>Factor</th><th>Contrast</th><th>Effect</th><th>p</th></tr></thead><tbody>${DATA.significance.map(r => `<tr><td>${r.factor}</td><td>${r.contrast}</td><td>${r.effect_size}</td><td>${r.p_value}</td></tr>`).join('')}</tbody></table>`;
}

function renderHumanChart() {
  const el = document.querySelector('#human-country-chart');
  el.innerHTML = DATA.humanCountry.map(d => {
    const personal = d.personalAllow;
    const norm = d.normAllow;
    return `<div class="dot-row">
      <div class="bar-label">${d.country}</div>
      <div class="dot-lane" title="Personal ${personal}%, Norm ${norm}%">
        <span class="dot personal" style="left:${personal}%"></span>
        <span class="dot norm" style="left:${norm}%"></span>
      </div>
    </div>`;
  }).join('') + '<p class="muted">Teal = personal choice allows preference; lavender = norm judgment allows preference.</p>';
}

function renderAlignment(metric = 'majority') {
  const el = document.querySelector('#alignment-chart');
  let rows;
  let suffix;
  let max;
  if (metric === 'uncertainty') {
    rows = DATA.uncertaintyAvg.map(d => ({label: d.model, value: d.corr})).sort((a,b) => b.value - a.value);
    suffix = '';
    max = .28;
  } else if (metric === 'mae') {
    rows = DATA.alignmentAvg.map(d => ({label: d.model, value: d.mae})).sort((a,b) => a.value - b.value);
    suffix = '';
    max = .55;
  } else if (metric === 'gap') {
    rows = DATA.alignmentAvg.map(d => ({label: d.model, value: d.cultureGap})).sort((a,b) => b.value - a.value);
    suffix = ' pp';
    max = 28;
  } else {
    rows = DATA.alignmentAvg.map(d => ({label: d.model, value: d.majority})).sort((a,b) => b.value - a.value);
    suffix = '%';
    max = 90;
  }
  barChart(el, rows, {max, suffix});
  const note = alignmentNotes[metric];
  document.querySelector('#alignment-interpretation').innerHTML = `<strong>${note.title}</strong><p>${note.text}</p>`;
}

function initModelTabs() {
  document.querySelectorAll('[data-view]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-view]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      ['model-chart','region-chart','significance-table'].forEach(id => document.getElementById(id).classList.add('hidden'));
      const view = btn.dataset.view;
      setModelViewNote(view);
      const target = view === 'models' ? 'model-chart' : view === 'regions' ? 'region-chart' : 'significance-table';
      document.getElementById(target).classList.remove('hidden');
    });
  });
}

function initAlignmentTabs() {
  const group = document.querySelector('[aria-label="Alignment metric view"]');
  if (!group) return;
  group.addEventListener('click', event => {
    const btn = event.target.closest('[data-align]');
    if (!btn) return;
    group.querySelectorAll('[data-align]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    renderAlignment(btn.dataset.align);
  });
}

function initExamples() {
  let idx = 0;
  const set = () => {
    const ex = DATA.examples[idx];
    document.querySelector('#example-count').textContent = `${idx + 1} / ${DATA.examples.length}`;
    document.querySelector('#example-title').textContent = ex.title;
    document.querySelector('#example-scenario').textContent = ex.scenario;
    document.querySelector('#example-culture').innerHTML = `<span class="example-choice-label">Follow culture</span>${ex.culture.replace(/^Follow culture:\s*/i, '')}`;
    document.querySelector('#example-preference').innerHTML = `<span class="example-choice-label">Allow preference</span>${ex.preference.replace(/^Allow preference:\s*/i, '')}`;
    document.querySelector('#example-note').textContent = ex.note;
  };
  document.querySelectorAll('[data-example-dir]').forEach(btn => btn.addEventListener('click', () => {
    idx = (idx + (btn.dataset.exampleDir === 'next' ? 1 : -1) + DATA.examples.length) % DATA.examples.length;
    set();
  }));
  set();
}

function initHeroChoice() {
  const text = {
    culture: 'Follow the host-home norm and advise removing shoes.',
    preference: 'Allow the guest to keep shoes on because the stated preference is personally important.'
  };
  document.querySelectorAll('.choice').forEach(btn => btn.addEventListener('click', () => {
    document.querySelectorAll('.choice').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelector('#choice-explainer').textContent = text[btn.dataset.choice];
  }));
}

document.addEventListener('DOMContentLoaded', () => {
  renderModelCharts();
  renderHumanChart();
  renderAlignment('majority');
  initModelTabs();
  initAlignmentTabs();
  initExamples();
  initHeroChoice();
});
