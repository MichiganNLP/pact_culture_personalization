const DATA = window.PACT_DATA;
const maxOf = (arr, key) => Math.max(...arr.map(d => Math.abs(d[key])));

function barChart(el, rows, options = {}) {
  const max = options.max ?? maxOf(rows, 'value');
  el.innerHTML = rows.map(row => {
    const v = row.value;
    const pct = Math.min(100, Math.abs(v) / max * 100);
    const cls = v < 0 ? 'negative' : '';
    const suffix = options.suffix ?? '%';
    return `<div class="bar-row">
      <div class="bar-label">${row.label}</div>
      <div class="bar-track"><div class="bar-fill ${cls}" style="width:${pct}%"></div></div>
      <div class="bar-value">${v}${suffix}</div>
    </div>`;
  }).join('');
}

function renderModelCharts() {
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
  if (metric === 'uncertainty') {
    const avg = Object.values(DATA.uncertainty.reduce((acc, d) => {
      if (!acc[d.model]) acc[d.model] = {label: d.model, vals: []};
      if (d.corr !== null) acc[d.model].vals.push(d.corr);
      return acc;
    }, {})).map(d => ({label: d.label, value: +(d.vals.reduce((a,b)=>a+b,0) / d.vals.length).toFixed(3)}));
    barChart(el, avg.sort((a,b)=>b.value-a.value), {max: .28, suffix: ''});
    return;
  }
  const key = metric === 'majority' ? 'majority' : metric === 'mae' ? 'mae' : 'cultureGap';
  const suffix = metric === 'mae' ? '' : metric === 'gap' ? ' pp' : '%';
  const rows = DATA.alignment.map(d => ({label: `${d.model} ${d.frame}`, value: d[key]}));
  barChart(el, rows, {max: metric === 'majority' ? 90 : metric === 'mae' ? .55 : 40, suffix});
}

function initTabs() {
  document.querySelectorAll('[data-view]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-view]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      ['model-chart','region-chart','significance-table'].forEach(id => document.getElementById(id).classList.add('hidden'));
      const target = btn.dataset.view === 'models' ? 'model-chart' : btn.dataset.view === 'regions' ? 'region-chart' : 'significance-table';
      document.getElementById(target).classList.remove('hidden');
    });
  });
}

function initExamples() {
  let idx = 0;
  const set = () => {
    const ex = DATA.examples[idx];
    document.querySelector('#example-count').textContent = `${idx + 1} / ${DATA.examples.length}`;
    document.querySelector('#example-title').textContent = ex.title;
    document.querySelector('#example-scenario').textContent = ex.scenario;
    document.querySelector('#example-culture').textContent = ex.culture;
    document.querySelector('#example-preference').textContent = ex.preference;
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
  renderAlignment();
  initTabs();
  initExamples();
  initHeroChoice();
  document.querySelector('#alignment-metric').addEventListener('change', e => renderAlignment(e.target.value));
});
