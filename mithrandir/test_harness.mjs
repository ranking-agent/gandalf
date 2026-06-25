import { JSDOM } from 'jsdom';
import fs from 'fs';

const html = fs.readFileSync('./index.html', 'utf8');
const appjs = fs.readFileSync('./app.js', 'utf8');

// ---- mocked backend responses -------------------------------------------
const EXPAND = {
  curie: 'CHEBI:6801', name: 'Mock metformin', categories: ['biolink:ChemicalEntity'],
  errors: [],
  groups: [
    { key: 'out|biolink:affects', predicate: 'biolink:affects', direction: 'out', count: 5,
      neighbors: [
        { id: 'G:1', name: 'Gene one',   categories: ['biolink:Gene'] },
        { id: 'G:2', name: 'Gene two',   categories: ['biolink:Gene'] },
        { id: 'G:3', name: 'Gene three', categories: ['biolink:Gene'] },
        { id: 'P:1', name: 'Prot one',   categories: ['biolink:Protein'] },
        { id: 'P:2', name: 'Prot two',   categories: ['biolink:Protein'] },
      ] },
    { key: 'in|biolink:has_participant', predicate: 'biolink:has_participant', direction: 'in', count: 2,
      neighbors: [
        { id: 'W:1', name: 'Pathway one', categories: ['biolink:Pathway'] },
        { id: 'W:2', name: 'Pathway two', categories: ['biolink:Pathway'] },
      ] },
  ],
};
const DEG = { 'G:1': 10, 'G:2': 900, 'G:3': 50, 'P:1': 5, 'P:2': 7, 'W:1': 3, 'W:2': 4 };
const SEARCH = { results: [{ curie: 'CHEBI:6801', label: 'metformin', types: ['biolink:ChemicalEntity'] }] };

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failures = 0;
const assert = (cond, msg) => { if (!cond) { failures++; console.log('  ✗ ' + msg); } else { console.log('  ✓ ' + msg); } };

const dom = new JSDOM(html.replace('<script src="/app.js"></script>', `<script>${appjs}</script>`), {
  runScripts: 'dangerously',
  pretendToBeVisual: true,
  beforeParse(window) {
    window.fetch = async (url, opts) => {
      const u = String(url);
      let body;
      if (u.startsWith('/api/search')) body = SEARCH;
      else if (u.startsWith('/api/expand')) body = EXPAND;
      else if (u.startsWith('/api/degrees')) {
        const req = JSON.parse(opts.body);
        const degrees = {};
        for (const c of req.curies) degrees[c] = DEG[c] ?? null;
        body = { degrees };
      } else body = {};
      return { ok: true, status: 200, json: async () => body };
    };
    if (!window.Element.prototype.scrollIntoView) window.Element.prototype.scrollIntoView = () => {};
    window.requestAnimationFrame = (cb) => setTimeout(() => cb(Date.now()), 0);
    window.cancelAnimationFrame = (id) => clearTimeout(id);
  },
});

const { window } = dom;
const { document } = window;

async function run() {
  console.log('Autocomplete:');
  const input = document.getElementById('searchInput');
  input.value = 'metf';
  input.dispatchEvent(new window.Event('input'));
  await sleep(350); // debounce + fetch
  const acItems = document.querySelectorAll('#acList .ac-item');
  assert(acItems.length === 1, 'dropdown shows 1 search result');
  assert(/metformin/.test(acItems[0]?.textContent || ''), 'result label rendered');
  acItems[0].dispatchEvent(new window.Event('mousedown'));
  await sleep(50);

  console.log('Start panel + relationships:');
  assert(document.getElementById('overlay').hidden === true, 'overlay closed after pick');
  const panels = document.querySelectorAll('.panel');
  assert(panels.length === 1, 'one panel created');
  const pills = document.querySelectorAll('.panel .pred-pill');
  assert(pills.length === 2, 'two predicate pills');
  // sorted by count desc -> affects(5) before has_participant(2)
  assert(/affects/.test(pills[0].textContent), 'predicates sorted by count desc');
  assert(pills[0].classList.contains('dir-out') && pills[1].classList.contains('dir-in'),
    'direction classes applied (out then in)');

  console.log('Select predicate -> type accordion + degree sort:');
  pills[0].click();
  await sleep(120); // degree fetch + re-render
  const heads = document.querySelectorAll('.panel .cat-head');
  assert(heads.length === 2, 'two type groups (Gene, Protein)');
  // groups sorted by size desc: Gene(3) before Protein(2)
  assert(/Gene/.test(heads[0].textContent) && /3/.test(heads[0].textContent), 'largest type group first with count 3');
  assert(/Protein/.test(heads[1].textContent) && /2/.test(heads[1].textContent), 'second type group Protein count 2');
  // first (largest) group auto-expanded; within it, degree desc => G:2(900),G:3(50),G:1(10)
  const firstBody = heads[0].nextElementSibling;
  const ids = [...firstBody.querySelectorAll('.nb-item')].map((n) => n.dataset.nid);
  assert(JSON.stringify(ids) === JSON.stringify(['G:2', 'G:3', 'G:1']),
    'nodes within group sorted by degree desc (got ' + ids.join(',') + ')');

  console.log('Step into a node -> chosen highlight persists on source panel:');
  const target = firstBody.querySelector('.nb-item[data-nid="G:3"]');
  target.click();
  await sleep(80);
  assert(document.querySelectorAll('.panel').length === 2, 'second panel pushed');
  // source panel (index 0) should mark G:3 as chosen
  const chosen = document.querySelectorAll('.panel')[0].querySelector('.nb-item.is-chosen');
  assert(chosen && chosen.dataset.nid === 'G:3', 'chosen node highlighted on source panel');
  // breadcrumb has 2 node chips + 1 predicate chip
  assert(document.querySelectorAll('#path .crumb-node').length === 2, 'breadcrumb shows 2 nodes');
  assert(document.querySelectorAll('#path .crumb-pred').length === 1, 'breadcrumb shows 1 predicate connector');

  console.log('Navigate back, then diverge to a different node:');
  document.getElementById('prevBtn').click();
  await sleep(60);
  const acc2 = document.querySelectorAll('.panel')[0];
  const g1 = acc2.querySelector('.nb-item[data-nid="G:1"]');
  assert(!!g1, 'back on source panel, neighbours present');
  g1.click();
  await sleep(80);
  assert(document.querySelectorAll('.panel').length === 2, 'divergence truncated forward path (still 2 panels)');
  const chosen2 = document.querySelectorAll('.panel')[0].querySelector('.nb-item.is-chosen');
  assert(chosen2 && chosen2.dataset.nid === 'G:1', 'chosen highlight moved to new node G:1');

  console.log('\n' + (failures ? `FAILED (${failures})` : 'ALL CHECKS PASSED'));
  process.exit(failures ? 1 : 0);
}

window.addEventListener('error', (e) => { failures++; console.log('  ✗ runtime error: ' + (e.error?.stack || e.message)); });
setTimeout(run, 50);
