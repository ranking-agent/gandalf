/* ===========================================================================
   GANDALF Explorer — frontend logic (vanilla JS, no build step).

   Mental model
   ------------
   A traversal is a `path`: a list of "steps". Each step has a root node and,
   once the user drills in, a chosen relationship (predicate + direction) and a
   chosen neighbour. The chosen neighbour becomes the root of the next step.

   The screen shows ONE step at a time as a panel of three columns:
       [ root node ] [ relationships ] [ connected nodes ]
   Panels live side by side in a track; navigating slides the track left/right.
   The top filmstrip shows the whole path with a frame around the current triple.
   ========================================================================= */

'use strict';

const CHUNK = 40;        // neighbours rendered per scroll chunk
const DEGREE_CAP = 300;  // max neighbours we fetch degrees for per relationship

const EXAMPLES = [
  { name: 'Metformin',         id: 'CHEBI:6801',      kind: 'Drug' },
  { name: 'Aspirin',           id: 'CHEBI:15365',     kind: 'Drug' },
  { name: 'Caffeine',          id: 'CHEBI:27732',     kind: 'Chemical' },
  { name: 'TP53',              id: 'NCBIGene:7157',   kind: 'Gene' },
  { name: 'EGFR',              id: 'NCBIGene:1956',   kind: 'Gene' },
  { name: 'Type 2 diabetes',   id: 'MONDO:0005148',   kind: 'Disease' },
  { name: 'Asthma',            id: 'MONDO:0004979',   kind: 'Disease' },
  { name: 'Seizure',           id: 'HP:0001250',      kind: 'Phenotype' },
];

// ---- state ---------------------------------------------------------------
/** @type {Array<Step>} */
let path = [];
let current = 0;

const degreeCache = new Map();   // curie -> degree|null
const expandCache = new Map();   // curie -> raw expand result

const panelEls = [];             // DOM panel per step index

// ---- dom refs ------------------------------------------------------------
const $ = (sel, root = document) => root.querySelector(sel);
const track = $('#track');
const stage = $('#stage');
const pathEl = $('#path');
const tripleFrame = $('#tripleFrame');
const overlay = $('#overlay');
const prevBtn = $('#prevBtn');
const nextBtn = $('#nextBtn');
const stepReadout = $('#stepReadout');

// ---- small helpers -------------------------------------------------------
const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html != null) n.innerHTML = html;
  return n;
};
const esc = (s) => String(s == null ? '' : s)
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const predLabel = (p) => String(p || '').replace(/^biolink:/, '').replace(/_/g, ' ');
const catLabel = (c) => String(c || '').replace(/^biolink:/, '');

// Biolink categories arrive as an unordered mix of specific classes and their
// abstract ancestors/mixins. To group at the most granular level we rank by
// rough hierarchy depth and keep the deepest. Abstract/mixin classes get low
// ranks; concrete leaves get high ones. Unknown classes default to 26 — above
// every abstract ancestor below, but below explicit leaves.
const CAT_RANK = {
  // roots / abstract / mixins
  Entity: 1, NamedThing: 5, PhysicalEssenceOrOccurrent: 5, Occurrent: 6,
  PhysicalEssence: 6, ThingWithTaxon: 6, SubjectOfInvestigation: 6,
  Attribute: 8, OntologyClass: 8, InformationContentEntity: 8,
  BiologicalEntity: 10, ChemicalEntityOrGeneOrGeneProduct: 12,
  ChemicalEntityOrProteinOrPolypeptide: 12, ChemicalOrDrugOrTreatment: 12,
  GenomicEntity: 14, MacromolecularMachineMixin: 14,
  GeneOrGeneProduct: 16, DiseaseOrPhenotypicFeature: 16,
  BiologicalProcessOrActivity: 16, PhysiologicalEntity: 16,
  ChemicalEntity: 18, AnatomicalEntity: 18, OrganismTaxon: 20, MolecularEntity: 22,
  ChemicalMixture: 22, MolecularMixture: 24, ComplexMolecularMixture: 24,
  // concrete leaves
  SmallMolecule: 30, Drug: 30, Food: 30, Disease: 30, PhenotypicFeature: 30,
  ClinicalFinding: 30, Pathway: 30, BiologicalProcess: 30, MolecularActivity: 30,
  PhysiologicalProcess: 30, Behavior: 30, Cell: 30, GrossAnatomicalStructure: 30,
  Genotype: 30, Haplotype: 30, Treatment: 28, DrugExposure: 28, Procedure: 28,
  ClinicalIntervention: 28, Device: 28, DiagnosticAid: 28,
  BehavioralFeature: 32, CellularComponent: 32, SequenceVariant: 32,
  MacromolecularComplex: 32, Gene: 34, GeneProduct: 34, Transcript: 34,
  Polypeptide: 34, Snv: 34, Protein: 36, RNAProduct: 36, NoncodingRNAProduct: 38,
  ProteinIsoform: 38, MicroRNA: 40, SiRNA: 40,
};
const catRank = (c) => {
  const r = CAT_RANK[String(c).replace(/^biolink:/, '')];
  return r == null ? 26 : r;
};
function primaryCategory(cats) {
  if (!cats || !cats.length) return '__unknown__';
  let best = cats[0], bestR = catRank(cats[0]);
  for (const c of cats) {
    const r = catRank(c);
    if (r > bestR) { best = c; bestR = r; }
  }
  return best;
}
const fmt = (n) => (n == null ? '—' : Number(n).toLocaleString());
const arrow = (dir) => (dir === 'in' ? '←' : '→');
function truncate(s, max) {
  s = String(s || '');
  if (s.length <= max) return s;
  const cut = s.slice(0, max);
  const sp = cut.lastIndexOf(' ');
  return (sp > max * 0.6 ? cut.slice(0, sp) : cut).replace(/[\s,;:.]+$/, '') + '…';
}

// =========================================================================
// Start screen
// =========================================================================
function buildExamples() {
  const grid = $('#examples');
  grid.innerHTML = '';
  for (const ex of EXAMPLES) {
    const b = el('button', 'example');
    b.innerHTML =
      `<div class="ex-name">${esc(ex.name)}</div>` +
      `<div class="ex-meta"><span class="ex-id">${esc(ex.id)}</span>` +
      `<span class="ex-kind">${esc(ex.kind)}</span></div>`;
    b.addEventListener('click', () => startFrom(ex.id, ex.name, [`biolink:${ex.kind}`]));
    grid.appendChild(b);
  }
}

function validCurie(s) { return /^[A-Za-z0-9.\-]+:[A-Za-z0-9.\-_:]+$/.test(s); }

let acResults = [];
let acActive = -1;
let acSeq = 0;          // guards against out-of-order responses
let acDebounce = 0;

function wireStart() {
  const input = $('#searchInput');
  const list = $('#acList');
  const err = $('#curieError');

  const closeList = () => { list.hidden = true; input.setAttribute('aria-expanded', 'false'); acActive = -1; };

  const pick = (r) => { err.textContent = ''; closeList(); startFrom(r.curie, r.label || r.curie, r.types || []); };

  const renderList = () => {
    if (!acResults.length) { closeList(); return; }
    list.innerHTML = '';
    acResults.forEach((r, idx) => {
      const item = el('button', `ac-item${idx === acActive ? ' ac-active' : ''}`);
      item.setAttribute('role', 'option');
      const type = (r.types && r.types[0]) ? catLabel(r.types[0]) : '';
      item.innerHTML =
        `<span class="ac-text">` +
          `<span class="ac-label">${esc(r.label || r.curie)}</span>` +
          `<span class="ac-id">${esc(r.curie)}</span>` +
        `</span>` +
        (type ? `<span class="ac-type">${esc(type)}</span>` : '');
      item.addEventListener('mousedown', (e) => { e.preventDefault(); pick(r); });
      item.addEventListener('mouseenter', () => { acActive = idx; highlight(); });
      list.appendChild(item);
    });
    list.hidden = false;
    input.setAttribute('aria-expanded', 'true');
  };

  const highlight = () => {
    [...list.children].forEach((c, idx) => c.classList && c.classList.toggle('ac-active', idx === acActive));
  };

  const note = (html) => { list.innerHTML = `<div class="ac-note">${html}</div>`; list.hidden = false; };

  const runSearch = async (q) => {
    const seq = ++acSeq;
    note(`<span class="spin"></span>searching…`);
    try {
      const r = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
      const data = await r.json();
      if (seq !== acSeq) return; // a newer keystroke superseded this one
      acResults = (data && data.results) || [];
      acActive = -1;
      if (!acResults.length) {
        note(validCurie(q)
          ? `No name match. Press <b>Enter</b> to use “${esc(q)}” as a CURIE.`
          : `No matches. ${data && data.error ? 'Name search is unreachable right now — you can still paste a CURIE.' : 'Try a different term, or paste a CURIE.'}`);
      } else {
        renderList();
      }
    } catch (e) {
      if (seq !== acSeq) return;
      note(`Name search failed. You can still paste a CURIE and press Enter.`);
    }
  };

  input.addEventListener('input', () => {
    const q = input.value.trim();
    err.textContent = '';
    clearTimeout(acDebounce);
    if (q.length < 2) { closeList(); return; }
    acDebounce = setTimeout(() => runSearch(q), 220);
  });

  input.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown' && acResults.length) {
      e.preventDefault(); acActive = (acActive + 1) % acResults.length; highlight();
    } else if (e.key === 'ArrowUp' && acResults.length) {
      e.preventDefault(); acActive = (acActive - 1 + acResults.length) % acResults.length; highlight();
    } else if (e.key === 'Enter') {
      const q = input.value.trim();
      if (acActive >= 0 && acResults[acActive]) { pick(acResults[acActive]); }
      else if (acResults.length === 1) { pick(acResults[0]); }
      else if (validCurie(q)) { err.textContent = ''; closeList(); startFrom(q, q, []); }
      else { err.textContent = 'Pick a result, or paste a CURIE (PREFIX:identifier).'; }
    } else if (e.key === 'Escape') {
      closeList();
    }
  });

  input.addEventListener('blur', () => setTimeout(closeList, 150));

  $('#newStart').addEventListener('click', () => {
    overlay.hidden = false; input.value = ''; err.textContent = '';
    acResults = []; acActive = -1; closeList(); input.focus();
  });
}

// =========================================================================
// Traversal control
// =========================================================================
function startFrom(curie, name, categories) {
  overlay.hidden = true;
  path = [makeStep(curie, name, categories)];
  current = 0;
  // wipe old panels
  for (const p of panelEls) p && p.remove();
  panelEls.length = 0;
  reconcilePanels();
  setCurrent(0);
  loadStep(0);
}

function makeStep(curie, name, categories) {
  return {
    curie,
    name: name || curie,
    categories: categories || [],
    status: 'idle',        // idle | loading | loaded | error
    error: null,
    groups: null,          // [{key,predicate,direction,count,neighbors,...}]
    selectedKey: null,     // expanded predicate group
    selectedNeighborId: null, // the neighbour clicked to advance (persists)
    wiki: { status: 'idle' }, // about-card lookup state
  };
}

async function loadStep(i) {
  const step = path[i];
  if (!step || step.status === 'loaded' || step.status === 'loading') {
    if (step && step.status === 'loaded') updatePanelPreds(i);
    return;
  }
  step.status = 'loading';
  updatePanelPreds(i);

  if (expandCache.has(step.curie)) {
    applyExpand(step, expandCache.get(step.curie));
    finishLoad(i);
    return;
  }
  try {
    const r = await fetch(`/api/expand?curie=${encodeURIComponent(step.curie)}`);
    const data = await r.json();
    if (!r.ok || data.error) throw new Error(data.error || `HTTP ${r.status}`);
    expandCache.set(step.curie, data);
    applyExpand(step, data);
    finishLoad(i);
  } catch (e) {
    step.status = 'error';
    step.error = String(e.message || e);
    updatePanelNode(i);
    updatePanelPreds(i);
  }
}

function applyExpand(step, data) {
  if (data.name) step.name = data.name;
  if (data.categories && data.categories.length) step.categories = data.categories;
  step.serverErrors = data.errors || [];
  step.groups = (data.groups || []).map((g) => ({
    ...g,
    _rendered: 0,
    _degreesLoaded: false,
    _degreeLoading: false,
    _maxDegree: 0,
    _note: '',
  }));
  step.status = 'loaded';
}

function finishLoad(i) {
  updatePanelNode(i);
  updatePanelPreds(i);
  loadWiki(i);
  if (i === current) drawConnectors(i);
}

function selectPredicate(i, key) {
  const step = path[i];
  if (!step || !step.groups) return;
  if (step.selectedKey !== key) step.selectedNeighborId = null;
  step.selectedKey = key;
  // truncate any forward path that was built from a different choice
  if (i < path.length - 1) {
    truncatePathTo(i);
  }
  updatePanelPreds(i);
  updatePanelNeighbors(i);
  drawConnectors(i);
  loadDegrees(i, key);
}

function stepInto(i, neighbor) {
  // record forward path; if diverging, truncate
  truncatePathTo(i);
  path[i].selectedNeighborId = neighbor.id; // persists the connector + highlight
  updatePanelNeighbors(i);                  // re-mark the chosen node on the source panel
  const next = makeStep(neighbor.id, neighbor.name, neighbor.categories);
  path.push(next);
  reconcilePanels();
  renderBreadcrumb();
  setCurrent(i + 1);
  loadStep(i + 1);
}

function truncatePathTo(i) {
  if (path.length > i + 1) {
    const removed = panelEls.splice(i + 1);
    for (const p of removed) p && p.remove();
    path.length = i + 1;
  }
}

// =========================================================================
// Degrees
// =========================================================================
async function loadDegrees(i, key) {
  const step = path[i];
  const group = step.groups.find((g) => g.key === key);
  if (!group || group._degreesLoaded || group._degreeLoading) {
    if (group && group._degreesLoaded) renderNeighborStatus(i, group);
    return;
  }
  group._degreeLoading = true;
  renderNeighborStatus(i, group);

  const slice = group.neighbors.slice(0, DEGREE_CAP);
  const need = slice.map((n) => n.id).filter((id) => !degreeCache.has(id));

  try {
    if (need.length) {
      const r = await fetch('/api/degrees', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ curies: need }),
      });
      const data = await r.json();
      const map = (data && data.degrees) || {};
      for (const id of need) degreeCache.set(id, map[id] != null ? map[id] : null);
    }
  } catch (e) {
    // degrees are best-effort; fall back to unsorted
  }

  // attach degrees + sort (known degrees descending, unknown last)
  for (const n of group.neighbors) n.degree = degreeCache.has(n.id) ? degreeCache.get(n.id) : undefined;
  group.neighbors.sort((a, b) => {
    const da = (a.degree == null) ? -1 : a.degree;
    const db = (b.degree == null) ? -1 : b.degree;
    if (db !== da) return db - da;
    return (a.name || a.id).localeCompare(b.name || b.id);
  });
  group._maxDegree = group.neighbors.reduce((m, n) => Math.max(m, n.degree || 0), 0);
  group._degreesLoaded = true;
  group._degreeLoading = false;
  group._note = group.count > DEGREE_CAP
    ? `sorted by degree · sampled first ${DEGREE_CAP.toLocaleString()} of ${group.count.toLocaleString()}`
    : `sorted by degree · ${group.count.toLocaleString()} node${group.count === 1 ? '' : 's'}`;

  // re-render the list from the top in sorted order
  group._rendered = 0;
  group._catState = null; // re-default the type accordion against the new order
  if (step.selectedKey === key) {
    updatePanelNeighbors(i);
    if (i === current) drawConnectors(i);
  }
}

// =========================================================================
// Panels — create skeletons, then fill in place
// =========================================================================
function reconcilePanels() {
  // create any missing panels
  for (let i = 0; i < path.length; i++) {
    if (!panelEls[i]) {
      const p = buildPanelSkeleton(i);
      panelEls[i] = p;
      track.appendChild(p);
      updatePanelNode(i);
      updatePanelPreds(i);
      updatePanelNeighbors(i);
    }
  }
  // remove extras
  while (panelEls.length > path.length) {
    const p = panelEls.pop();
    p && p.remove();
  }
}

function buildPanelSkeleton(i) {
  const panel = el('div', 'panel');
  panel.dataset.index = String(i);
  panel.innerHTML =
    `<svg class="connector-layer" data-role="svg"></svg>` +
    `<div class="col col-node"><div class="col-head">Node</div>` +
      `<div class="col-scroll" data-role="node"></div></div>` +
    `<div class="col col-pred"><div class="col-head">Relationships</div>` +
      `<div class="col-scroll" data-role="pred"></div></div>` +
    `<div class="col col-nb"><div class="col-head">Connected nodes</div>` +
      `<div class="nb-toolbar" data-role="nbtool"></div>` +
      `<div class="col-scroll" data-role="nb"></div></div>`;

  // redraw connectors when relationship / neighbour columns scroll
  const predScroll = panel.querySelector('[data-role="pred"]');
  const nbScroll = panel.querySelector('[data-role="nb"]');
  predScroll.addEventListener('scroll', () => { if (i === current) drawConnectors(i); }, { passive: true });
  nbScroll.addEventListener('scroll', () => {
    if (i === current) drawConnectors(i);
  }, { passive: true });
  return panel;
}

function updatePanelNode(i) {
  const panel = panelEls[i]; if (!panel) return;
  const step = path[i];
  const host = panel.querySelector('[data-role="node"]');
  const cats = (step.categories || []).map((c) => `<span class="cat-chip">${esc(catLabel(c))}</span>`).join('');
  let meta = '';
  if (step.status === 'loaded') {
    const total = (step.groups || []).reduce((s, g) => s + g.count, 0);
    const rels = (step.groups || []).length;
    meta = `<div class="node-meta"><b>${fmt(rels)}</b> relationship type${rels === 1 ? '' : 's'} · ` +
           `<b>${fmt(total)}</b> connected node slot${total === 1 ? '' : 's'}</div>`;
  } else if (step.status === 'loading') {
    meta = `<div class="node-meta">Loading neighbours…</div>`;
  } else if (step.status === 'error') {
    meta = `<div class="node-meta" style="color:#ffb9a3">Could not load neighbours.</div>`;
  }
  host.innerHTML =
    `<div class="node-card">` +
      `<div class="nc-eyebrow">${i === 0 ? 'Start node' : `Step ${i + 1}`}</div>` +
      `<div class="nc-name">${esc(step.name)}</div>` +
      `<div class="nc-id">${esc(step.curie)}</div>` +
      `<a class="nc-source" href="https://bioregistry.io/${encodeURIComponent(step.curie)}" target="_blank" rel="noopener">Source record ↗</a>` +
      `<div class="cats">${cats}</div>` +
      meta +
    `</div>` +
    `<div class="wiki-card" data-role="wiki"></div>`;
  updatePanelWiki(i);
}

function updatePanelWiki(i) {
  const panel = panelEls[i]; if (!panel) return;
  const step = path[i];
  const host = panel.querySelector('[data-role="wiki"]');
  if (!host) return;
  const w = step.wiki || { status: 'idle' };

  if (w.status === 'idle' || w.status === 'loading') {
    host.innerHTML =
      `<div class="wiki-head">About</div>` +
      `<div class="wiki-loading"><span class="spin"></span> Looking up Wikipedia…</div>`;
    return;
  }
  if (w.status === 'none' || w.status === 'error') {
    host.innerHTML =
      `<div class="wiki-head">About</div>` +
      `<div class="wiki-miss">${w.status === 'error'
        ? 'Wikipedia lookup unavailable.'
        : 'No Wikipedia match by name. Use the source record above.'}</div>`;
    return;
  }
  // loaded
  const d = w.data || {};
  const ambiguous = d.type === 'disambiguation';
  const extract = truncate(d.extract || '', 240);
  host.innerHTML =
    `<div class="wiki-head">About <span class="wiki-src">· Wikipedia</span></div>` +
    (d.description ? `<div class="wiki-desc">${esc(d.description)}</div>` : '') +
    (ambiguous ? `<div class="wiki-warn">Ambiguous title — verify this matches the node.</div>` : '') +
    (extract ? `<div class="wiki-extract">${esc(extract)}</div>` : '') +
    `<a class="wiki-link" href="${esc(d.url || '#')}" target="_blank" rel="noopener">Read on Wikipedia ↗</a>`;
}

const wikiCache = new Map(); // name -> result

async function loadWiki(i) {
  const step = path[i];
  if (!step) return;
  if (step.wiki && step.wiki.status !== 'idle') return;
  const name = (step.name || '').trim();
  // a bare CURIE isn't a useful search term
  if (!name || name === step.curie || validCurie(name)) {
    step.wiki = { status: 'none' };
    updatePanelWiki(i);
    return;
  }
  if (wikiCache.has(name)) {
    const r = wikiCache.get(name);
    step.wiki = r.found ? { status: 'loaded', data: r } : { status: 'none' };
    updatePanelWiki(i);
    return;
  }
  step.wiki = { status: 'loading' };
  updatePanelWiki(i);
  try {
    const r = await fetch(`/api/wiki?name=${encodeURIComponent(name)}`);
    const data = await r.json();
    wikiCache.set(name, data);
    step.wiki = data && data.found ? { status: 'loaded', data } : { status: 'none' };
  } catch (e) {
    step.wiki = { status: 'error' };
  }
  updatePanelWiki(i);
}

function updatePanelPreds(i) {
  const panel = panelEls[i]; if (!panel) return;
  const step = path[i];
  const host = panel.querySelector('[data-role="pred"]');

  if (step.status === 'loading' || step.status === 'idle') {
    host.innerHTML = helixLoaderMarkup();
    startHelixLoader();
    return;
  }
  if (step.status === 'error') {
    host.innerHTML =
      `<div class="error-block">Couldn’t reach the knowledge graph.<br>` +
      `<small style="font-family:var(--font-mono)">${esc(step.error)}</small><br>` +
      `<button class="retry">Try again</button></div>`;
    host.querySelector('.retry').addEventListener('click', () => { step.status = 'idle'; loadStep(i); });
    return;
  }
  if (!step.groups || step.groups.length === 0) {
    if (step.serverErrors && step.serverErrors.length) {
      host.innerHTML =
        `<div class="error-block">The knowledge graph returned no usable response.<br>` +
        `<small style="font-family:var(--font-mono)">${esc(step.serverErrors.join('; '))}</small><br>` +
        `<button class="retry">Try again</button></div>`;
      host.querySelector('.retry').addEventListener('click', () => {
        expandCache.delete(step.curie);
        step.status = 'idle';
        loadStep(i);
      });
      return;
    }
    host.innerHTML = `<div class="empty">No relationships found for this node in the graph.</div>`;
    return;
  }

  const list = el('div', 'pred-list');
  for (const g of step.groups) {
    const b = el('button', `pred-pill dir-${g.direction}${step.selectedKey === g.key ? ' is-selected' : ''}`);
    b.dataset.key = g.key;
    b.dataset.dir = g.direction;
    b.innerHTML =
      `<span class="pp-dir">${arrow(g.direction)}</span>` +
      `<span class="pp-text">` +
        `<span class="pp-name">${esc(predLabel(g.predicate))}</span>` +
        `<span class="pp-full">${g.direction === 'in' ? 'incoming' : 'outgoing'} · ${esc(g.predicate)}</span>` +
      `</span>` +
      `<span class="pp-count">${fmt(g.count)}</span>`;
    b.addEventListener('click', () => selectPredicate(i, g.key));
    list.appendChild(b);
  }
  host.innerHTML = '';
  host.appendChild(list);
}

function renderNeighborStatus(i, group) {
  const panel = panelEls[i]; if (!panel) return;
  const tool = panel.querySelector('[data-role="nbtool"]');
  if (!group) { tool.innerHTML = ''; return; }
  let txt;
  if (group._degreeLoading && !group._degreesLoaded) {
    txt = `<span class="spin"></span>loading node degrees…`;
  } else if (group._degreesLoaded) {
    txt = esc(group._note);
  } else {
    txt = `${fmt(group.count)} node${group.count === 1 ? '' : 's'}`;
  }
  tool.innerHTML = `<span class="nb-status">${txt}</span>`;
}

function updatePanelNeighbors(i) {
  const panel = panelEls[i]; if (!panel) return;
  const step = path[i];
  const host = panel.querySelector('[data-role="nb"]');
  const tool = panel.querySelector('[data-role="nbtool"]');

  if (!step.selectedKey || !step.groups) {
    tool.innerHTML = '';
    host.innerHTML = `<div class="empty">Pick a relationship on the left to list the nodes connected through it.</div>`;
    return;
  }
  const group = step.groups.find((g) => g.key === step.selectedKey);
  if (!group) { host.innerHTML = ''; return; }

  renderNeighborStatus(i, group);

  if (group.count === 0) {
    host.innerHTML = `<div class="empty">No nodes connected via this relationship.</div>`;
    return;
  }

  // partition neighbours by primary node type, preserving (degree-sorted) order
  const cats = buildCategories(group);
  group._cats = cats;
  if (!group._catState) group._catState = {};
  for (const c of cats) {
    if (group._catState[c.cat] === undefined) {
      group._catState[c.cat] = { expanded: false, rendered: 0 };
    }
  }
  // default expansion: the chosen node's type, else the largest type
  const chosenCat = chosenCategory(group, step.selectedNeighborId);
  const anyOpen = cats.some((c) => group._catState[c.cat].expanded);
  if (!anyOpen) {
    const want = chosenCat || (cats[0] && cats[0].cat);
    if (want) group._catState[want].expanded = true;
  }
  if (chosenCat) group._catState[chosenCat].expanded = true;

  host.innerHTML = '';
  const acc = el('div', 'cat-acc');
  for (const c of cats) {
    const st = group._catState[c.cat];
    const head = el('button', `cat-head${st.expanded ? ' is-open' : ''}`);
    head.innerHTML =
      `<span class="ch-twist">▸</span>` +
      `<span class="ch-label">${esc(c.label)}</span>` +
      `<span class="ch-count">${fmt(c.nodes.length)}</span>`;
    const body = el('div', 'cat-nodes');
    if (!st.expanded) body.style.display = 'none';
    c._body = body;
    head.addEventListener('click', () => {
      st.expanded = !st.expanded;
      head.classList.toggle('is-open', st.expanded);
      body.style.display = st.expanded ? '' : 'none';
      if (st.expanded && st.rendered === 0) renderCatChunk(i, group, c, body);
      drawConnectors(i);
    });
    acc.appendChild(head);
    acc.appendChild(body);
    if (st.expanded) { st.rendered = 0; renderCatChunk(i, group, c, body); }
  }
  host.appendChild(acc);

  if (group.count > DEGREE_CAP && group._degreesLoaded) {
    host.appendChild(el('div', 'muted-hint',
      `Degree ranking covers the first ${DEGREE_CAP.toLocaleString()} of ${group.count.toLocaleString()} nodes ` +
      `via this relationship; the rest follow unranked.`));
  }

  if (step.selectedNeighborId) ensureChosenVisible(i, group);
}

function buildCategories(group) {
  const map = new Map();
  for (const n of group.neighbors) {
    const key = primaryCategory(n.categories);
    if (!map.has(key)) map.set(key, []);
    map.get(key).push(n);
  }
  const arr = [...map.entries()].map(([cat, nodes]) => ({
    cat,
    label: cat === '__unknown__' ? 'Other / untyped' : catLabel(cat),
    nodes,
  }));
  arr.sort((a, b) => b.nodes.length - a.nodes.length || a.label.localeCompare(b.label));
  return arr;
}

function chosenCategory(group, id) {
  if (!id) return null;
  const n = group.neighbors.find((x) => x.id === id);
  if (!n) return null;
  return primaryCategory(n.categories);
}

function nbItem(i, group, n) {
  const step = path[i];
  const item = el('button', `nb-item${n.id === step.selectedNeighborId ? ' is-chosen' : ''}`);
  item.dataset.nid = n.id;
  let deg;
  if (n.degree === undefined) {
    deg = `<span class="deg-val unknown">·</span>`;
  } else if (n.degree == null) {
    deg = `<span class="deg-val unknown">n/a</span>`;
  } else {
    const pct = group._maxDegree ? Math.max(4, Math.round((n.degree / group._maxDegree) * 100)) : 0;
    deg = `<span class="deg-val">${fmt(n.degree)}</span>` +
          `<span class="deg-bar"><i style="width:${pct}%"></i></span>`;
  }
  item.innerHTML =
    `<span class="nb-text">` +
      `<span class="nb-name">${esc(n.name || n.id)}</span>` +
      `<span class="nb-id">${esc(n.id)}</span>` +
    `</span>` +
    `<span class="nb-degree" title="node degree">${deg}</span>` +
    `<span class="nb-go">›</span>`;
  item.addEventListener('click', () => stepInto(i, n));
  return item;
}

function renderCatChunk(i, group, c, body) {
  const st = group._catState[c.cat];
  const oldMore = body.querySelector('.show-more');
  if (oldMore) oldMore.remove();
  const end = Math.min(st.rendered + CHUNK, c.nodes.length);
  for (let k = st.rendered; k < end; k++) body.appendChild(nbItem(i, group, c.nodes[k]));
  st.rendered = end;
  const remaining = c.nodes.length - st.rendered;
  if (remaining > 0) {
    const more = el('button', 'show-more',
      `Show ${Math.min(CHUNK, remaining)} more · ${fmt(remaining)} left`);
    more.addEventListener('click', () => { renderCatChunk(i, group, c, body); drawConnectors(i); });
    body.appendChild(more);
  }
}

function ensureChosenVisible(i, group) {
  const id = path[i].selectedNeighborId;
  if (!id || !group._cats) return;
  let cat = null, idx = -1;
  for (const c of group._cats) {
    const k = c.nodes.findIndex((n) => n.id === id);
    if (k >= 0) { cat = c; idx = k; break; }
  }
  if (!cat || !cat._body) return;
  const st = group._catState[cat.cat];
  st.expanded = true;
  cat._body.style.display = '';
  const head = cat._body.previousElementSibling;
  if (head) head.classList.add('is-open');
  while (st.rendered <= idx && st.rendered < cat.nodes.length) renderCatChunk(i, group, cat, cat._body);
  const elc = cat._body.querySelector(`.nb-item[data-nid="${id.replace(/"/g, '\\"')}"]`);
  if (elc) elc.scrollIntoView({ block: 'nearest' });
}

// =========================================================================
// DNA helix loader (shown in the relationships column while TRAPI responds)
// =========================================================================
function prefersReducedMotion() {
  return !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
}

function helixLoaderMarkup() {
  if (prefersReducedMotion()) {
    return `<div class="loading-block"><span class="spin"></span> Querying TRAPI for neighbours…</div>`;
  }
  const N = 16, H = 230, marginY = 20, step = (H - 2 * marginY) / (N - 1);
  let rungs = '';
  for (let i = 0; i < N; i++) {
    const y = (marginY + i * step).toFixed(1);
    rungs +=
      `<g class="rung" data-y="${y}" data-i="${i}">` +
      `<line class="bond" x1="120" y1="${y}" x2="120" y2="${y}"/>` +
      `<circle class="s2" cx="120" cy="${y}" r="3"/>` +
      `<circle class="s1" cx="120" cy="${y}" r="3"/>` +
      `</g>`;
  }
  return (
    `<div class="helix-loader">` +
      `<svg class="helix-svg" viewBox="0 0 240 ${H}" data-amp="78" data-cx="120" aria-hidden="true">${rungs}</svg>` +
      `<div class="helix-caption">Querying TRAPI<span class="dots"><i>.</i><i>.</i><i>.</i></span></div>` +
      `<div class="helix-sub">tracing neighbours along every relationship</div>` +
    `</div>`
  );
}

let helixRaf = 0;
function startHelixLoader() {
  if (helixRaf) return;
  const tick = (t) => {
    const svgs = document.querySelectorAll('svg.helix-svg');
    if (!svgs.length) { helixRaf = 0; return; }     // self-stops when loaders are gone
    svgs.forEach((svg) => updateHelix(svg, t));
    helixRaf = requestAnimationFrame(tick);
  };
  helixRaf = requestAnimationFrame(tick);
}

function updateHelix(svg, t) {
  const A = +svg.dataset.amp;
  const cx = +svg.dataset.cx;
  const tt = t * 0.0024;
  svg.querySelectorAll('.rung').forEach((g) => {
    const i = +g.dataset.i;
    const phase = i * 0.55 + tt;
    const s = Math.sin(phase), c = Math.cos(phase);
    const xL = cx + A * s, xR = cx - A * s;
    const s1 = g.querySelector('.s1'); // teal strand, depth = c
    const s2 = g.querySelector('.s2'); // coral strand, depth = -c
    const bond = g.querySelector('.bond');
    s1.setAttribute('cx', xL.toFixed(2));
    s1.setAttribute('r', (3.0 + 2.0 * ((c + 1) / 2)).toFixed(2));
    s1.setAttribute('opacity', (0.35 + 0.65 * ((c + 1) / 2)).toFixed(2));
    s2.setAttribute('cx', xR.toFixed(2));
    s2.setAttribute('r', (3.0 + 2.0 * ((-c + 1) / 2)).toFixed(2));
    s2.setAttribute('opacity', (0.35 + 0.65 * ((-c + 1) / 2)).toFixed(2));
    bond.setAttribute('x1', xL.toFixed(2));
    bond.setAttribute('x2', xR.toFixed(2));
    bond.setAttribute('opacity', (0.12 + 0.32 * Math.abs(s)).toFixed(2));
  });
}

// =========================================================================
// Breadcrumb filmstrip + the moving "viewing" frame
// =========================================================================
function renderBreadcrumb() {
  pathEl.innerHTML = '';
  for (let i = 0; i < path.length; i++) {
    const step = path[i];
    const node = el('button', `crumb-node${i === current ? ' is-root' : ''}`);
    node.dataset.nodeIndex = String(i);
    node.innerHTML =
      `<div class="cn-name">${esc(step.name)}</div>` +
      `<div class="cn-id">${esc(step.curie)}</div>`;
    node.addEventListener('click', () => setCurrent(i));
    pathEl.appendChild(node);

    // predicate connector between node i and node i+1
    if (i < path.length - 1) {
      const g = step.groups && step.selectedKey
        ? step.groups.find((x) => x.key === step.selectedKey) : null;
      const dir = g ? g.direction : 'out';
      const label = g ? predLabel(g.predicate) : 'related to';
      const cp = el('div', `crumb-pred dir-${dir}`);
      cp.dataset.predIndex = String(i);
      cp.innerHTML = `<div class="cp-label">${esc(label)}</div><div class="cp-arrow">${arrow(dir)}</div>`;
      pathEl.appendChild(cp);
    }
  }
  positionTripleFrame();
}

function positionTripleFrame() {
  const wrapRect = pathEl.parentElement.getBoundingClientRect();
  const startNode = pathEl.querySelector(`.crumb-node[data-node-index="${current}"]`);
  if (!startNode) { tripleFrame.style.opacity = '0'; return; }

  const endNode = pathEl.querySelector(`.crumb-node[data-node-index="${current + 1}"]`);
  const els = [startNode];
  const predEl = pathEl.querySelector(`.crumb-pred[data-pred-index="${current}"]`);
  if (endNode) { if (predEl) els.push(predEl); els.push(endNode); }

  let left = Infinity, right = -Infinity, top = Infinity, bottom = -Infinity;
  for (const e of els) {
    const r = e.getBoundingClientRect();
    left = Math.min(left, r.left); right = Math.max(right, r.right);
    top = Math.min(top, r.top);   bottom = Math.max(bottom, r.bottom);
  }
  const padX = 8, padY = 6;
  tripleFrame.style.opacity = '1';
  tripleFrame.style.left = (left - wrapRect.left - padX) + 'px';
  tripleFrame.style.top = (top - wrapRect.top - padY) + 'px';
  tripleFrame.style.width = (right - left + padX * 2) + 'px';
  tripleFrame.style.height = (bottom - top + padY * 2) + 'px';
}

function scrollPathToCurrent() {
  const startNode = pathEl.querySelector(`.crumb-node[data-node-index="${current}"]`);
  if (!startNode) return;
  const r = startNode.getBoundingClientRect();
  const pr = pathEl.getBoundingClientRect();
  if (r.left < pr.left + 40 || r.right > pr.right - 40) {
    startNode.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
  }
}

// =========================================================================
// Connectors (curved SVG links node→predicate and predicate→neighbour)
// =========================================================================
let connectorRaf = 0;
function drawConnectors(i) {
  if (i !== current) return;
  cancelAnimationFrame(connectorRaf);
  connectorRaf = requestAnimationFrame(() => paintConnectors(i));
}

function paintConnectors(i) {
  const panel = panelEls[i]; if (!panel) return;
  const svg = panel.querySelector('[data-role="svg"]');
  const pr = panel.getBoundingClientRect();
  const W = panel.clientWidth, H = panel.clientHeight;
  svg.setAttribute('width', W);
  svg.setAttribute('height', H);
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.innerHTML = '';

  const nodeCard = panel.querySelector('.node-card');
  const predScroll = panel.querySelector('[data-role="pred"]');
  if (!nodeCard) return;

  const nr = nodeCard.getBoundingClientRect();
  const x1 = nr.right - pr.left;
  const y1 = nr.top - pr.top + nr.height / 2;

  // only connect to relationship pills currently visible in their column
  const psRect = predScroll.getBoundingClientRect();
  const pills = panel.querySelectorAll('.pred-pill');
  pills.forEach((pill) => {
    const r = pill.getBoundingClientRect();
    if (r.bottom < psRect.top + 2 || r.top > psRect.bottom - 2) return;
    const x2 = r.left - pr.left;
    const y2 = r.top - pr.top + r.height / 2;
    const selected = pill.classList.contains('is-selected');
    const dir = pill.dataset.dir;
    svg.appendChild(curve(x1, y1, x2, y2, dir, selected ? 0.95 : 0.4, selected ? 2 : 1.25));
  });

  // selected predicate → the chosen node (or the top of the list if none yet)
  const sel = panel.querySelector('.pred-pill.is-selected');
  const nbCol = panel.querySelector('.col-nb');
  if (sel && nbCol) {
    const step = path[i];
    const nbRect = nbCol.getBoundingClientRect();
    const sr = sel.getBoundingClientRect();
    const sx = sr.right - pr.left;
    const sy = sr.top - pr.top + sr.height / 2;
    const tx = nbRect.left - pr.left;

    let chosenEl = null;
    if (step.selectedNeighborId) {
      chosenEl = panel.querySelector(`.nb-item[data-nid="${step.selectedNeighborId.replace(/"/g, '\\"')}"]`);
    }
    const target = chosenEl || (nbCol.querySelector('.nb-item') || nbCol.querySelector('.cat-acc') || nbCol);
    const tr = target.getBoundingClientRect();
    // clamp the endpoint to the visible neighbour viewport so the link stays sensible while scrolled
    const top = nbRect.top - pr.top + 14;
    const bot = nbRect.bottom - pr.top - 14;
    const ty = Math.min(Math.max(tr.top - pr.top + tr.height / 2, top), bot);
    const linked = !!chosenEl;
    svg.appendChild(curve(sx, sy, tx, ty, sel.dataset.dir, linked ? 0.95 : 0.7, linked ? 2.25 : 1.75));
    if (linked) svg.appendChild(endDot(tx, ty, sel.dataset.dir));
  }
}

function endDot(x, y, dir) {
  const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
  c.setAttribute('cx', x); c.setAttribute('cy', y); c.setAttribute('r', '3.5');
  c.setAttribute('fill', dir === 'in' ? 'var(--in)' : 'var(--out)');
  return c;
}

function curve(x1, y1, x2, y2, dir, opacity, width) {
  const dx = Math.max(28, (x2 - x1) * 0.55);
  const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  p.setAttribute('d', `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`);
  p.setAttribute('fill', 'none');
  const color = dir === 'in' ? 'var(--in)' : 'var(--out)';
  p.setAttribute('stroke', color);
  p.setAttribute('stroke-width', String(width));
  p.setAttribute('stroke-opacity', String(opacity));
  p.setAttribute('stroke-linecap', 'round');
  return p;
}

// =========================================================================
// Navigation
// =========================================================================
function setCurrent(i) {
  current = Math.max(0, Math.min(path.length - 1, i));
  track.style.transform = `translateX(${-current * 100}%)`;
  panelEls.forEach((p, idx) => p && p.classList.toggle('is-current', idx === current));
  renderBreadcrumb();
  scrollPathToCurrent();
  updateControls();
  // ensure current step's data is present
  if (path[current] && path[current].status === 'idle') loadStep(current);
  drawConnectors(current);
}

function updateControls() {
  prevBtn.disabled = current <= 0;
  nextBtn.disabled = current >= path.length - 1;
  const step = path[current];
  const where = step ? `${esc(step.name)}` : '';
  stepReadout.innerHTML = `triple <b>${current + 1}</b> of <b>${path.length}</b> · viewing <b>${where}</b>`;
}

// =========================================================================
// Wire-up
// =========================================================================
function init() {
  buildExamples();
  wireStart();

  prevBtn.addEventListener('click', () => setCurrent(current - 1));
  nextBtn.addEventListener('click', () => setCurrent(current + 1));

  document.addEventListener('keydown', (e) => {
    if (!overlay.hidden) return;
    const tag = (e.target && e.target.tagName) || '';
    if (tag === 'INPUT' || tag === 'TEXTAREA') return;
    if (e.key === 'ArrowLeft') { setCurrent(current - 1); }
    else if (e.key === 'ArrowRight') { setCurrent(current + 1); }
  });

  let resizeRaf = 0;
  window.addEventListener('resize', () => {
    cancelAnimationFrame(resizeRaf);
    resizeRaf = requestAnimationFrame(() => { positionTripleFrame(); drawConnectors(current); });
  });
  pathEl.addEventListener('scroll', () => positionTripleFrame(), { passive: true });

  $('#searchInput').focus();
}

document.addEventListener('DOMContentLoaded', init);
