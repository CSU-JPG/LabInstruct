#!/usr/bin/env node
// ═══════════════════════════════════════════════════════════════════
// build_data.mjs — build the data.js that survey.html loads, from the
// specs / checklists under data/ plus the media paths under first_frames/
// and outputs/. Carries task metadata, checklist items, prompt text and the
// Chinese translations the survey's zh/en switch falls back from.
//
// Usage:  node build_data.mjs [--if-stale]
// Output: ./data.js   (loaded directly by survey.html)
//
// --if-stale rebuilds only when data.js is missing or older than its inputs;
// the start scripts use it so a double-click is a no-op once the build is
// current.
// ═══════════════════════════════════════════════════════════════════
import { readdirSync, readFileSync, writeFileSync, statSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const OUT_DIR = dirname(fileURLToPath(import.meta.url)); // human_eval/
const ROOT = join(OUT_DIR, '..');                        // data/
const CHECKLISTS_DIR = join(ROOT, 'checklists');
const SPECS_DIR = join(ROOT, 'specs');
const PROMPTS_ZH = JSON.parse(readFileSync(join(OUT_DIR, 'prompts.json'), 'utf8'));

// Display names, matching bench/models.yaml and the paper's table.
const MODEL_LABELS = {
  'cosmos3-super': 'Cosmos3-Super',
  'cosmos3-nano': 'Cosmos3-Nano',
  'ltx2.3': 'LTX-2.3',
  'minimax-h3': 'MiniMax H3',
  'wan2.2': 'Wan2.2-I2V-A14B',
  'wan3.0': 'Wan3.0',
  'lingbot-video': 'LingBot-Video',
  'seedance2.0': 'Seedance 2.0',
};

// Reviewer slots in a fixed order. A model with no generated videos yet keeps
// its slot and shows a missing-video placeholder instead of shifting the others.
const MODEL_ORDER = [
  'ltx2.3',
  'wan2.2',
  'cosmos3-nano',
  'cosmos3-super',
  'lingbot-video',
  'minimax-h3',
  'seedance2.0',
  'wan3.0',
];

// Chinese discipline labels, shown when the survey is switched to zh.
const DISC_LABELS = {
  agronomy: '农学',
  materials_science: '材料科学',
  chemistry: '化学',
  biology: '生物学',
  physics: '物理学',
};

// The Chinese title / prompt_for_gen come from prompts.json in this directory
// and the per-item question_zh from each checklist; the survey falls back to
// the English text wherever a translation is missing.
const models = MODEL_ORDER;

const OUT_FILE = join(OUT_DIR, 'data.js');

if (process.argv.includes('--if-stale') && existsSync(OUT_FILE)) {
  const inputs = [join(OUT_DIR, 'prompts.json'), fileURLToPath(import.meta.url)];
  for (const d of [CHECKLISTS_DIR, SPECS_DIR]) {
    for (const f of readdirSync(d)) inputs.push(join(d, f));
  }
  const newest = Math.max(...inputs.map((f) => statSync(f).mtimeMs));
  if (statSync(OUT_FILE).mtimeMs >= newest) {
    console.log('data.js is up to date with specs/ and checklists/; skipping the build.');
    process.exit(0);
  }
}

const checklistFiles = readdirSync(CHECKLISTS_DIR).filter((f) => f.endsWith('_qa.json')).sort();
const tasks = [];

for (const f of checklistFiles) {
  const cl = JSON.parse(readFileSync(join(CHECKLISTS_DIR, f), 'utf8'));
  const taskId = cl.task_id;

  const spec = JSON.parse(readFileSync(join(SPECS_DIR, cl.source_spec), 'utf8'));
  const pzh = PROMPTS_ZH[taskId] || {};

  // Per-model preview path for this task's generated video.
  const generated = {};
  for (const mid of models) {
    generated[mid] = `../outputs/${mid}/${taskId}/video.mp4`;
  }

  tasks.push({
    id: taskId,
    level: cl.task_level || (taskId.includes('_L2_') ? 'L2' : 'L1'),
    discipline: spec.discipline || '',
    domain: spec.domain || '',
    disciplineLabel: DISC_LABELS[spec.discipline] || spec.discipline || '',
    title: spec.title || taskId,
    description: spec.description || '',
    promptForGen: spec.prompt_for_gen || '',
    titleCn: pzh.title_zh || '',
    promptForGenCn: pzh.prompt_for_gen_zh || '',
    sourceVideo: `../video_clips/${taskId}.mp4`,
    poster: `../first_frames/${taskId}.jpg`,
    generated,
    items: cl.items.map((it) => ({ ...it, questionCn: it.question_zh || '' })),
  });
}

const data = {
  generatedAt: new Date().toISOString(),
  models: models.map((id) => ({ id, label: MODEL_LABELS[id] || id })),
  tasks,
};

writeFileSync(OUT_FILE, 'window.SURVEY_DATA = ' + JSON.stringify(data, null, 2) + ';\n', 'utf8');
console.log(`wrote data.js: ${tasks.length} tasks, ${models.length} models`);
for (const m of models) console.log(`  - ${m} -> ${MODEL_LABELS[m] || m}`);
