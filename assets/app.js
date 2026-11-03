// Step Logger page: live count, label + Start/Stop, step count entry, live
// status, 5 s plot, recordings list.
// Python sends "live" {count, count_sigma, cadence, cadence_sigma, detector,
// use_det}, "samples" ([t_ms, ax, ay, az, gx, gy, gz] rows), "status" and
// "recordings" {files: [{file, kb, steps}]}; this page sends "reset",
// "mode" {mode: "cnn+det" | "cnn"}, "start" {label}, "stop" and "steps" {file, steps}.

const WINDOW = 250;   // samples on the plot (5 s at 50 Hz)

const ui = new WebUI();
const $ = id => document.getElementById(id);

let label = 'walk';
let recording = false;
let showGyro = true;   // the swing spike per stride is clearest in the gyro
let buffer = [];

// --- connection ---
ui.on_connect(() => setConn(true));
ui.on_disconnect(() => setConn(false));

function setConn(on) {
  $('conn').textContent = on ? 'connected' : 'disconnected';
  $('conn').className = 'pill ' + (on ? 'on' : 'off');
}

// --- live count ---
document.querySelectorAll('#modes button').forEach(b => {
  b.addEventListener('click', () => ui.send_message('mode', { mode: b.dataset.mode }));
});
$('reset').addEventListener('click', () => ui.send_message('reset'));

ui.on_message('live', m => {
  $('count').textContent = Math.round(m.count);
  $('count-sigma').textContent = `± ${m.count_sigma.toFixed(1)}`;
  $('cadence').textContent = `${m.cadence.toFixed(2)} ± ${m.cadence_sigma.toFixed(2)} /s`;
  $('det-count').textContent = m.detector;
  const mode = m.use_det ? 'cnn+det' : 'cnn';
  document.querySelectorAll('#modes button').forEach(b => b.classList.toggle('selected', b.dataset.mode === mode));
});

// --- controls ---
document.querySelectorAll('#labels button').forEach(b => {
  b.addEventListener('click', () => {
    if (recording) return;
    label = b.dataset.label;
    document.querySelectorAll('#labels button').forEach(o => o.classList.toggle('selected', o === b));
  });
});

$('record').addEventListener('click', () => {
  if (recording) ui.send_message('stop');
  else {
    closeSteps();
    ui.send_message('start', { label });
  }
});

$('show-accel').addEventListener('click', () => setView(false));
$('show-gyro').addEventListener('click', () => setView(true));

function setView(gyro) {
  showGyro = gyro;
  $('show-accel').classList.toggle('selected', !gyro);
  $('show-gyro').classList.toggle('selected', gyro);
  draw();
}

// --- step count entry ---
function openSteps(file, steps) {
  $('steps-file').textContent = file;
  $('steps-input').value = steps ?? '';
  $('steps-entry').hidden = false;
  $('steps-input').focus();
}

function closeSteps() {
  $('steps-entry').hidden = true;
}

$('steps-entry').addEventListener('submit', e => {
  e.preventDefault();
  const steps = parseInt($('steps-input').value, 10);
  if (!(steps >= 0)) return;
  ui.send_message('steps', { file: $('steps-file').textContent, steps });
  closeSteps();
});

// --- messages from the board ---
ui.on_message('status', s => {
  // Just stopped a walk: ask for its step count
  if (recording && !s.recording && s.label === 'walk') openSteps(s.file);
  recording = s.recording;
  $('rec-badge').hidden = !recording;
  $('record').textContent = recording ? 'Stop' : 'Start';
  $('record').classList.toggle('recording', recording);
  document.querySelectorAll('#labels button').forEach(b => (b.disabled = recording));
  $('file').textContent = s.file || '–';
  $('elapsed').textContent = `${s.elapsed.toFixed(1)} s`;
  $('rows').textContent = s.rows;
  $('hz').textContent = s.hz ? `${s.hz} Hz` : '– Hz';
  $('gap').textContent = s.max_gap ? `${s.max_gap} ms` : '– ms';
  $('dropped').textContent = s.dropped;
});

ui.on_message('samples', m => {
  buffer = buffer.concat(m.samples).slice(-WINDOW);
  draw();
});

ui.on_message('recordings', m => {
  const list = $('recordings');
  list.innerHTML = '';
  if (!m.files.length) {
    list.innerHTML = '<li class="empty">None yet</li>';
    return;
  }
  for (const f of m.files) {
    const li = document.createElement('li');
    const steps = f.steps == null ? '<span class="missing">needs steps</span>' : `${f.steps} steps`;
    li.innerHTML = `<span>${f.file}</span><span>${steps} · ${f.kb} KB</span>`;
    if (f.file.startsWith('walk_')) {
      li.classList.add('editable');
      li.addEventListener('click', () => { if (!recording) openSteps(f.file, f.steps); });
    }
    list.appendChild(li);
  }
});

// --- plot ---
const canvas = $('plot');
const ctx = canvas.getContext('2d');

function draw() {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w * dpr) { canvas.width = w * dpr; canvas.height = h * dpr; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const css = getComputedStyle(document.documentElement);
  const cols = showGyro ? [4, 5, 6] : [1, 2, 3];
  const colors = ['--x', '--y', '--z'].map(c => css.getPropertyValue(c));

  // Symmetric y range that grows to fit the data, with a minimum
  let max = showGyro ? 100 : 2;
  for (const s of buffer) for (const c of cols) max = Math.max(max, Math.abs(s[c]));
  const y = v => h / 2 - (v / max) * (h / 2 - 4);
  const x = i => (i / (WINDOW - 1)) * w;

  // zero line and range labels
  ctx.strokeStyle = css.getPropertyValue('--border');
  ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
  ctx.fillStyle = css.getPropertyValue('--muted');
  ctx.font = '11px system-ui';
  ctx.fillText(`+${max.toFixed(showGyro ? 0 : 1)}`, 4, 12);
  ctx.fillText(`-${max.toFixed(showGyro ? 0 : 1)}`, 4, h - 4);

  const offset = WINDOW - buffer.length;   // newest sample on the right edge
  cols.forEach((c, k) => {
    ctx.strokeStyle = colors[k];
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    buffer.forEach((s, i) => {
      const px = x(i + offset), py = y(s[c]);
      i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    });
    ctx.stroke();
  });
}

window.addEventListener('resize', draw);
draw();
