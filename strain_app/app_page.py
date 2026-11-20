"""The single page of the application (HTML + CSS + JavaScript, no external resources)."""

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>nartfem Strain Calculator</title>
<style>
:root{
  --bg:#f3f2ee; --surface:#fcfcfb; --line:#dddbd3; --line-soft:#ebe9e2;
  --ink:#0b0b0b; --ink2:#52514e; --ink3:#85837c;
  --s1:#2a78d6; --s2:#eb6834; --accent:#1c5cab; --bad:#b3261e; --warn:#8a5a00;
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
header{display:flex;align-items:baseline;gap:14px;padding:12px 20px;border-bottom:1px solid var(--line);background:var(--surface)}
header h1{font-size:17px;margin:0;font-weight:650;letter-spacing:-.01em}
header .sub{color:var(--ink2);font-size:13px}
header .sp{flex:1}
main{display:grid;grid-template-columns:minmax(360px,38%) 1fr;gap:16px;padding:16px 20px;height:calc(100% - 50px)}
.col{display:flex;flex-direction:column;min-height:0;gap:12px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:8px}
.card h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--ink2);margin:0;font-weight:650}
.bar{display:flex;align-items:center;gap:8px;padding:10px 12px;border-bottom:1px solid var(--line-soft);flex-wrap:wrap}
button,select,.filebtn{font:inherit;font-size:13px;padding:6px 11px;border:1px solid var(--line);border-radius:6px;background:#fff;color:var(--ink);cursor:pointer}
button:hover,.filebtn:hover,select:hover{border-color:#b9b6ab}
button:focus-visible,select:focus-visible,textarea:focus-visible{outline:2px solid var(--s1);outline-offset:1px}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600;padding:6px 18px}
button.primary:hover{background:#184f95}
button:disabled{opacity:.45;cursor:default}
button.link{border:0;background:none;color:var(--accent);padding:2px 4px}
#editorCard{flex:1;display:flex;flex-direction:column;min-height:260px}
textarea{flex:1;width:100%;border:0;resize:none;padding:12px 14px;font:12.5px/1.5 var(--mono);background:transparent;color:var(--ink);tab-size:2;white-space:pre;overflow:auto;border-radius:0 0 8px 8px}
#msg{padding:8px 12px;font-size:13px;border-top:1px solid var(--line-soft);display:none}
#msg.err{display:block;color:var(--bad);background:#fdf1ef}
#msg.ok{display:block;color:var(--ink2)}
#logCard{height:170px;display:flex;flex-direction:column}
#log{flex:1;overflow:auto;margin:0;padding:8px 12px;font:12px/1.45 var(--mono);color:var(--ink2);white-space:pre-wrap}
#right{overflow:auto;padding-right:4px}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.tile{padding:12px 14px}
.tile .k{font-size:12px;color:var(--ink2)}
.tile .v{font-size:25px;font-weight:650;letter-spacing:-.02em;margin-top:2px;font-variant-numeric:tabular-nums}
.tile .v small{font-size:12px;font-weight:500;color:var(--ink2);margin-left:4px;letter-spacing:0}
.tile .n{font-size:12px;color:var(--ink3);margin-top:2px}
.tile.main{border-color:#b9cde8}
.pad{padding:12px 14px}
.meta{font-size:13px;color:var(--ink2)}
.meta b{color:var(--ink);font-weight:600}
.flag{color:var(--warn);font-weight:600}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:5px 10px;text-align:right;border-bottom:1px solid var(--line-soft);white-space:nowrap}
th{white-space:normal;max-width:92px;vertical-align:bottom;font-weight:600;color:var(--ink2);font-size:12px;position:sticky;top:0;background:var(--surface)}
tbody tr{cursor:pointer}
tbody tr:hover{background:#f2f5fa}
tbody tr.sel{background:#e6eefa}
.tablewrap{max-height:260px;overflow:auto}
.chart{position:relative}
.chart svg{display:block;width:100%;height:auto}
.legend{display:flex;gap:16px;font-size:12.5px;color:var(--ink2);padding:0 14px 4px}
.legend i{display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}
.tip{position:absolute;pointer-events:none;background:#fff;border:1px solid var(--line);border-radius:6px;padding:6px 9px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.08);display:none;white-space:nowrap;font-variant-numeric:tabular-nums}
.tip i{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}
#empty{color:var(--ink2);padding:40px 20px;text-align:center}
#prog{height:3px;background:var(--line-soft);border-radius:2px;overflow:hidden;flex:1;min-width:60px;visibility:hidden}
#prog div{height:100%;width:0;background:var(--s1);transition:width .2s}
details{font-size:13px;color:var(--ink2)}
details summary{cursor:pointer;color:var(--accent)}
details pre{font:12px/1.5 var(--mono);background:#f6f5f1;border:1px solid var(--line-soft);border-radius:6px;padding:10px;overflow:auto;color:var(--ink)}
details td,details th{text-align:left;white-space:normal;vertical-align:top}
@media (max-width:1000px){main{grid-template-columns:1fr;height:auto}.tiles{grid-template-columns:repeat(2,1fr)}#right{overflow:visible}}
</style>
</head>
<body>
<header>
  <h1>nartfem Strain Calculator</h1>
  <span class="sub">critical pavement strains for single, dual and wide-base tire loads</span>
  <span class="sp"></span>
  <span class="sub" id="ver"></span>
  <button id="quit" title="Close the application">Quit</button>
</header>
<main>
  <div class="col">
    <div class="card" id="editorCard">
      <div class="bar">
        <h2>Input</h2><span style="flex:1"></span>
        <select id="examples" aria-label="Load an example"><option value="">Examples&hellip;</option></select>
        <label class="filebtn">Open&hellip;<input type="file" id="file" accept=".json,.txt" hidden></label>
        <button id="save">Save</button>
      </div>
      <textarea id="input" spellcheck="false" aria-label="Model input (JSON)"></textarea>
      <div id="msg"></div>
      <div class="bar" style="border-top:1px solid var(--line-soft);border-bottom:0">
        <button class="primary" id="run">Run</button>
        <button id="stop" disabled>Stop</button>
        <div id="prog"><div></div></div>
        <span class="meta" id="progText"></span>
      </div>
    </div>
    <div class="card" id="logCard">
      <div class="bar"><h2>Log</h2></div>
      <pre id="log"></pre>
    </div>
    <details>
      <summary>Input reference</summary>
      <table>
        <tr><th>load type</th><th>entries</th></tr>
        <tr><td>single</td><td><code>pressure_psi</code>, <code>radius_in</code> (or <code>load_lb</code>)</td></tr>
        <tr><td>dual</td><td><code>pressure_psi</code>, <code>radius_in</code> (or <code>load_lb</code> per tire), <code>spacing_in</code> (center to center)</td></tr>
        <tr><td>footprint</td><td><code>rib_width_in</code>, <code>rib_lengths_in</code> [..], <code>rib_pressures_psi</code> [..], <code>groove_width_in</code>, <code>n_tires</code> (1 wide-base, 2 dual), <code>tire_spacing_in</code></td></tr>
        <tr><td>circles</td><td><code>circles</code>: list of <code>x_in</code>, <code>y_in</code>, <code>radius_in</code>, <code>pressure_psi</code></td></tr>
      </table>
      <p>Layers are listed top to bottom; the last one is the subgrade (no thickness). Layer types: <code>elastic</code> (<code>E_psi</code>),
      <code>universal</code> (<code>k</code>: [k1, k2, k3]), <code>ktheta</code> (<code>k</code>: [k1_psi, k2]), <code>elastic_depth</code> (<code>E_table</code>).
      Units: in, psi, lb, pcf. x is across the lane, y is the direction of travel. Tension is positive.</p>
      <p>Batch runs: add a block such as</p>
      <pre>"sweep": { "AC.thickness_in": [3, 5, 7], "AC.E_psi": [200000, 400000] }</pre>
      <p>Every combination is run. Keys are <code>layer name.entry</code> or <code>load.entry</code>.
      The tensile strain is taken at the bottom of the top layer(s) named AC / HMA / asphalt; to choose another layer add
      <code>"output": {"asphalt_bottom_layer": 2}</code>.</p>
    </details>
  </div>

  <div class="col" id="right">
    <div class="card" id="empty">Paste or open a model, then press <b>Run</b>.<br>Use <b>Examples</b> to start from a working input.</div>
    <div id="results" hidden>
      <div class="col">
        <div class="card pad" id="sweepCard" hidden>
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
            <h2>Batch results (microstrain)</h2><span style="flex:1"></span>
            <button id="copyTable">Copy table</button><button id="csvTable">Download CSV</button>
          </div>
          <div class="tablewrap"><table id="sweepTable"></table></div>
          <div class="meta" style="margin-top:6px">Click a row to see its details below.</div>
        </div>
        <div class="card pad meta" id="caseMeta"></div>
        <div class="tiles" id="tiles"></div>
        <div class="card">
          <div class="pad" style="padding-bottom:4px;display:flex;align-items:center;gap:8px">
            <h2 id="t1"></h2><span style="flex:1"></span><button id="csvProfile">Download profile CSV</button></div>
          <div class="legend"><span><i style="background:var(--s1)"></i>&epsilon;xx (transverse)</span><span><i style="background:var(--s2)"></i>&epsilon;yy (longitudinal)</span></div>
          <div class="chart" id="c1"></div>
        </div>
        <div class="card">
          <div class="pad" style="padding-bottom:4px"><h2 id="t2"></h2></div>
          <div class="chart" id="c2"></div>
        </div>
        <div class="card">
          <div class="pad" style="padding-bottom:4px"><h2>Contact area (plan view)</h2></div>
          <div class="chart" id="c3"></div>
        </div>
      </div>
    </div>
  </div>
</main>
<script>
"use strict";
const $ = id => document.getElementById(id);
const fmt = (v, d=1) => (v === null || v === undefined || !isFinite(v)) ? "–" : Number(v).toLocaleString("en-US", {minimumFractionDigits:d, maximumFractionDigits:d});
const sgn = (v, d=1) => (v > 0 ? "+" : v < 0 ? "−" : "") + fmt(Math.abs(v), d);
let RESULT = null, SEL = 0, EXAMPLES = [], logCount = 0, timer = null;

async function api(path, body){
  const r = await fetch(path, body === undefined ? {} : {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
  return r.json();
}
function setMsg(text, cls){ const m = $("msg"); m.textContent = text || ""; m.className = text ? cls : ""; }
function download(name, text, type){
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], {type: type || "text/plain"}));
  a.download = name; document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}
function pretty(obj){
  // JSON with short numeric lists kept on one line
  let s = JSON.stringify(obj, null, 2);
  return s.replace(/\[\s*((?:-?[\d.eE+]+,\s*)*-?[\d.eE+]+)\s*\]/g, (m, inner) => "[" + inner.replace(/\s+/g, " ") + "]");
}

// ---------------------------------------------------------------- input
api("/api/examples").then(d => {
  EXAMPLES = d.examples; $("ver").textContent = "version " + d.version;
  for (const e of EXAMPLES){ const o = document.createElement("option"); o.value = e.id; o.textContent = e.label; $("examples").appendChild(o); }
  const saved = (() => { try { return localStorage.getItem("nartfem_input"); } catch(e){ return null; } })();
  $("input").value = saved || pretty(EXAMPLES[0].data);
});
$("examples").onchange = e => {
  const ex = EXAMPLES.find(x => x.id === e.target.value);
  if (ex){ $("input").value = pretty(ex.data); setMsg("Loaded example: " + ex.label, "ok"); }
  e.target.value = "";
};
$("file").onchange = e => {
  const f = e.target.files[0]; if (!f) return;
  const rd = new FileReader();
  rd.onload = () => { $("input").value = rd.result; setMsg("Opened " + f.name, "ok"); };
  rd.readAsText(f); e.target.value = "";
};
$("save").onclick = () => {
  let name = "model";
  try { name = (JSON.parse($("input").value).name || "model").replace(/[^\w\-]+/g, "_").slice(0, 60); } catch(e){}
  download(name + ".json", $("input").value, "application/json");
};
$("input").addEventListener("keydown", e => {
  if (e.key === "Tab"){ e.preventDefault(); const t = e.target, s = t.selectionStart; t.setRangeText("  ", s, t.selectionEnd, "end"); }
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") $("run").click();
});

// ---------------------------------------------------------------- run
$("run").onclick = async () => {
  setMsg("");
  try { localStorage.setItem("nartfem_input", $("input").value); } catch(e){}
  $("run").disabled = true;
  let r;
  try { r = await api("/api/run", {text: $("input").value}); }
  catch(e){ r = {ok:false, error:"The application is not responding. Start it again."}; }
  if (!r.ok){ $("run").disabled = false; setMsg(r.error, "err"); return; }
  $("log").textContent = ""; logCount = 0;
  $("run").disabled = true; $("stop").disabled = false;
  $("prog").style.visibility = "visible"; $("prog").firstElementChild.style.width = "2%";
  $("progText").textContent = r.total > 1 ? "0 of " + r.total : "running…";
  timer = setInterval(poll, 350);
};
$("stop").onclick = () => api("/api/stop", {});
async function poll(){
  let s;
  try { s = await api("/api/job?since=" + logCount); } catch(e){ return; }
  if (s.log.length){ const L = $("log"); L.textContent += s.log.join("\n") + "\n"; L.scrollTop = L.scrollHeight; logCount = s.n_log; }
  if (s.total > 1){ $("prog").firstElementChild.style.width = (100 * s.done / s.total) + "%"; $("progText").textContent = s.done + " of " + s.total; }
  if (!s.running){
    clearInterval(timer); timer = null;
    $("run").disabled = false; $("stop").disabled = true; $("prog").style.visibility = "hidden";
    $("progText").textContent = "";
    if (s.error){ setMsg(s.error, "err"); return; }
    if (s.result && s.result.cases.length){
      RESULT = s.result; SEL = 0; render();
      setMsg("Finished " + RESULT.cases.length + (RESULT.cases.length > 1 ? " cases" : " case") + " in " + fmt(s.elapsed, 1) + " s.", "ok");
    }
  }
}
setInterval(() => { fetch("/api/ping", {method:"POST", body:"{}"}).catch(() => {}); }, 20000);
$("quit").onclick = async () => {
  await api("/api/quit", {});
  document.body.innerHTML = '<div style="padding:60px;text-align:center;color:#52514e">The application has been closed. You can close this page.</div>';
};

// ---------------------------------------------------------------- results
const COLS = [["eps_xx_max","εxx max AC bottom"],["eps_yy_max","εyy max AC bottom"],["eps_h_max","max tensile AC bottom"],
              ["eps_zz_subgrade","εzz subgrade top"],["eps_zz_above_interface","εzz above subgrade"]];
function tableRows(){
  const keys = RESULT.sweep_keys;
  const head = keys.concat(COLS.map(c => c[1])).concat(["converged"]);
  const rows = RESULT.rows.map(r => keys.map(k => r[k]).concat(COLS.map(c => Number(r[c[0]].toFixed(2)))).concat([r.converged ? "yes" : "NO"]));
  return [head].concat(rows);
}
function render(){
  $("empty").hidden = true; $("results").hidden = false;
  const multi = RESULT.cases.length > 1;
  $("sweepCard").hidden = !multi;
  if (multi){
    const T = tableRows();
    let h = "<thead><tr>" + T[0].map(x => "<th>" + x + "</th>").join("") + "</tr></thead><tbody>";
    T.slice(1).forEach((r, i) => { h += '<tr data-i="' + i + '"' + (i === SEL ? ' class="sel"' : "") + ">" +
      r.map((x, j) => "<td" + (x === "NO" ? ' class="flag"' : "") + ">" + (typeof x === "number" && j >= RESULT.sweep_keys.length ? fmt(x, 1) : x) + "</td>").join("") + "</tr>"; });
    $("sweepTable").innerHTML = h + "</tbody>";
    $("sweepTable").querySelectorAll("tbody tr").forEach(tr => tr.onclick = () => { SEL = +tr.dataset.i; render(); });
  }
  const r = RESULT.cases[SEL], c = r.critical;
  const lab = Object.entries(r.label || {}).map(([k, v]) => k + " = " + v).join(", ");
  const struct = r.layers.map(L => L.name + (L.thickness_in ? " " + fmt(L.thickness_in, 2).replace(/\.?0+$/, "") + " in" : "")).join(" / ");
  let meta = "<b>" + (r.name || "Model") + "</b>" + (lab ? " &nbsp;·&nbsp; " + lab : "") + "<br>" + struct +
    " &nbsp;·&nbsp; load " + fmt(r.total_load_lb, 0) + " lb on " + r.n_circles + (r.n_circles > 1 ? " circles" : " circle") + "<br>" + r.info.method;
  if (r.nonlinear){
    meta += r.info.converged ? " &nbsp;·&nbsp; converged in " + r.info.iterations + " iterations"
                             : ' &nbsp;·&nbsp; <span class="flag">not converged after ' + r.info.iterations + " iterations (see log)</span>";
    meta += "<br>layer moduli near the load (psi): " + r.info.moduli.map(m => m.name + " " + fmt(m.mean_near, 0)).join(", ");
  }
  $("caseMeta").innerHTML = meta;
  const loc = p => "at x = " + fmt(p.x_in, 2) + ", y = " + fmt(p.y_in, 2) + " in";
  const tile = (k, v, n, main) => '<div class="card tile' + (main ? " main" : "") + '"><div class="k">' + k + '</div><div class="v">' + v + "<small>με</small></div><div class=\"n\">" + n + "</div></div>";
  $("tiles").innerHTML =
    tile("Max tensile strain, asphalt bottom", sgn(c.eps_h_max.value), c.eps_h_direction + " · depth " + fmt(r.z_tensile_in, 2) + " in", true) +
    tile("Vertical strain, top of subgrade", sgn(c.eps_zz_subgrade.value), "compressive · depth " + fmt(r.z_subgrade_in, 2) + " in · " + loc(c.eps_zz_subgrade), true) +
    tile("εxx max (transverse)", sgn(c.eps_xx_max.value), loc(c.eps_xx_max)) +
    tile("εyy max (longitudinal)", sgn(c.eps_yy_max.value), loc(c.eps_yy_max));
  const p = r.profile;
  $("t1").textContent = "Horizontal strain at the asphalt bottom, along x at y = " + fmt(p.y_tensile_in, 2) + " in (microstrain)";
  $("t2").textContent = "Vertical strain at the top of the subgrade, along x at y = " + fmt(p.y_subgrade_in, 2) + " in (microstrain)";
  const xr = xRange(r);
  lineChart($("c1"), p.x_in, [{name:"εxx", color:"var(--s1)", y:p.eps_xx}, {name:"εyy", color:"var(--s2)", y:p.eps_yy}], xr, r.circles);
  lineChart($("c2"), p.x_in, [{name:"εzz", color:"var(--s1)", y:p.eps_zz}], xr, r.circles);
  planView($("c3"), r.circles, xr);
}
function xRange(r){
  let lo = Infinity, hi = -Infinity;
  for (const c of r.circles){ lo = Math.min(lo, c[0] - c[2]); hi = Math.max(hi, c[0] + c[2]); }
  const m = Math.max(2.0 * (hi - lo), 12);
  return [Math.max(r.profile.x_in[0], lo - m), Math.min(r.profile.x_in[r.profile.x_in.length - 1], hi + m)];
}
function niceTicks(lo, hi, n){
  const span = hi - lo || 1, step0 = Math.pow(10, Math.floor(Math.log10(span / n)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * step0).find(s => span / s <= n) || step0 * 10;
  const t = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) t.push(Math.abs(v) < 1e-9 ? 0 : v);
  return t;
}
const W = 860, ML = 62, MR = 16;
function lineChart(el, X, series, xr, circles){
  const H = 250, MT = 10, MB = 34;
  const idx = X.map((x, i) => i).filter(i => X[i] >= xr[0] && X[i] <= xr[1]);
  let lo = 0, hi = 0;
  for (const s of series) for (const i of idx){ lo = Math.min(lo, s.y[i]); hi = Math.max(hi, s.y[i]); }
  const pad = 0.08 * (hi - lo || 1); if (hi > 0) hi += pad; if (lo < 0) lo -= pad;
  const sx = x => ML + (x - xr[0]) / (xr[1] - xr[0]) * (W - ML - MR), sy = y => MT + (hi - y) / (hi - lo) * (H - MT - MB);
  let g = "";
  // tire contact extents (merged along x)
  const ext = contactExtents(circles);
  for (const e of ext) g += '<rect x="' + sx(e[0]) + '" y="' + MT + '" width="' + (sx(e[1]) - sx(e[0])) + '" height="' + (H - MT - MB) + '" fill="#000" opacity=".045"/>';
  for (const t of niceTicks(lo, hi, 6)){
    g += '<line x1="' + ML + '" x2="' + (W - MR) + '" y1="' + sy(t) + '" y2="' + sy(t) + '" stroke="' + (t === 0 ? "#9a988f" : "#ebe9e2") + '" stroke-width="1"/>';
    g += '<text x="' + (ML - 8) + '" y="' + (sy(t) + 4) + '" text-anchor="end" font-size="11.5" fill="#52514e">' + fmt(t, 0) + "</text>";
  }
  for (const t of niceTicks(xr[0], xr[1], 10)) g += '<text x="' + sx(t) + '" y="' + (H - MB + 16) + '" text-anchor="middle" font-size="11.5" fill="#52514e">' + t + "</text>";
  g += '<text x="' + (ML + (W - ML - MR) / 2) + '" y="' + (H - 3) + '" text-anchor="middle" font-size="11.5" fill="#85837c">x, across the lane (in) — shaded: tire contact</text>';
  for (const s of series){
    g += '<path d="' + idx.map((i, k) => (k ? "L" : "M") + sx(X[i]).toFixed(1) + " " + sy(s.y[i]).toFixed(1)).join("") + '" fill="none" stroke="' + s.color + '" stroke-width="2" stroke-linejoin="round"/>';
  }
  g += '<line id="cx" y1="' + MT + '" y2="' + (H - MB) + '" stroke="#85837c" stroke-width="1" visibility="hidden"/>';
  series.forEach((s, k) => g += '<circle class="dot" r="4" fill="' + s.color + '" stroke="#fcfcfb" stroke-width="2" visibility="hidden"/>');
  el.innerHTML = '<svg viewBox="0 0 ' + W + " " + H + '" role="img">' + g + '<rect x="' + ML + '" y="' + MT + '" width="' + (W - ML - MR) + '" height="' + (H - MT - MB) + '" fill="transparent"/></svg><div class="tip"></div>';
  const svg = el.firstElementChild, tip = el.lastElementChild, cx = svg.querySelector("#cx"), dots = svg.querySelectorAll(".dot");
  svg.onmousemove = ev => {
    const b = svg.getBoundingClientRect(), px = (ev.clientX - b.left) / b.width * W;
    const xv = xr[0] + (px - ML) / (W - ML - MR) * (xr[1] - xr[0]);
    if (xv < xr[0] || xv > xr[1]){ svg.onmouseleave(); return; }
    let best = idx[0]; for (const i of idx) if (Math.abs(X[i] - xv) < Math.abs(X[best] - xv)) best = i;
    cx.setAttribute("x1", sx(X[best])); cx.setAttribute("x2", sx(X[best])); cx.setAttribute("visibility", "visible");
    series.forEach((s, k) => { dots[k].setAttribute("cx", sx(X[best])); dots[k].setAttribute("cy", sy(s.y[best])); dots[k].setAttribute("visibility", "visible"); });
    tip.innerHTML = "x = " + fmt(X[best], 2) + " in<br>" + series.map(s => '<i style="background:' + s.color + '"></i>' + s.name + " " + sgn(s.y[best]) + " με").join("<br>");
    tip.style.display = "block";
    const left = sx(X[best]) / W * b.width;
    tip.style.left = (left > b.width * 0.7 ? left - tip.offsetWidth - 12 : left + 12) + "px"; tip.style.top = "14px";
  };
  svg.onmouseleave = () => { cx.setAttribute("visibility", "hidden"); dots.forEach(d => d.setAttribute("visibility", "hidden")); tip.style.display = "none"; };
}
function contactExtents(circles){
  const iv = circles.map(c => [c[0] - c[2], c[0] + c[2]]).sort((a, b) => a[0] - b[0]); const out = [];
  for (const v of iv){ if (out.length && v[0] <= out[out.length - 1][1] + 1e-6) out[out.length - 1][1] = Math.max(out[out.length - 1][1], v[1]); else out.push(v.slice()); }
  return out;
}
function planView(el, circles, xr){
  let ylo = Infinity, yhi = -Infinity, plo = Infinity, phi = -Infinity;
  for (const c of circles){ ylo = Math.min(ylo, c[1] - c[2]); yhi = Math.max(yhi, c[1] + c[2]); plo = Math.min(plo, c[3]); phi = Math.max(phi, c[3]); }
  const k = (W - ML - MR) / (xr[1] - xr[0]);                         // equal scale in x and y
  const half = Math.max(yhi, -ylo) + 1.0, H = Math.max(90, 2 * half * k + 44), cy = (H - 24) / 2;
  const sx = x => ML + (x - xr[0]) * k, sy = y => cy - y * k;
  const steps = ["#cde2fb", "#9ec5f4", "#6da7ec", "#2a78d6", "#184f95"];
  const col = p => phi - plo < 1e-9 ? steps[3] : steps[Math.min(4, Math.floor((p - plo) / (phi - plo) * 5))];
  let g = '<line x1="' + ML + '" x2="' + (W - MR) + '" y1="' + cy + '" y2="' + cy + '" stroke="#ebe9e2"/>';
  for (const t of niceTicks(xr[0], xr[1], 10)) g += '<text x="' + sx(t) + '" y="' + (H - 6) + '" text-anchor="middle" font-size="11.5" fill="#52514e">' + t + "</text>";
  for (const c of circles) g += '<circle cx="' + sx(c[0]).toFixed(1) + '" cy="' + sy(c[1]).toFixed(1) + '" r="' + (c[2] * k).toFixed(1) + '" fill="' + col(c[3]) + '" stroke="#fcfcfb" stroke-width="1"><title>x ' + c[0].toFixed(2) + ", y " + c[1].toFixed(2) + " in · radius " + c[2].toFixed(3) + " in · " + c[3].toFixed(1) + " psi</title></circle>";
  g += '<text x="' + (ML - 8) + '" y="' + (cy + 4) + '" text-anchor="end" font-size="11.5" fill="#52514e">y = 0</text>';
  const note = phi - plo < 1e-9 ? fmt(plo, 1) + " psi" : "contact pressure " + fmt(plo, 0) + " psi (light) to " + fmt(phi, 0) + " psi (dark)";
  g += '<text x="' + (W - MR) + '" y="14" text-anchor="end" font-size="11.5" fill="#85837c">' + note + " · travel direction is vertical</text>";
  el.innerHTML = '<svg viewBox="0 0 ' + W + " " + H.toFixed(0) + '" role="img">' + g + "</svg>";
}
// ---------------------------------------------------------------- export
const csv = rows => rows.map(r => r.map(x => /[",\n]/.test(String(x)) ? '"' + String(x).replace(/"/g, '""') + '"' : x).join(",")).join("\n") + "\n";
$("csvTable").onclick = () => download("nartfem_batch_results.csv", csv(tableRows()), "text/csv");
$("copyTable").onclick = async () => {
  const text = tableRows().map(r => r.join("\t")).join("\n");
  try { await navigator.clipboard.writeText(text); setMsg("Table copied. Paste it into Excel.", "ok"); }
  catch(e){ download("nartfem_batch_results.tsv", text); }
};
$("csvProfile").onclick = () => {
  const r = RESULT.cases[SEL], p = r.profile;
  const rows = [["x_in", "eps_xx_AC_bottom_microstrain (y=" + p.y_tensile_in + ")", "eps_yy_AC_bottom_microstrain (y=" + p.y_tensile_in + ")", "eps_zz_subgrade_top_microstrain (y=" + p.y_subgrade_in + ")"]];
  p.x_in.forEach((x, i) => rows.push([x, p.eps_xx[i], p.eps_yy[i], p.eps_zz[i]]));
  download("nartfem_profile.csv", csv(rows), "text/csv");
};
</script>
</body>
</html>
"""
