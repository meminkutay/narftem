"""Build station_finder.html: a self-contained page (map + nearest-station search) from
MERRA2LIST.csv.  Open the generated file in a browser; no server needed.

    python scripts/make_station_finder.py          (CONFIG block)

Data: HCM_filename, LATITUDE, LONGITUDE, STATE, ELEVATIONm, UTC_standard_time are embedded in the
page.  Map tiles: Esri / USGS (no key).  City lookup: Open-Meteo geocoder (Photon fallback) -
both work from a local file; "lat, lon" entry and map clicks need no lookup.
"""
import csv
import json
import os

# ----------------------------------------------------------------------------- CONFIG
CONFIG = dict(
    CSV="/Users/Kutay/HCM/MERRA2LIST.csv",
    OUT="/Users/Kutay/HCM/station_finder.html",
)
# -------------------------------------------------------------------------------------

HTML = r'''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Climate station finder</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
 body{margin:0;font-family:Calibri,Arial,sans-serif;font-size:14px;display:grid;grid-template-columns:380px 1fr;height:100vh}
 #side{padding:12px;overflow:auto;border-right:1px solid #ddd;background:#fafafa}
 #map{height:100vh}
 h2{margin:0 0 8px;font-size:18px;border-bottom:3px solid #FFCD00;padding-bottom:4px}
 input,button,select{font:inherit;padding:6px} input{width:100%;box-sizing:border-box}
 button{background:#333;color:#fff;border:0;cursor:pointer;margin-top:6px}
 table{border-collapse:collapse;width:100%;margin-top:10px} th,td{border:1px solid #ccc;padding:4px 6px;text-align:left}
 th{background:#e6e6e6} tr.best td{font-weight:bold;background:#fff3b0}
 .small{color:#555;font-size:12px} code{background:#eee;padding:1px 4px}
</style></head><body>
<div id="side">
 <h2>MERRA-2 climate station finder</h2>
 <div class="small">Type a city/state (or an address, or "lat, lon") and press Find, or click anywhere on the map. Nearest stations of <code>MERRA2LIST.csv</code> are listed; the file name is <code>&lt;HCM_filename&gt;.hcm</code> in the HCM folder.</div>
 <input id="q" placeholder="e.g. East Lansing, MI   or   42.74, -84.48" onkeydown="if(event.key==='Enter')find()">
 <button onclick="find()">Find</button>
 <label class="small">Show nearest <select id="n"><option>5</option><option selected>8</option><option>15</option></select> stations</label>
 <div id="msg" class="small"></div>
 <table id="t"><thead><tr><th>#</th><th>file</th><th>lat</th><th>lon</th><th>state</th><th>dist (km)</th></tr></thead><tbody></tbody></table>
</div>
<div id="map"></div>
<script>
const S = DATA;
const map = L.map('map').setView([39.5,-98.5],4);
const esri = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',{maxZoom:19,attribution:'Tiles &copy; Esri'}).addTo(map);
const esriTopo = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',{maxZoom:19,attribution:'Tiles &copy; Esri'});
const usgs = L.tileLayer('https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}',{maxZoom:16,attribution:'USGS The National Map'});
L.control.layers({'Esri Streets':esri,'Esri Topo':esriTopo,'USGS Topo':usgs}).addTo(map);
const all = L.layerGroup().addTo(map);
S.forEach(s=>L.circleMarker([s[1],s[2]],{radius:2,color:'#888',weight:1}).bindTooltip(s[0]+' ('+s[3]+')').addTo(all));
let hits = L.layerGroup().addTo(map), pin=null;
function hav(a,b,c,d){const r=Math.PI/180,R=6371;const x=Math.sin((c-a)*r/2)**2+Math.cos(a*r)*Math.cos(c*r)*Math.sin((d-b)*r/2)**2;return 2*R*Math.asin(Math.sqrt(x));}
function show(lat,lon,label){
 hits.clearLayers(); if(pin) map.removeLayer(pin);
 pin = L.marker([lat,lon]).addTo(map).bindPopup(label||(lat.toFixed(3)+', '+lon.toFixed(3))).openPopup();
 const n=+document.getElementById('n').value;
 const near = S.map(s=>[s,hav(lat,lon,s[1],s[2])]).sort((a,b)=>a[1]-b[1]).slice(0,n);
 const tb=document.querySelector('#t tbody'); tb.innerHTML='';
 near.forEach(([s,d],i)=>{
  const tr=document.createElement('tr'); if(i===0) tr.className='best';
  tr.innerHTML=`<td>${i+1}</td><td>${s[0]}.hcm</td><td>${s[1]}</td><td>${s[2]}</td><td>${s[3]}</td><td>${d.toFixed(1)}</td>`;
  tr.onclick=()=>map.setView([s[1],s[2]],8); tb.appendChild(tr);
  L.circleMarker([s[1],s[2]],{radius:i?6:9,color:i?'#1f5fbf':'#d40000',weight:2,fillOpacity:.5}).bindTooltip(`${s[0]}  ${d.toFixed(1)} km`).addTo(hits);
 });
 document.getElementById('msg').innerHTML=`<b>${label||''}</b> (${lat.toFixed(4)}, ${lon.toFixed(4)}) &rarr; nearest: <b>${near[0][0][0]}.hcm</b>, ${near[0][1].toFixed(1)} km, lat ${near[0][0][1]}, lon ${near[0][0][2]}, UTC ${near[0][0][5]}`;
 map.setView([lat,lon],7);
}
async function find(){
 const q=document.getElementById('q').value.trim(); if(!q) return;
 const m=q.match(/^\s*(-?\d+(\.\d+)?)\s*,\s*(-?\d+(\.\d+)?)\s*$/);
 if(m){ show(+m[1],+m[3],q); return; }
 const msg=document.getElementById('msg'); msg.textContent='Looking up "'+q+'" ...';
 try{
  let name=q.replace(/,\s*[A-Za-z]{2}\s*$/,'');
  const st=(q.match(/,\s*([A-Za-z]{2})\s*$/)||[])[1];
  let r=await fetch('https://geocoding-api.open-meteo.com/v1/search?count=10&language=en&format=json&name='+encodeURIComponent(name));
  let j=await r.json(); let res=(j.results||[]).filter(x=>['US','CA','MX'].includes(x.country_code));
  if(st&&res.length){const f=res.filter(x=>(x.admin1_code||'').toUpperCase()===st.toUpperCase()||(x.admin1||'').toUpperCase().startsWith(st.toUpperCase()));if(f.length)res=f;}
  if(res.length){const x=res[0];show(+x.latitude,+x.longitude,[x.name,x.admin1,x.country_code].filter(Boolean).join(', '));return;}
  r=await fetch('https://photon.komoot.io/api/?limit=1&q='+encodeURIComponent(q));
  j=await r.json();
  if(j.features&&j.features.length){const g=j.features[0];const c=g.geometry.coordinates;const pr=g.properties;show(+c[1],+c[0],[pr.name,pr.state,pr.country].filter(Boolean).join(', '));return;}
  msg.textContent='Not found: '+q;
 }catch(e){msg.textContent='City lookup failed ('+e.message+'). Enter "lat, lon" or click the map.';}
}
map.on('click',e=>show(e.latlng.lat,e.latlng.lng,'map click'));
</script></body></html>'''


def build(csv_path, out_path):
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append([r["HCM_filename"], float(r["LATITUDE"]), float(r["LONGITUDE"]), r.get("STATE", ""),
                             float(r["ELEVATIONm"] or 0), r.get("UTC_standard_time", "")])
            except (ValueError, KeyError):
                pass                                   # rows without coordinates are skipped
    data = json.dumps(rows, separators=(",", ":"))
    with open(out_path, "w") as f:
        f.write(HTML.replace("DATA", data, 1))
    print("%d stations -> %s" % (len(rows), out_path))


if __name__ == "__main__":
    build(CONFIG["CSV"], CONFIG["OUT"])
