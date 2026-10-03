/* =========================================================================
   ATLAS 119 — a real-time map of the actual stellar neighborhood.
   Pure WebGL2, zero dependencies. Data: HYG v4.1 (Hipparcos/Yale/Gliese).
   Units: world space is parsecs, equatorial coordinate frame.
   ========================================================================= */
'use strict';

/* ---------------- decode the baked catalog ---------------- */
const N = STARDATA.count;
const raw = atob(STARDATA.b64);
const bytes = new Uint8Array(raw.length);
for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
const F = new Float32Array(bytes.buffer);        // stride 8: x y z vx vy vz absmag ci
const STRIDE = 8;

const META = STARMETA;
const NAMES = META.names;                         // {index: properName}
const nameToIdx = {};
for (const k in NAMES) nameToIdx[NAMES[k].toLowerCase()] = +k;

/* ---------------- tiny vec/mat library ---------------- */
const V = {
  sub: (a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]],
  add: (a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]],
  scale:(a,s)=>[a[0]*s,a[1]*s,a[2]*s],
  len: a=>Math.hypot(a[0],a[1],a[2]),
  norm: a=>{const l=Math.hypot(a[0],a[1],a[2])||1; return [a[0]/l,a[1]/l,a[2]/l];},
  cross:(a,b)=>[a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]],
  dot:(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2],
};
function perspective(fovy, aspect, near, far){
  const f = 1/Math.tan(fovy/2), nf = 1/(near-far);
  return [f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0];
}
function lookAt(eye, fwd, up){
  const z = V.scale(V.norm(fwd),-1);
  const x = V.norm(V.cross(up, z));
  const y = V.cross(z, x);
  return [x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0,
          -V.dot(x,eye),-V.dot(y,eye),-V.dot(z,eye),1];
}
function mul4(a,b){ // a*b, column-major
  const o = new Array(16);
  for (let c=0;c<4;c++) for (let r=0;r<4;r++){
    o[c*4+r] = a[r]*b[c*4] + a[4+r]*b[c*4+1] + a[8+r]*b[c*4+2] + a[12+r]*b[c*4+3];
  }
  return o;
}

/* ---------------- GL setup ---------------- */
const canvas = document.getElementById('gl');
const hud = document.getElementById('hud');
const ctx = hud.getContext('2d');
const gl = canvas.getContext('webgl2', {antialias:true, alpha:false, depth:false});
if (!gl) { document.body.innerHTML = '<p style="padding:40px">WebGL2 required.</p>'; throw 0; }

function makeProgram(vs, fs){
  const compile = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
      throw new Error(gl.getShaderInfoLog(s));
    return s;
  };
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}

/* ---------------- star shader ----------------
   Physically honest: apparent magnitude from absolute magnitude + distance
   to the CAMERA, color from B−V index via Ballesteros temperature + a
   Planckian-locus fit. Flux-ish mapping into point size + intensity. */
const starVS = `#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aVel;
layout(location=2) in float aMag;   // absolute magnitude
layout(location=3) in float aCI;    // B-V color index
uniform mat4 uVP;
uniform vec3 uCam;
uniform float uTime;        // years from present
uniform float uExposure;
uniform float uPx;          // devicePixelRatio
uniform float uAbs;         // 0 = apparent sky, 1 = true-luminosity mode
out vec3 vColor;
out float vInt;

vec3 tempColor(float T){
  T = clamp(T, 1200.0, 40000.0) * 0.01;
  float r, g, b;
  if (T <= 66.0){ r = 1.0; g = clamp(0.39008157*log(T) - 0.63184144, 0.0, 1.0); }
  else { r = clamp(1.29293618*pow(T-60.0,-0.1332047592), 0.0, 1.0);
         g = clamp(1.12989086*pow(T-60.0,-0.0755148492), 0.0, 1.0); }
  if (T >= 66.0) b = 1.0;
  else if (T <= 19.0) b = 0.0;
  else b = clamp(0.54320678*log(T-10.0) - 1.19625408, 0.0, 1.0);
  return vec3(r,g,b);
}

void main(){
  vec3 pos = aPos + aVel * uTime;
  vec4 clip = uVP * vec4(pos, 1.0);
  gl_Position = clip;
  float d = max(distance(pos, uCam), 1e-9);
  float geoMag = aMag + 5.0*(log(d)*0.4342944819 - 1.0);
  // true-luminosity mode: compress the huge absolute-mag range for display.
  // Standing on top of a star (the Sun at a few AU) always wins, so Sol stays visible.
  float absVis = (aMag + 2.0)*0.7 + 4.2;
  float close = smoothstep(-6.0, -16.0, geoMag);
  float appMag = mix(geoMag, absVis, uAbs * (1.0 - close));
  float b = uExposure * pow(10.0, -0.4*appMag);        // m=0 -> b=exposure
  float size = clamp(8.0*pow(max(b, 0.0), 0.25), 1.6, 64.0);
  size += clamp((-12.0 - appMag) * 14.0, 0.0, 200.0);  // the Sun, up close, is a disc
  vInt = min(90.0 * b/(size*size), 6.0);
  // Ballesteros 2012: T from B−V
  float ci = aCI;
  float T = 4600.0*(1.0/(0.92*ci+1.7) + 1.0/(0.92*ci+0.62));
  vColor = pow(tempColor(T), vec3(1.7));               // deepen color saturation
  gl_PointSize = size * uPx;
  if (vInt < 0.002) gl_Position = vec4(2.0,2.0,2.0,0.0);  // cull invisible
}`;
const starFS = `#version 300 es
precision highp float;
in vec3 vColor;
in float vInt;
out vec4 frag;
void main(){
  vec2 p = gl_PointCoord*2.0 - 1.0;
  float r2 = dot(p,p);
  if (r2 > 1.0) discard;
  float halo = exp(-4.0*r2);
  float core = exp(-16.0*r2);
  vec3 c = vColor * vInt * halo + vec3(1.0) * vInt * core * 0.4;
  frag = vec4(c, 1.0);
}`;

/* simple line shader (constellations, grid) — endpoints carry velocity so
   constellation figures deform through time exactly like the stars do */
const lineVS = `#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aVel;
uniform mat4 uVP;
uniform float uTime;
void main(){ gl_Position = uVP * vec4(aPos + aVel*uTime, 1.0); }`;
const lineFS = `#version 300 es
precision highp float;
uniform vec4 uColor;
out vec4 frag;
void main(){ frag = uColor; }`;

const starProg = makeProgram(starVS, starFS);
const lineProg = makeProgram(lineVS, lineFS);
const U = n => gl.getUniformLocation(starProg, n);
const UL = n => gl.getUniformLocation(lineProg, n);

/* star VAO straight from the decoded catalog buffer */
const starVAO = gl.createVertexArray();
gl.bindVertexArray(starVAO);
const starBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, starBuf);
gl.bufferData(gl.ARRAY_BUFFER, F, gl.STATIC_DRAW);
gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0,3,gl.FLOAT,false,32,0);
gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1,3,gl.FLOAT,false,32,12);
gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2,1,gl.FLOAT,false,32,24);
gl.enableVertexAttribArray(3); gl.vertexAttribPointer(3,1,gl.FLOAT,false,32,28);

/* constellation line VAO: duplicated endpoint attributes (pos+vel) */
let conSegs = [];          // flat [ia, ib, ...]
for (const c of META.constellations) for (const s of c.segs) conSegs.push(s[0], s[1]);
const conData = new Float32Array(conSegs.length * 6);
conSegs.forEach((si, i) => {
  conData.set(F.subarray(si*STRIDE, si*STRIDE+6), i*6);
});
const conVAO = gl.createVertexArray();
gl.bindVertexArray(conVAO);
const conBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, conBuf);
gl.bufferData(gl.ARRAY_BUFFER, conData, gl.STATIC_DRAW);
gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0,3,gl.FLOAT,false,24,0);
gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1,3,gl.FLOAT,false,24,12);

/* grid: equatorial rings around Sol (static, zero velocity) */
const gridVerts = [];
for (const R of [1,5,10,25,50,100,250,500,1000]){
  const SEG = 180;
  for (let i=0;i<SEG;i++){
    const a0 = i/SEG*2*Math.PI, a1 = (i+1)/SEG*2*Math.PI;
    gridVerts.push(R*Math.cos(a0), R*Math.sin(a0), 0, 0,0,0,
                   R*Math.cos(a1), R*Math.sin(a1), 0, 0,0,0);
  }
}
const gridData = new Float32Array(gridVerts);
const gridVAO = gl.createVertexArray();
gl.bindVertexArray(gridVAO);
const gridBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, gridBuf);
gl.bufferData(gl.ARRAY_BUFFER, gridData, gl.STATIC_DRAW);
gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0,3,gl.FLOAT,false,24,0);
gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1,3,gl.FLOAT,false,24,12);
gl.bindVertexArray(null);

gl.enable(gl.BLEND);
gl.blendFunc(gl.ONE, gl.ONE);           // additive — light adds, like the sky

/* ---------------- camera & state ---------------- */
const cam = {
  pos: [0.00002, 0.00002, 0.00001],     // ~4 AU from the Sun: Earth's sky
  yaw: 1.4, pitch: 0.1,                 // set properly after load
  speed: 2e-5,                          // pc per second base
};
let timeYears = 0, timePlaying = false;
let showCon = true, showLab = true, showGrid = false;
let absMode = 0, absTarget = 0;        // true-luminosity mode (animated)
let selected = -1;
let fly = null;                         // active fly-to animation
let tour = null;
let exposure = Math.pow(10, 1.4);
let fov = 60;                           // degrees; pinch narrows this

function fwdVec(){
  const cp = Math.cos(cam.pitch);
  return [cp*Math.cos(cam.yaw), cp*Math.sin(cam.yaw), Math.sin(cam.pitch)];
}
function starPos(i, t){
  const o = i*STRIDE;
  return [F[o]+F[o+3]*t, F[o+1]+F[o+4]*t, F[o+2]+F[o+5]*t];
}

/* aim camera at Betelgeuse for the opening shot (Orion overhead) */
{
  const b = nameToIdx['betelgeuse'];
  if (b !== undefined){
    const d = V.norm(V.sub(starPos(b,0), cam.pos));
    cam.yaw = Math.atan2(d[1], d[0]);
    cam.pitch = Math.asin(d[2]);
  }
}

/* ---------------- resize ---------------- */
let W=0, H=0, PX=1;
function resize(){
  PX = Math.min(devicePixelRatio || 1, 2);
  W = canvas.clientWidth; H = canvas.clientHeight;
  canvas.width = W*PX; canvas.height = H*PX;
  hud.width = W*PX; hud.height = H*PX;
  ctx.setTransform(PX,0,0,PX,0,0);
}
addEventListener('resize', resize); resize();

/* ---------------- input ---------------- */
const $ = id => document.getElementById(id);
const TOUCH = matchMedia('(pointer: coarse)').matches || navigator.maxTouchPoints > 0;
if (TOUCH){
  document.body.classList.add('touch');
  document.querySelector('#intro .go').textContent = 'TAP TO BEGIN';
}
const keys = {};
addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' && e.target.type !== 'range') return;
  keys[e.code] = true;
  if (e.code === 'Space') e.preventDefault();
});
addEventListener('keyup', e => keys[e.code] = false);

function lookBy(dx, dy){
  const sens = 0.0028 * (fov / 60);     // zoomed in, the sky pans more slowly
  cam.yaw -= dx * sens;
  cam.pitch = Math.max(-1.55, Math.min(1.55, cam.pitch - dy * sens));
}
function setFov(next){ fov = Math.max(12, Math.min(95, next)); }

/* one finger pans, two fingers pinch-zoom and pan together */
const ptrs = new Map();
let pinch = null, panMoved = 0, lastTap = 0;
canvas.addEventListener('pointerdown', e => {
  if (e.pointerType === 'mouse' && e.button !== 0) return;
  try { canvas.setPointerCapture(e.pointerId); } catch (err) { /* synthetic or already released */ }
  ptrs.set(e.pointerId, {x: e.clientX, y: e.clientY});
  if (ptrs.size === 1) panMoved = 0;
  if (ptrs.size >= 2){
    const [a, b] = [...ptrs.values()];
    pinch = { dist: Math.hypot(a.x-b.x, a.y-b.y) || 1, midX:(a.x+b.x)/2, midY:(a.y+b.y)/2 };
    panMoved = 100;
  }
});
canvas.addEventListener('pointermove', e => {
  if (!ptrs.has(e.pointerId)) return;
  const prev = ptrs.get(e.pointerId);
  ptrs.set(e.pointerId, {x: e.clientX, y: e.clientY});
  if (ptrs.size >= 2){
    const [a, b] = [...ptrs.values()];
    const dist = Math.hypot(a.x-b.x, a.y-b.y) || 1;
    const midX = (a.x+b.x)/2, midY = (a.y+b.y)/2;
    if (!pinch) pinch = { dist, midX, midY };
    setFov(fov * (pinch.dist / dist));
    lookBy(midX - pinch.midX, midY - pinch.midY);
    pinch = { dist, midX, midY };
    fly = null; stopTour();
    return;
  }
  if (ptrs.size === 1){
    const dx = e.clientX - prev.x, dy = e.clientY - prev.y;
    panMoved += Math.abs(dx) + Math.abs(dy);
    lookBy(dx, dy);
    if (panMoved > 4){ fly = null; stopTour(); }
  }
});
function endPointer(e){
  if (!ptrs.has(e.pointerId)) return;
  const wasTap = e.type === 'pointerup' && ptrs.size === 1 && panMoved < 10;
  ptrs.delete(e.pointerId);
  pinch = null;
  if (wasTap){
    const now = performance.now();
    if (now - lastTap < 320){ fov = 60; lastTap = 0; }
    else { lastTap = now; pick(e.clientX, e.clientY); }
  }
  if (ptrs.size === 1) panMoved = 100;
}
canvas.addEventListener('pointerup', endPointer);
canvas.addEventListener('pointercancel', endPointer);
canvas.addEventListener('contextmenu', e => e.preventDefault());
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  if (e.ctrlKey) setFov(fov * Math.exp(e.deltaY * 0.008));   // trackpad pinch
  else {
    cam.speed *= Math.pow(1.25, -Math.sign(e.deltaY));
    cam.speed = Math.max(1e-9, Math.min(cam.speed, 5000));
  }
}, {passive:false});

/* thumb stick: up flies forward, sideways strafes. Touch devices only. */
const stick = { active:false, x:0, y:0 };
const stickEl = $('stick'), knobEl = $('knob');
function placeKnob(dx, dy){ knobEl.style.transform = `translate(${dx}px, ${dy}px)`; }
function stickFrom(e){
  const r = stickEl.getBoundingClientRect();
  const max = r.width / 2;
  let dx = e.clientX - (r.left + r.width/2);
  let dy = e.clientY - (r.top + r.height/2);
  const mag = Math.hypot(dx, dy) || 1;
  if (mag > max){ dx *= max/mag; dy *= max/mag; }
  stick.x = dx / max;
  stick.y = dy / max;
  placeKnob(dx, dy);
}
stickEl.addEventListener('pointerdown', e => {
  if (document.body.classList.contains('arrange')) return;
  stickEl.setPointerCapture(e.pointerId);
  stick.active = true;
  stickFrom(e);
  e.preventDefault();
});
stickEl.addEventListener('pointermove', e => { if (stick.active) stickFrom(e); });
function stickEnd(){ stick.active = false; stick.x = stick.y = 0; placeKnob(0, 0); }
stickEl.addEventListener('pointerup', stickEnd);
stickEl.addEventListener('pointercancel', stickEnd);

/* ---------------- picking ---------------- */
let curVP = null;
function project(p){                 // -> [x,y,visible] in CSS px
  const v = [
    curVP[0]*p[0]+curVP[4]*p[1]+curVP[8]*p[2]+curVP[12],
    curVP[1]*p[0]+curVP[5]*p[1]+curVP[9]*p[2]+curVP[13],
    0,
    curVP[3]*p[0]+curVP[7]*p[1]+curVP[11]*p[2]+curVP[15],
  ];
  if (v[3] <= 0) return null;
  return [(v[0]/v[3]*0.5+0.5)*W, (0.5-v[1]/v[3]*0.5)*H];
}
function pick(mx, my){
  let best = -1, bestScore = Infinity;
  for (let i=0;i<N;i++){
    const p = starPos(i, timeYears);
    const s = project(p);
    if (!s) continue;
    const dx = s[0]-mx, dy = s[1]-my;
    const d2 = dx*dx+dy*dy;
    if (d2 > 900) continue;                       // 30 px radius
    const dist = Math.max(V.len(V.sub(p, cam.pos)), 1e-5);
    const appMag = F[i*STRIDE+6] + 5*(Math.log10(dist)-1);
    const score = d2 + appMag*40;                 // prefer bright
    if (score < bestScore){ bestScore = score; best = i; }
  }
  select(best);
}

/* ---------------- info panel ---------------- */
const LY = 3.26156;
function fmtDist(pc){
  const ly = pc*LY;
  if (ly < 0.01) return (pc*206265).toFixed(0)+' AU';
  if (ly < 100) return ly.toFixed(2)+' ly';
  return Math.round(ly).toLocaleString()+' ly';
}
function starTemp(ci){ return 4600*(1/(0.92*ci+1.7) + 1/(0.92*ci+0.62)); }
function tempRGB(T){
  T = Math.min(Math.max(T,1200),40000)/100;
  let r,g,b;
  if (T<=66){ r=255; g=Math.min(Math.max(99.47*Math.log(T)-161.12,0),255); }
  else { r=Math.min(Math.max(329.7*Math.pow(T-60,-0.1332),0),255);
         g=Math.min(Math.max(288.12*Math.pow(T-60,-0.0755),0),255); }
  if (T>=66) b=255; else if (T<=19) b=0;
  else b=Math.min(Math.max(138.52*Math.log(T-10)-305.04,0),255);
  return [r|0,g|0,b|0];
}
function select(i){
  selected = i;
  const panel = $('info');
  if (i < 0){ panel.style.display='none'; return; }
  panel.style.display = 'block';
  const o = i*STRIDE;
  const absmag = F[o+6], ci = F[o+7];
  const dSol = V.len(starPos(i,0));
  const dCam = V.len(V.sub(starPos(i,timeYears), cam.pos));
  const inf = (META.info[i]||'||').split('|');     // spect | bf | con
  const T = starTemp(ci);
  const [r,g,b] = tempRGB(T);
  const lum = Math.pow(10, (4.85-absmag)/2.5);
  $('iName').textContent = NAMES[i] || (inf[1] || 'Star #'+i);
  $('iDes').textContent = [inf[1], inf[2] && ('in '+inf[2])].filter(Boolean).join(' · ') || 'HYG catalog entry '+i;
  $('iDist').textContent = i===0 ? '—' : fmtDist(dSol);
  $('iDCam').textContent = fmtDist(dCam);
  $('iSpect').textContent = inf[0] || '—';
  $('iTemp').innerHTML = `<span class="swatch" style="background:rgb(${r},${g},${b})"></span>${Math.round(T).toLocaleString()} K`;
  $('iLum').textContent = lum >= 100 ? Math.round(lum).toLocaleString()+' × Sun'
                      : lum >= 0.01 ? lum.toPrecision(3)+' × Sun' : lum.toExponential(1)+' × Sun';
  $('iMag').textContent = i===0 ? 'mag −26.7 (it is the Sun)'
      : 'mag '+(absmag + 5*(Math.log10(Math.max(dSol,1e-5))-1)).toFixed(2);
}
$('btnGoto').onclick = () => { if (selected>=0) flyTo(selected); };

/* ---------------- fly-to ---------------- */
function flyTo(i, dur, after){
  const target = starPos(i, timeYears);
  const toCam = V.sub(cam.pos, target);
  const d = V.len(toCam);
  const stop = i===0 ? 0.00002 : Math.max(0.25, d*0.001);
  const dir = d > 1e-9 ? V.scale(toCam, 1/d) : [0,-1,0];
  const end = V.add(target, V.scale(dir, stop));
  const sYaw = cam.yaw, sPitch = cam.pitch;
  const look = V.norm(V.sub(target, end));
  let eYaw = Math.atan2(look[1], look[0]);
  const ePitch = Math.asin(Math.max(-1,Math.min(1,look[2])));
  while (eYaw - sYaw >  Math.PI) eYaw -= 2*Math.PI;
  while (eYaw - sYaw < -Math.PI) eYaw += 2*Math.PI;
  fly = { t:0, dur: dur || Math.min(6, 1.6 + Math.log10(1+d)*1.4),
          start:[...cam.pos], end, sYaw, sPitch, eYaw, ePitch, after, target:i };
  select(i);
}
function faceToward(i, dur, after){
  const look = V.norm(V.sub(starPos(i,timeYears), cam.pos));
  let eYaw = Math.atan2(look[1], look[0]);
  const ePitch = Math.asin(Math.max(-1,Math.min(1,look[2])));
  const sYaw = cam.yaw;
  while (eYaw - sYaw >  Math.PI) eYaw -= 2*Math.PI;
  while (eYaw - sYaw < -Math.PI) eYaw += 2*Math.PI;
  fly = { t:0, dur: dur||2, start:[...cam.pos], end:[...cam.pos],
          sYaw, sPitch:cam.pitch, eYaw, ePitch, after };
}
const ease = t => t<0.5 ? 4*t*t*t : 1-Math.pow(-2*t+2,3)/2;

/* ---------------- tour ---------------- */
const caption = $('caption');
let capTimer = null;
function say(text, holdMs){
  caption.innerHTML = text;
  caption.style.opacity = 1;
  clearTimeout(capTimer);
  if (holdMs) capTimer = setTimeout(()=>caption.style.opacity=0, holdMs);
}
function stopTour(){
  if (!tour) return;
  tour.cancelled = true; tour = null;
  $('tglTour').classList.remove('on');
  caption.style.opacity = 0;
}
function startTour(){
  stopTour(); fly = null;
  const T = { cancelled:false }; tour = T;
  $('tglTour').classList.add('on');
  const idx = n => nameToIdx[n.toLowerCase()];
  const sleep = ms => new Promise(r=>setTimeout(r,ms));
  const flown = (i,dur) => new Promise(r=>flyTo(i,dur,r));
  const faced = (i,dur) => new Promise(r=>faceToward(i,dur,r));
  (async()=>{
    try{
      cam.pos = [0.00002,0.00002,0.00001];
      await faced(idx('Betelgeuse'), 2);
      if (T.cancelled) return;
      say('This is the night sky — the real one.<br><span style="font-size:13px;color:#9fc3e8">Every point is a real star from the Hipparcos space telescope survey. There is Orion.</span>');
      await sleep(6000); if (T.cancelled) return;
      say('The nearest star system: <b>Alpha Centauri</b>, 4.3 light-years.', 0);
      await flown(idx('Rigil Kentaurus'), 7); if (T.cancelled) return;
      await sleep(1200); if (T.cancelled) return;
      say('Now turn around…', 0);
      await faced(0, 3.5); if (T.cancelled) return;
      select(0);
      say('That yellow point is <b>the Sun</b>.<br><span style="font-size:13px;color:#9fc3e8">Everything that has ever happened to you happened there.</span>');
      await sleep(7000); if (T.cancelled) return;
      say('<b>Sirius</b> — the brightest star in Earth’s sky. A blue-white furnace 25× brighter than the Sun.', 0);
      await flown(idx('Sirius'), 6); if (T.cancelled) return;
      await sleep(4500); if (T.cancelled) return;
      say('<b>The Pleiades</b> — a real cluster of newborn stars, 440 light-years out.', 0);
      await flown(idx('Alcyone'), 7); if (T.cancelled) return;
      await sleep(4500); if (T.cancelled) return;
      say('<b>Betelgeuse</b> — a dying red supergiant. If it sat where the Sun is, it would swallow Jupiter.', 0);
      await flown(idx('Betelgeuse'), 7); if (T.cancelled) return;
      await sleep(5000); if (T.cancelled) return;
      // pull far out
      say('Pulling back…', 0);
      const far = { t:0, dur:9, start:[...cam.pos], end:[300,-1200,500],
                    sYaw:cam.yaw, sPitch:cam.pitch, eYaw:0, ePitch:0, after:null };
      const look = V.norm(V.sub([0,0,0], far.end));
      far.eYaw = Math.atan2(look[1], look[0]); far.ePitch = Math.asin(look[2]);
      while (far.eYaw - far.sYaw >  Math.PI) far.eYaw -= 2*Math.PI;
      while (far.eYaw - far.sYaw < -Math.PI) far.eYaw += 2*Math.PI;
      await new Promise(r=>{ far.after = r; fly = far; }); if (T.cancelled) return;
      say('109,401 suns. Each measured, named, catalogued.<br><span style="font-size:13px;color:#9fc3e8">And this is only our neighborhood — 0.00005% of the Milky Way.</span>');
      await sleep(7000); if (T.cancelled) return;
      absTarget = 1; $('tglAbs').classList.add('on');
      say('Switching to <b>true luminosity</b> — every star drawn by its real power, not its distance.<br><span style="font-size:13px;color:#9fc3e8">This is what our corner of the galaxy actually looks like.</span>');
      await sleep(8000); if (T.cancelled) return;
      say('Drag the <b>time machine</b> below: the constellations are temporary.<br><span style="font-size:13px;color:#9fc3e8">You are free to explore. Click any star.</span>', 9000);
      stopTour();
    } catch(e){ console.error(e); }
  })();
}

/* ---------------- UI wiring ---------------- */
const famous = ['Sol','Rigil Kentaurus','Sirius','Vega','Betelgeuse','Polaris','Antares','Alcyone','Deneb','Altair'];
const chipBox = $('chips');
for (const n of famous){
  const i = nameToIdx[n.toLowerCase()];
  if (i === undefined) continue;
  const b = document.createElement('button');
  b.textContent = n === 'Rigil Kentaurus' ? 'α Centauri' : n === 'Alcyone' ? 'Pleiades' : n;
  b.onclick = () => { stopTour(); flyTo(i); $('nav').classList.remove('open'); };
  chipBox.appendChild(b);
}
{
  const dl = $('starnames');
  const sorted = Object.values(NAMES).sort();
  for (const n of sorted){
    const o = document.createElement('option'); o.value = n; dl.appendChild(o);
  }
}
$('search').addEventListener('change', e => {
  const i = nameToIdx[e.target.value.trim().toLowerCase()];
  if (i !== undefined){ stopTour(); flyTo(i); e.target.blur(); $('nav').classList.remove('open'); }
});
function wireToggle(id, get, set){
  $(id).onclick = () => { set(!get()); $(id).classList.toggle('on', get()); };
}
wireToggle('tglCon', ()=>showCon, v=>showCon=v);
wireToggle('tglLab', ()=>showLab, v=>showLab=v);
wireToggle('tglGrid', ()=>showGrid, v=>showGrid=v);
$('tglAbs').onclick = () => {
  absTarget = absTarget ? 0 : 1;
  $('tglAbs').classList.toggle('on', !!absTarget);
};
$('tglTour').onclick = () => { if (tour) stopTour(); else startTour(); };
$('tglFull').onclick = () => {
  if (!document.fullscreenElement) document.documentElement.requestFullscreen();
  else document.exitFullscreen();
};
document.addEventListener('fullscreenchange', () => {
  const on = !!document.fullscreenElement;
  $('tglFull').classList.toggle('on', on);
  $('tglFull').textContent = on ? 'exit fullscreen' : 'fullscreen';
});
$('exposure').oninput = e => exposure = Math.pow(10, +e.target.value);
$('menuBtn').onclick = () => {
  const open = $('nav').classList.toggle('open');
  $('menuBtn').classList.toggle('on', open);
  $('menuBtn').textContent = open ? 'close' : 'menu';
};
/* Panels stay put until this is on. Then each outlined panel can be dragged. */
let arranging = false;
$('tglMove').onclick = e => {
  e.stopPropagation();
  arranging = !arranging;
  document.body.classList.toggle('arrange', arranging);
  $('tglMove').classList.toggle('on', arranging);
  $('tglMove').textContent = arranging ? 'lock panels' : 'move panels';
};
let panelDrag = null;
function pinPanel(el){
  const r = el.getBoundingClientRect();
  el.style.left = r.left + 'px';
  el.style.top = r.top + 'px';
  el.style.right = 'auto';
  el.style.bottom = 'auto';
  el.style.transform = 'none';
  el.style.width = r.width + 'px';
  el.style.margin = '0';
}
document.addEventListener('pointerdown', e => {
  if (!arranging || e.target.closest('#tglMove')) return;
  const el = e.target.closest('[data-move]');
  if (!el || el.style.display === 'none') return;
  e.preventDefault();
  e.stopPropagation();
  pinPanel(el);
  const r = el.getBoundingClientRect();
  panelDrag = { el, dx: e.clientX - r.left, dy: e.clientY - r.top };
  el.classList.add('dragging');
}, true);
document.addEventListener('pointermove', e => {
  if (!panelDrag) return;
  const el = panelDrag.el;
  const w = el.offsetWidth, h = el.offsetHeight;
  let x = e.clientX - panelDrag.dx;
  let y = e.clientY - panelDrag.dy;
  x = Math.max(-w + 28, Math.min(innerWidth - 28, x));
  y = Math.max(0, Math.min(innerHeight - 28, y));
  el.style.left = x + 'px';
  el.style.top = y + 'px';
}, true);
function endPanelDrag(){
  if (!panelDrag) return;
  panelDrag.el.classList.remove('dragging');
  panelDrag = null;
}
document.addEventListener('pointerup', endPanelDrag, true);
document.addEventListener('pointercancel', endPanelDrag, true);
function bumpSpeed(factor){
  cam.speed = Math.max(1e-9, Math.min(5000, cam.speed * factor));
}
$('spdMinus').onclick = () => bumpSpeed(1/1.8);
$('spdPlus').onclick = () => bumpSpeed(1.8);

const timeSlider = $('time'), timeVal = $('timeval'), playBtn = $('playbtn');
timeSlider.oninput = e => { timeYears = +e.target.value; };
playBtn.onclick = () => { timePlaying = !timePlaying; playBtn.classList.toggle('on', timePlaying);
                          playBtn.innerHTML = timePlaying ? '&#10074;&#10074;' : '&#9654;'; };
function fmtTime(y){
  if (Math.abs(y) < 50) return 'present day';
  const s = y>0 ? '+' : '−';
  return s + Math.abs(Math.round(y)).toLocaleString() + ' years';
}

$('intro').onclick = () => {
  const el = $('intro');
  el.style.opacity = 0;
  setTimeout(()=>el.remove(), 1300);
  startTour();
};

/* ---------------- labels ---------------- */
const namedIdx = Object.keys(NAMES).map(Number);
const conCentroids = META.constellations.map(c => {
  const set = new Set(); c.segs.forEach(s=>{set.add(s[0]);set.add(s[1]);});
  return { name:c.name, stars:[...set] };
});
let conAlpha = 1;
function drawHUD(){
  ctx.clearRect(0,0,W,H);
  if (showLab){
    // star labels: brightest on screen
    const items = [];
    for (const i of namedIdx){
      const p = starPos(i, timeYears);
      const s = project(p);
      if (!s || s[0]<-20||s[0]>W+20||s[1]<-20||s[1]>H+20) continue;
      const d = Math.max(V.len(V.sub(p, cam.pos)), 1e-6);
      const mApp = F[i*STRIDE+6] + 5*(Math.log10(d)-1);
      const m = mApp*(1-absMode) + ((F[i*STRIDE+6]+2)*0.7+4.2)*absMode;
      if (m > 5.2) continue;
      items.push([m, s, NAMES[i], i]);
    }
    items.sort((a,b)=>a[0]-b[0]);
    ctx.font = '11px Segoe UI';
    ctx.textAlign = 'left';
    for (const [m,s,name,i] of items.slice(0,28)){
      const a = Math.min(1, Math.max(0.25, 1-(m+1)/7));
      ctx.fillStyle = `rgba(160,195,230,${a})`;
      ctx.fillText(name, s[0]+8, s[1]-6);
    }
    // constellation names
    if (showCon && conAlpha > 0.03){
      ctx.font = '600 11px Segoe UI';
      ctx.textAlign = 'center';
      ctx.fillStyle = `rgba(100,140,190,${0.55*conAlpha})`;
      for (const c of conCentroids){
        let x=0,y=0,z=0;
        for (const i of c.stars){ const p=starPos(i,timeYears); x+=p[0];y+=p[1];z+=p[2]; }
        const n = c.stars.length;
        const s = project([x/n,y/n,z/n]);
        if (s && s[0]>0&&s[0]<W&&s[1]>0&&s[1]<H) ctx.fillText(c.name.toUpperCase(), s[0], s[1]);
      }
    }
  }
  // selection reticle
  if (selected >= 0){
    const s = project(starPos(selected, timeYears));
    if (s){
      ctx.strokeStyle = 'rgba(130,190,255,0.9)';
      ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.arc(s[0], s[1], 13, 0, 7); ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(s[0]+13,s[1]); ctx.lineTo(s[0]+21,s[1]);
      ctx.moveTo(s[0]-13,s[1]); ctx.lineTo(s[0]-21,s[1]);
      ctx.moveTo(s[0],s[1]+13); ctx.lineTo(s[0],s[1]+21);
      ctx.moveTo(s[0],s[1]-13); ctx.lineTo(s[0],s[1]-21);
      ctx.stroke();
    }
  }
}

/* ---------------- main loop ---------------- */
const C = 9.7156e-9;                    // speed of light in pc/s
let lastT = performance.now(), frames = 0, fpsT = lastT, fps = 0;
function frame(now){
  const dt = Math.min((now-lastT)/1000, 0.1); lastT = now;
  frames++; if (now-fpsT > 800){ fps = frames*1000/(now-fpsT); frames=0; fpsT=now; }

  /* time machine */
  if (timePlaying){
    timeYears += 4000*dt;
    if (timeYears > 100000){ timeYears = 100000; playBtn.onclick(); }
    timeSlider.value = timeYears;
  }
  timeVal.textContent = fmtTime(timeYears);

  /* flight animation */
  if (fly){
    fly.t += dt;
    const k = ease(Math.min(fly.t/fly.dur, 1));
    cam.pos = [
      fly.start[0]+(fly.end[0]-fly.start[0])*k,
      fly.start[1]+(fly.end[1]-fly.start[1])*k,
      fly.start[2]+(fly.end[2]-fly.start[2])*k,
    ];
    cam.yaw = fly.sYaw + (fly.eYaw-fly.sYaw)*k;
    cam.pitch = fly.sPitch + (fly.ePitch-fly.sPitch)*k;
    if (fly.t >= fly.dur){ const after = fly.after; fly = null; if (after) after(); }
  }

  /* manual flight */
  const fwd = fwdVec();
  const right = V.norm(V.cross(fwd, [0,0,1]));
  const up = V.cross(right, fwd);
  let v = [0,0,0], moving = false, thrust = 1;
  const addv = (vec,s)=>{ v = V.add(v, V.scale(vec,s)); moving = true; };
  if (keys.KeyW) addv(fwd,1);
  if (keys.KeyS) addv(fwd,-1);
  if (keys.KeyD) addv(right,1);
  if (keys.KeyA) addv(right,-1);
  if (keys.KeyE || keys.Space) addv(up,1);
  if (keys.KeyQ) addv(up,-1);
  if (stick.active){
    const mag = Math.min(1, Math.hypot(stick.x, stick.y));
    if (mag > 0.08){ addv(fwd, -stick.y); addv(right, stick.x); thrust = mag; }
  }
  if (moving){
    fly = null; stopTour();
    const sp = cam.speed * (keys.ShiftLeft||keys.ShiftRight ? 6 : 1) * thrust;
    cam.pos = V.add(cam.pos, V.scale(V.norm(v), sp*dt*60));
  }

  /* matrices */
  // 1e-7 pc ≈ 0.02 AU, so the Sun is visible from the 4 AU fly-to standoff.
  // No depth buffer, so the huge near/far ratio does not cause z-fighting.
  const proj = perspective(fov*Math.PI/180, W/H, 1e-7, 60000);
  const view = lookAt(cam.pos, fwd, [0,0,1]);
  curVP = mul4(proj, view);

  /* render */
  gl.viewport(0,0,canvas.width,canvas.height);
  gl.clearColor(0.004,0.006,0.012,1);
  gl.clear(gl.COLOR_BUFFER_BIT);

  absMode += (absTarget - absMode) * Math.min(dt*2.5, 1);

  gl.useProgram(starProg);
  gl.uniformMatrix4fv(U('uVP'), false, curVP);
  gl.uniform3fv(U('uCam'), cam.pos);
  gl.uniform1f(U('uTime'), timeYears);
  gl.uniform1f(U('uExposure'), exposure);
  gl.uniform1f(U('uPx'), PX);
  gl.uniform1f(U('uAbs'), absMode);
  gl.bindVertexArray(starVAO);
  gl.drawArrays(gl.POINTS, 0, N);

  /* constellations are an Earth-sky illusion: fade them as you leave home */
  const dSolPc = V.len(cam.pos);
  const conA = Math.max(0, Math.min(1, 1 - (dSolPc-35)/140)) * (1-absMode);
  conAlpha = conA;

  gl.useProgram(lineProg);
  gl.uniformMatrix4fv(UL('uVP'), false, curVP);
  if (showCon && conA > 0.01){
    gl.uniform1f(UL('uTime'), timeYears);
    gl.uniform4f(UL('uColor'), 0.10*conA, 0.22*conA, 0.38*conA, 1);
    gl.bindVertexArray(conVAO);
    gl.drawArrays(gl.LINES, 0, conSegs.length);
  }
  if (showGrid){
    gl.uniform1f(UL('uTime'), 0);
    gl.uniform4f(UL('uColor'), 0.05, 0.10, 0.18, 1);
    gl.bindVertexArray(gridVAO);
    gl.drawArrays(gl.LINES, 0, gridVerts.length/6);
  }
  gl.bindVertexArray(null);

  drawHUD();

  /* readouts */
  const spC = cam.speed/C;
  const zoom = 60 / fov;
  $('zoom').textContent = Math.abs(zoom - 1) > 0.04
    ? (zoom >= 10 ? Math.round(zoom) : zoom.toFixed(1)) + '× zoom' : '';
  $('spd').textContent = spC >= 1e6 ? (spC/1e6).toPrecision(3)+' million × c'
                        : spC >= 1 ? Math.round(spC).toLocaleString()+' × c'
                        : spC.toPrecision(2)+' × c';
  const dSol = V.len(cam.pos)*LY;
  $('dsol').textContent = dSol < 0.001 ? 'at the Sun'
      : dSol < 1 ? Math.round(dSol*63241).toLocaleString()+' AU from home'
      : dSol.toFixed(dSol<100?2:0).toLocaleString()+' light-years from home';
  $('stats').textContent = `${N.toLocaleString()} real stars · ${fps|0} fps`;

  if (selected >= 0 && $('info').style.display !== 'none'){
    const dCam = V.len(V.sub(starPos(selected,timeYears), cam.pos));
    $('iDCam').textContent = fmtDist(dCam);
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

/* console handle for curious minds */
window.ATLAS = { cam, flyTo, select, starPos, get N(){return N;}, get fov(){return fov;},
  setTime: y => { timeYears = y; timeSlider.value = y; },
  setAbs: v => { absTarget = v; $('tglAbs').classList.toggle('on', !!v); } };
