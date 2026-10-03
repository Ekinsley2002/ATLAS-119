"""
Bake HYG v4.1 star catalog + Stellarium constellation data into compact
JS-embeddable binary for Atlas 119.

Output:
  data_stars.js : window.STARDATA = { count, b64 }  (Float32 x,y,z,vx,vy,vz,absmag,ci per star)
  data_meta.js  : window.STARMETA = { names, info, constellations }
"""
import csv, json, struct, base64, math

# ---------- load stars ----------
stars = []          # list of dicts we keep
hip_to_idx = {}     # hip number -> index into kept list

with open('hygdata_v41.csv', encoding='utf-8') as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            dist = float(row['dist'])
        except ValueError:
            continue
        # HYG marks unknown parallax with dist >= 100000 -> drop (fake positions)
        if dist >= 99999:
            continue
        x, y, z = float(row['x']), float(row['y']), float(row['z'])
        vx = float(row['vx'] or 0); vy = float(row['vy'] or 0); vz = float(row['vz'] or 0)
        try:
            absmag = float(row['absmag'])
        except ValueError:
            continue
        try:
            ci = float(row['ci'])
        except ValueError:
            ci = 0.6
        idx = len(stars)
        stars.append(dict(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz,
                          absmag=absmag, ci=ci,
                          proper=row['proper'].strip(),
                          bf=row['bf'].strip(), spect=row['spect'].strip(),
                          con=row['con'].strip(), mag=float(row['mag']),
                          dist=dist, lum=float(row['lum'] or 1)))
        if row['hip']:
            try:
                hip_to_idx[int(float(row['hip']))] = idx
            except ValueError:
                pass

n = len(stars)
print('kept stars:', n)
vmax = max(math.sqrt(s['vx']**2 + s['vy']**2 + s['vz']**2) for s in stars)
print('max speed (pc/yr):', vmax)

# ---------- binary buffer ----------
buf = bytearray()
for s in stars:
    buf += struct.pack('<8f', s['x'], s['y'], s['z'], s['vx'], s['vy'], s['vz'],
                       s['absmag'], s['ci'])
b64 = base64.b64encode(bytes(buf)).decode('ascii')
with open('data_stars.js', 'w') as f:
    f.write('window.STARDATA={count:%d,b64:"%s"};' % (n, b64))
print('data_stars.js bytes:', len(b64) + 40)

# ---------- metadata ----------
# proper names (all), plus info strings for bright / named / constellation stars
names = {}
for i, s in enumerate(stars):
    if s['proper']:
        names[i] = s['proper']

# ---------- constellations ----------
sky = json.load(open('skyculture.json', encoding='utf-8'))
cons = []
used = set()
for c in sky['constellations']:
    segs = []
    for line in c.get('lines', []):
        for a, b in zip(line, line[1:]):
            ia, ib = hip_to_idx.get(a), hip_to_idx.get(b)
            if ia is not None and ib is not None:
                segs.append([ia, ib]); used.add(ia); used.add(ib)
    if not segs:
        continue
    cn = c.get('common_name', {})
    nm = cn.get('native') or cn.get('english') or c['id'].split()[-1]
    cons.append(dict(name=nm, segs=segs))
print('constellations:', len(cons), 'segments:', sum(len(c["segs"]) for c in cons))

# info strings: spect|bf|con for stars that are named, bright, or in constellation lines
info = {}
for i, s in enumerate(stars):
    if s['proper'] or s['mag'] < 6.5 or i in used:
        info[i] = '%s|%s|%s' % (s['spect'], s['bf'], s['con'])

meta = dict(names=names, info=info, constellations=cons)
with open('data_meta.js', 'w') as f:
    f.write('window.STARMETA=' + json.dumps(meta, separators=(',', ':')) + ';')
print('named stars:', len(names), '| info entries:', len(info))

# sanity: famous stars present?
for want in ['Sirius', 'Betelgeuse', 'Vega', 'Polaris', 'Rigil Kentaurus', 'Proxima Centauri', "Barnard's Star", 'Sol']:
    found = [i for i, nm in names.items() if nm == want]
    if found:
        s = stars[found[0]]
        print('%-18s dist %8.2f pc  mag %5.2f' % (want, s['dist'], s['mag']))
    else:
        print('%-18s MISSING' % want)
