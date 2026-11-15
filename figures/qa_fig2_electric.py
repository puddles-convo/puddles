"""Verify the real standalone SVG electrical-port revision against its archive."""
from pathlib import Path
import hashlib,json
from playwright.sync_api import sync_playwright
from PIL import Image
root=Path(__file__).parent
archive=Path(json.loads((root/'fig2-electric-archive.json').read_text())['archive'])
old=json.loads((archive/'fig2-inference.json').read_text())
new=json.loads((root/'fig2-inference.json').read_text())
originals=[a for a in new['animations'] if not a['id'].startswith('electric_')]
assert originals==old['animations'], 'Existing semantic motion must be identical'
assert {k:v for k,v in new.items() if k!='animations'}=={k:v for k,v in old.items() if k!='animations'}
added=[a for a in new['animations'] if a['id'].startswith('electric_')]
assert not any(a['type'] in ('particle_emitter','path_motion') for a in added)
ports={('audio_in','start'),('audio_in','end'),('audio_out','end'),('local_play','start'),('local_play','end')}
overlays=[a for a in added if a['type']=='overlay']
assert {(a['anchor']['component'],a['anchor']['slot']) for a in overlays}==ports
assert len(overlays)==15
preserved=json.loads((archive/'unrelated-hashes.json').read_text())
assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in preserved.items())
report={'archive':str(archive),'unchanged_connections_blocks_and_semantic_animation':True,'unrelated_files_verified':len(preserved),'ports':sorted(ports),'electric_overlays':len(overlays),'ordinary_particles_added':False,'effect':'Native polyline overlays: green sleeve + flickering green filament + white-hot core; port-local fallback because neon/pulse excludes dashed spaghetti edges.'}
with sync_playwright() as pw:
 browser=pw.chromium.launch()
 page=browser.new_page(viewport={'width':1600,'height':1800},java_script_enabled=False)
 page.goto((root/'fig2-inference.svg').as_uri())
 page.evaluate("document.documentElement.style.cssText='width:1584px;height:auto;background:#080c18'")
 samples=[]; crops=[]
 for t in (.08,.2,.46,.9):
  page.evaluate('t=>document.querySelectorAll("svg").forEach(s=>{s.pauseAnimations();s.setCurrentTime(t)})',t)
  data=page.evaluate('''()=>Array.from(document.querySelectorAll('[data-shape="polyline"]')).filter(p=>p.id.includes('electric_')).map(p=>({id:p.id,opacity:getComputedStyle(p).opacity,points:Array.from(p.querySelector('polyline').animatedPoints).map(q=>[q.x,q.y]),color:p.querySelector('polyline').getAttribute('stroke')}))''')
  assert len(data)==15
  assert sum(p['color']=='#52FFA8' for p in data)==10
  assert sum(p['color']=='#E8FFF3' for p in data)==5
  samples.append({'time':t,'ports':data})
  full=root/f'fig2-electric-phase-{t:g}.png';page.locator('svg').first.screenshot(path=str(full))
  # Three audio/hardware regions only, no modifications to delivered SVG.
  im=Image.open(full);s=im.width/2840
  for label,box in [('input',(210,620,360,760)),('output',(2170,825,2370,945)),('local',(2490,1000,2630,1260))]:
   crop=im.crop(tuple(round(x*s) for x in box)).resize((600,560 if label=='input' else 360 if label=='output' else 1100))
   name=root/f'fig2-electric-{label}-{t:g}.png';crop.save(name);crops.append(str(name))
 for i in range(15):
  assert len({str(s['ports'][i]['points']) for s in samples})>1
  assert len({s['ports'][i]['opacity'] for s in samples})>1
 report.update(script_disabled_svg=True,all_filaments_change_geometry_and_intensity=True,phase_samples=samples,crops=crops)
 page.reload()
 read="""()=>Array.from(document.querySelectorAll('[data-shape="polyline"]')).filter(p=>p.id.includes('electric_')).map(p=>({id:p.id,opacity:getComputedStyle(p).opacity,points:Array.from(p.querySelector('polyline').animatedPoints).map(q=>[q.x,q.y])}))"""
 page.wait_for_timeout(120);early=page.evaluate(read)
 page.wait_for_timeout(270);late=page.evaluate(read)
 assert len(early)==15 and all(a!=b for a,b in zip(early,late))
 report['all_electric_layers_animate_in_normal_script_disabled_playback']=True
 browser.close()
report['passed']=True
(root/'fig2-electric-qa.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ('phase_samples','crops')},indent=2))
