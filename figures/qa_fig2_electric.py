"""Native electrical current: preserve architecture and verify indefinite motion."""
from pathlib import Path
import json, subprocess
from playwright.sync_api import sync_playwright
root=Path(__file__).parent
old=json.loads(subprocess.check_output(['git','show','0355d0c:figures/fig2-inference.json'],cwd=root.parent))
new=json.loads((root/'fig2-inference.json').read_text())
routes={'audio_in','audio_out','local_play'}
assert new['blocks']==old['blocks']
assert len(new['connections'])==len(old['connections'])
for a,b in zip(old['connections'],new['connections']):
 assert (a['id'],a['source'],a['destination'])==(b['id'],b['source'],b['destination'])
 if a['id'] not in routes:assert a==b
 else:assert b['type']=='regular_edge'
def unchanged(animations):
 return {a['id']:a for a in animations if not a['id'].startswith('electric_') and a['id'] not in {e+'_flow' for e in routes}}
assert unchanged(new['animations'])==unchanged(old['animations'])
added=[a for a in new['animations'] if a['id'].startswith('electric_')]
assert len([a for a in added if a.get('shape')=='pulse'])==6
assert not any(a['type'] in ('particle_emitter','path_motion') for a in added)
report={'blocks_and_endpoints_unchanged':True,'other_routes_and_motion_unchanged':True,'native_pulses':6,'ordinary_particles_added':False}
with sync_playwright() as pw:
 browser=pw.chromium.launch()
 page=browser.new_page(java_script_enabled=False)
 page.goto((root/'fig2-inference.svg').as_uri())
 read="""()=>Array.from(document.querySelectorAll('[data-shape="pulse"] path')).map(p=>({id:p.parentElement.id,offset:getComputedStyle(p).strokeDashoffset,d:p.getAttribute('d'),repeat:Array.from(p.querySelectorAll('animate')).map(a=>a.getAttribute('repeatCount'))}))"""
 phases=[]
 for t in (.15,.45,25.15,25.45,49.15,49.45):
  page.evaluate('t=>document.querySelectorAll("svg").forEach(s=>{s.pauseAnimations();s.setCurrentTime(t)})',t)
  phases.append({'time':t,'layers':page.evaluate(read)})
 assert all(len(p['layers'])==18 for p in phases)
 for a,b in zip(phases[::2],phases[1::2]):
  assert all(x['offset']!=y['offset'] for x,y in zip(a['layers'],b['layers']))
 assert all(p['repeat']==['indefinite'] for p in phases[0]['layers'])
 page.reload();page.wait_for_timeout(150);a=page.evaluate(read)
 page.wait_for_timeout(300);b=page.evaluate(read)
 assert all(x['offset']!=y['offset'] for x,y in zip(a,b))
 report.update(script_disabled_normal_playback=True,all_layers_move_at_early_25s_49s=True,repeat_indefinite=True,phases=phases,passed=True)
 browser.close()
(root/'fig2-electric-qa.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='phases'}))
