"""Actual standalone SVG, native image playback, routes, icons, fixed bank QA."""
from pathlib import Path
import base64,hashlib,json,xml.etree.ElementTree as ET
from playwright.sync_api import sync_playwright
root=Path(__file__).parent
recipe=json.loads((root/'fig2-inference.json').read_text()); ids=json.loads((root/'fig2-icon-ids.json').read_text())
report={}
def save(): (root/'fig2-qa.json').write_text(json.dumps(report,indent=2))
def orient(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
with sync_playwright() as pw:
 browser=pw.chromium.launch()
 page=browser.new_page(viewport={'width':1600,'height':1800},java_script_enabled=False)
 page.goto((root/('fig2-inference'+'.svg')).as_uri())
 page.evaluate("document.documentElement.style.cssText='width:1584px;height:auto;background:#080c18'")
 frames=[]; banks=[]; icons=[]
 for t in (.5,4,16):
  clocks=page.evaluate("""t=>Array.from(document.querySelectorAll('svg')).map(s=>{s.pauseAnimations();s.setCurrentTime(t);return {paused:s.animationsPaused(),time:s.getCurrentTime()}})""",t)
  assert clocks and all(c['paused'] and abs(c['time']-t)<.001 for c in clocks)
  out=root/f'fig2-preview-{t:g}.png';page.locator('svg').first.screenshot(path=str(out));frames.append(str(out))
  banks.append(page.evaluate("""()=>Array.from(document.querySelectorAll('text')).filter(t=>['Situation','People','What helps','Goals','Follow-up','Preferences'].includes(t.textContent)).map(t=>{let b=t.getBoundingClientRect();return {text:t.textContent,box:[b.x,b.y,b.width,b.height],font:getComputedStyle(t).fontSize}})"""))
  assert len(banks[-1])==6, 'Every real NoteKind category must be visible'
  icons.append(page.evaluate("""()=>Array.from(document.querySelectorAll('[data-shape="polyline"]')).map(p=>({id:p.id,points:Array.from(p.querySelector('polyline').animatedPoints).map(q=>[q.x,q.y])}))"""))
 report.update(frames=frames,script_disabled_svg=True,nested_clocks_verified=True,fixed_note_bank_coordinates=(banks[0]==banks[1]==banks[2]),bank_fonts=banks[0]);save()
 # Independent phase sweep avoids sampling two identical periodic/hold states.
 icons=[]
 for t in (.1,.8,1.9,2.4):
  icons.append(page.evaluate("""t=>{document.querySelectorAll('svg').forEach(s=>{s.pauseAnimations();s.setCurrentTime(t)});return Array.from(document.querySelectorAll('[data-shape="polyline"]')).map(p=>({id:p.id,points:Array.from(p.querySelector('polyline').animatedPoints).map(q=>[q.x,q.y])}))}""",t))
 changed={bid:any(a['points']!=b['points'] for later in icons[1:] for a,b in zip(icons[0],later) if 'ic_'+bid+'_' in a['id']) for bid in ids}
 report['icon_phase_samples']=[.1,.8,1.9,2.4]
 report['semantic_icons_changed']=changed
 paths=page.evaluate("""()=>{let inverse=document.documentElement.getScreenCTM().inverse();return Array.from(document.querySelectorAll('path[id^="edge_"][id$="-path-0"]')).map(p=>{let matrix=inverse.multiply(p.getScreenCTM());return {id:p.id,dash:p.getAttribute('stroke-dasharray'),points:Array.from({length:301},(_,i)=>{let q=p.getPointAtLength(p.getTotalLength()*i/300);let r=new DOMPoint(q.x,q.y).matrixTransform(matrix);return [r.x,r.y]})}})}""")
 assert len(paths)==len(recipe['connections'])>0,(len(paths),len(recipe['connections']))
 crossings=[];shared=[];hits=[]
 for i,a in enumerate(paths):
  for b in paths[i+1:]:
   if any(orient(p,q,r)*orient(p,q,s)<-1e-8 and orient(r,s,p)*orient(r,s,q)<-1e-8 for p,q in zip(a['points'],a['points'][1:]) for r,s in zip(b['points'],b['points'][1:])):crossings.append([a['id'],b['id']])
   near=sum(any((p[0]-q[0])**2+(p[1]-q[1])**2<.5**2 for q in b['points'][6:-6]) for p in a['points'][6:-6])
   if near>5:shared.append([a['id'],b['id'],near])
  for b in recipe['blocks']:
   x,y=b['position']['x'],b['position']['y'];w,h=b['dimensions']['width'],b['dimensions']['height']
   if any(x+4<p[0]<x+w-4 and y+4<p[1]<y+h-4 for p in a['points']):hits.append([a['id'],b['id']])
 report.update(paths_sampled=len(paths),wire_crossings=crossings,shared_segments=shared,block_intrusions=hits,all_links_dashed=all(p['dash'] for p in paths));save()
 # Text labels are checked for truncation and icon/title separation at visible phases.
 titles=page.evaluate("""()=>Array.from(document.querySelectorAll('text[data-role="title"]')).map(t=>{let b=t.getBoundingClientRect();return {text:t.textContent,x:b.x,y:b.y,width:b.width,height:b.height,font:getComputedStyle(t).fontSize,projected:parseFloat(getComputedStyle(t).fontSize)*t.getScreenCTM().a}})""")
 report['titles']=titles;report['truncated_titles']=[t['text'] for t in titles if '…' in t['text']]
 texts=page.evaluate("""()=>Array.from(document.querySelectorAll('text')).map(t=>{let b=t.getBoundingClientRect();return {text:t.textContent,x:b.x,y:b.y,r:b.right,b:b.bottom}}).filter(t=>t.r>t.x&&t.b>t.y)""")
 overlaps=[]
 for i,a in enumerate(texts):
  for b in texts[i+1:]:
   if min(a['r'],b['r'])-max(a['x'],b['x'])>1 and min(a['b'],b['b'])-max(a['y'],b['y'])>1:overlaps.append([a['text'],b['text']])
 report['text_box_overlaps']=overlaps
 icon_position=[]
 for bid in ids:
  index=next(i for i,b in enumerate(recipe['blocks']) if b['id']==bid)
  got=page.evaluate("""({index,bid})=>{let block=document.getElementById('block_'+index+'-svg'),t=block.querySelector('text[data-role="title"]').getBoundingClientRect();let ps=Array.from(block.querySelectorAll('[data-shape="polyline"]')).filter(x=>x.id.includes('ic_'+bid+'_')).map(x=>x.getBoundingClientRect());return {left:Math.max(...ps.map(x=>x.right))<t.left,aligned:Math.abs((Math.min(...ps.map(x=>x.top))+Math.max(...ps.map(x=>x.bottom)))/2-(t.top+t.bottom)/2)<18}}""",{'index':index,'bid':bid})
  icon_position.append({'id':bid,**got})
 report['icons_left_aligned']=icon_position
 page.reload();page.wait_for_timeout(500)
 read="()=>Array.from(document.querySelectorAll('[data-shape=polyline] polyline')).map(p=>Array.from(p.animatedPoints).map(q=>[q.x,q.y]))"
 first=page.evaluate(read);page.wait_for_timeout(1400);report['script_disabled_normal_playback']=first!=page.evaluate(read)
 player=browser.new_page(viewport={'width':1600,'height':1800})
 player.goto((root/'fig2-player.html').as_uri());player.wait_for_function("document.querySelector('img').complete && document.querySelector('img').naturalWidth>0")
 src=player.locator('img').get_attribute('src');report['player_embedded_bytes_equal']=base64.b64decode(src.split(',')[1])==(root/'fig2-inference.svg').read_bytes()
 player.wait_for_timeout(500);player.screenshot(path=str(root/'fig2-player-preview.png'),full_page=True)
 before=player.locator('img').screenshot();player.wait_for_timeout(1400);report['player_initial_motion']=before!=player.locator('img').screenshot()
 player.locator('#replay').click();player.wait_for_function("document.querySelector('img').complete && document.querySelector('img').naturalWidth>0")
 replay_bytes=player.evaluate("""async()=>Array.from(new Uint8Array(await (await fetch(document.querySelector('img').src)).arrayBuffer()))""")
 report['replay_bytes_equal']=bytes(replay_bytes)==(root/'fig2-inference.svg').read_bytes()
 again=player.locator('img').screenshot();player.wait_for_timeout(1300);report['player_post_replay_motion']=again!=player.locator('img').screenshot()
 report['svg_sha256']=hashlib.sha256((root/'fig2-inference.svg').read_bytes()).hexdigest()
 old=json.loads((root/'fig2-preservation-before.json').read_text())
 report['prior_figure_files_unchanged']=all((root/p).exists() and hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in old.items())
 report['prior_files_checked']=len(old)
 report['passed']=all([report['fixed_note_bank_coordinates'],all(changed.values()),not crossings,not shared,not hits,not overlaps,report['all_links_dashed'],not report['truncated_titles'],all(x['left'] and x['aligned'] for x in icon_position),report['script_disabled_normal_playback'],report['player_embedded_bytes_equal'],report['replay_bytes_equal'],report['player_initial_motion'],report['player_post_replay_motion'],report['prior_figure_files_unchanged']])
 save();print(json.dumps(report,indent=2));browser.close()
 assert report['passed'], 'Figure 2 custom QA failed; inspect fig2-qa.json'
