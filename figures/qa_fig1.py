"""Verify exported SVG, unchanged cloud, routing and inline player in Chromium."""
from pathlib import Path
import base64, hashlib, json, xml.etree.ElementTree as ET
from playwright.sync_api import sync_playwright
root=Path(__file__).parent
recipe=json.loads((root/'fig1-training.json').read_text())
labels=['Failure patterns','Research notes','Policy edits','Gate outcomes']
def stars(path):
    # Native star coordinates are world-space. Normalize only block translation
    # and generated block IDs; retain every keyframe, timing, radius and opacity.
    import re
    r=json.loads(path.with_suffix('.json').read_text())
    cloud=next(b for b in r['blocks'] if b['type']=='cluster_cloud')
    pos=cloud['position'];out=[]
    for e in ET.parse(path).getroot().iter():
        if e.get('data-role')!='cluster-star':continue
        attrs=dict(e.attrib)
        for axis,key in [('cx','x'),('cy','y')]:attrs[axis]=round(float(attrs[axis])-pos[key],3)
        attrs['fill']=re.sub(r'block_\d+','cloud',attrs['fill'])
        children=[]
        for c in e:
            a=dict(c.attrib);a['id']=re.sub(r'block_\d+','cloud',a['id'])
            if a.get('attributeName') in ('cx','cy'):
                offset=pos['x' if a['attributeName']=='cx' else 'y']
                a['values']=';'.join(f'{float(v)-offset:.3f}' for v in a['values'].split(';'))
            children.append(a)
        out.append((attrs,children))
    return out
current=stars(root/'fig1-training.svg')
# Normalized historical cloud fingerprint; archives are not required.
assert len(current)==360
assert hashlib.sha256(json.dumps(current,sort_keys=True).encode()).hexdigest() == '3b0c8ad99fd1ee6e322ff0fc10a9adf9fc0dbd723da908f27c8914ea45bd5800'
planned={'planned_score','planned_scores'}
assert all(a.get('target') not in planned for a in recipe['animations'] if a['type']=='particle_emitter')
assert not any(c['source'].startswith('gate') and c['destination'].startswith('sim') for c in recipe['connections'])
with sync_playwright() as pw:
    browser=pw.chromium.launch()
    page=browser.new_page(viewport={'width':1600,'height':1800},java_script_enabled=False)
    page.goto((root/'fig1-training.svg').as_uri())
    page.evaluate("document.documentElement.style.cssText='width:1584px;height:auto;background:#080c18'")
    frames=[]; clouds=[]
    for t in (.5,4,8,16,22):
        clocks=page.evaluate("""t=>Array.from(document.querySelectorAll('svg')).map(s=>{s.pauseAnimations();s.setCurrentTime(t);return {paused:s.animationsPaused(),time:s.getCurrentTime()}})""",t)
        assert all(c['paused'] and abs(c['time']-t)<.001 for c in clocks)
        out=root/f'fig1-preview-{t:g}.png';page.locator('svg').first.screenshot(path=str(out));frames.append(str(out))
        clouds.append(page.evaluate("""()=>Array.from(document.querySelectorAll('[data-role="cluster-star"]')).map(p=>[p.cx.animVal.value,p.cy.animVal.value])"""))
    assert all(clouds[0]!=c for c in clouds[1:])
    paths=page.evaluate("""()=>Array.from(document.querySelectorAll('path[data-role="regular-line"]')).map(p=>{let n=250,m=p.getScreenCTM();return {id:p.id,points:Array.from({length:n+1},(_,i)=>{let q=p.getPointAtLength(p.getTotalLength()*i/n);let r=new DOMPoint(q.x,q.y).matrixTransform(m);return [r.x,r.y]})}})""")
    expected=sum(c['type']=='regular_edge' for c in recipe['connections'])
    assert expected>0 and len(paths)==expected,(expected,len(paths))
    def orient(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    crossings=[]; shared=[]
    for i,a in enumerate(paths):
        for b in paths[i+1:]:
            if any(orient(p,q,r)*orient(p,q,s)<-1e-8 and orient(r,s,p)*orient(r,s,q)<-1e-8 for p,q in zip(a['points'],a['points'][1:]) for r,s in zip(b['points'],b['points'][1:])):crossings.append([a['id'],b['id']])
            near=sum(any((p[0]-q[0])**2+(p[1]-q[1])**2<.5**2 for q in b['points'][2:-2]) for p in a['points'][2:-2])
            if near>5:shared.append([a['id'],b['id'],near])
    assert not crossings and not shared,(crossings,shared)
    world_paths=page.evaluate("""()=>{let inverse=document.documentElement.getScreenCTM().inverse();return Array.from(document.querySelectorAll('path[data-role="regular-line"]')).map(p=>{let matrix=inverse.multiply(p.getScreenCTM());return {id:p.id,points:Array.from({length:251},(_,i)=>{let q=p.getPointAtLength(p.getTotalLength()*i/250);let r=new DOMPoint(q.x,q.y).matrixTransform(matrix);return [r.x,r.y]})}})}""")
    block_hits=[]
    for path in world_paths:
        for b in recipe['blocks']:
            x,y=b['position']['x'],b['position']['y'];w,h=b['dimensions']['width'],b['dimensions']['height']
            # The engine docks wires 3 units inside the glass outline.
            if any(x+4<p[0]<x+w-4 and y+4<p[1]<y+h-4 for p in path['points']):block_hits.append([path['id'],b['id']])
    assert not block_hits,block_hits
    fonts=page.evaluate("""names=>Array.from(document.querySelectorAll('text')).filter(e=>names.includes(e.textContent)).map(e=>({text:e.textContent,font:getComputedStyle(e).fontSize,projected:parseFloat(getComputedStyle(e).fontSize)*e.getScreenCTM().a}))""",labels+['Helper','Critic'])
    assert len(fonts)==6 and all(f['font']=='34px' for f in fonts)
    icons=[]
    for t in (.1,.8):
        icons.append(page.evaluate("""t=>{document.querySelectorAll('svg').forEach(s=>{s.pauseAnimations();s.setCurrentTime(t)});return Array.from(document.querySelectorAll('[data-shape="polyline"]')).map(p=>({id:p.id,points:Array.from(p.querySelector('polyline').animatedPoints).map(q=>[q.x,q.y])}))}""",t))
    changed={bid:any(a['points']!=b['points'] for a,b in zip(*icons) if 'ic_'+bid+'_' in a['id']) for bid in ('sim','helper','judge','critic','researcher','proposer','gate')}
    assert all(changed.values())
    # Planned strand paths must have stationary dashes and no particle schedule.
    tree=ET.parse(root/'fig1-training.svg').getroot()
    planned_details=[]
    for i,c in enumerate(recipe['connections']):
        if c['id'] not in planned:continue
        target=f'edge_{i}-path-0'
        node=next(e for e in tree.iter() if e.get('id')==target)
        assert node.get('stroke-dasharray')
        assert not any(a.get('target')==target and a['type'] in ('particle_emitter','path_motion','comet') for a in recipe['animations'])
        planned_details.append({'id':c['id'],'path':target,'dasharray':node.get('stroke-dasharray'),'particles':False})
    page.reload();page.wait_for_timeout(500)
    read="()=>Array.from(document.querySelectorAll('[data-role=cluster-star]')).map(p=>[p.cx.animVal.value,p.cy.animVal.value])"
    first=page.evaluate(read);page.wait_for_timeout(1500);assert first!=page.evaluate(read)
    player=browser.new_page(viewport={'width':1600,'height':1800})
    player.goto((root/'fig1-player.html').as_uri())
    player.wait_for_function("document.querySelector('img').complete && document.querySelector('img').naturalWidth>0")
    src=player.locator('img').get_attribute('src')
    assert base64.b64decode(src.split(',')[1])==(root/'fig1-training.svg').read_bytes()
    player.wait_for_timeout(500)
    player.screenshot(path=str(root/'fig1-player-preview.png'),full_page=True)
    before=player.locator('img').screenshot();player.wait_for_timeout(1400);after=player.locator('img').screenshot();assert before!=after
    player.locator('#replay').click()
    player.wait_for_function("document.querySelector('img').complete && document.querySelector('img').naturalWidth>0")
    assert player.locator('img').get_attribute('src')==src
    again=player.locator('img').screenshot();player.wait_for_timeout(1300);assert again!=player.locator('img').screenshot()
    report={'frames':frames,'script_disabled_svg':True,'nested_clocks_verified':True,'single_paths_sampled':len(paths),'single_wire_crossings':crossings,'single_wire_shared_segments':shared,'planned_paths':planned_details,'title_fonts':fonts,'semantic_icons_changed':changed,'cloud_stars':len(current),'single_wire_block_intrusions':block_hits,'cloud_geometry_and_smil_equal_after_translation_and_id_normalization':True,'cloud_changes_at_all_phases':True,'script_disabled_normal_playback_changes':True,'player_embedded_bytes_equal':True,'player_initial_and_post_replay_motion':True,'svg_sha256':hashlib.sha256((root/'fig1-training.svg').read_bytes()).hexdigest()}
    (root/'fig1-qa.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));browser.close()
