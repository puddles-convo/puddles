"""Regression: electrical current must be visible at real GitHub image widths.
Runs the exported SVG script-disabled, not the richer engine HTML. Pixel masks
are confined to existing audio route corridors, excluding unrelated green cards.
"""
from pathlib import Path
import json, sys
from PIL import Image
from playwright.sync_api import sync_playwright
root=Path(__file__).parent
source=Path(sys.argv[1]) if len(sys.argv)>1 else root/'fig2-inference.svg'
out=root/'fig2-published-evidence';out.mkdir(exist_ok=True)
tag=sys.argv[2] if len(sys.argv)>2 else 'after'
regions={'audio_in':(250,640,310,740),'audio_out':(2220,850,2340,910),'local_play':(2530,1020,2590,1240)}
report={'source':str(source),'samples':[],'failures':[]}
with sync_playwright() as pw:
 browser=pw.chromium.launch()
 page=browser.new_page(viewport={'width':1000,'height':1200},java_script_enabled=False)
 for width in (423,838):
  for theme,bg in [('dark','#222a32'),('light','#ffffff')]:
   page.goto(source.resolve().as_uri())
   page.evaluate('(v)=>{document.documentElement.style.cssText=`width:${v.width}px;height:auto;background:${v.bg}`}',{'width':width,'bg':bg})
   samples=[]
   for t in (.15,.45,.8,1.1,1.45,25.15,49.45):
    page.evaluate('t=>document.querySelectorAll("svg").forEach(s=>{s.pauseAnimations();s.setCurrentTime(t)})',t)
    filename=out/f'{tag}-{theme}-{width}-{t}.png'
    page.locator('svg').first.screenshot(path=str(filename))
    im=Image.open(filename).convert('RGB');s=im.width/2840
    stats={}
    for name,box in regions.items():
     crop=im.crop(tuple(round(x*s) for x in box))
     points=[(x,y) for y in range(crop.height) for x in range(crop.width) if (lambda r,g,b:g>125 and g>r*1.3 and g>b*1.12)(*crop.getpixel((x,y)))]
     stats[name]={'green_pixels':len(points),'extent':max((max(p[i] for p in points)-min(p[i] for p in points)+1 for i in (0,1)),default=0) if points else 0}
    samples.append({'time':t,'routes':stats,'file':str(filename)})
   for name in regions:
    visible=[x for x in samples if x['routes'][name]['green_pixels']>=15 and x['routes'][name]['extent']>=10]
    if not visible:report['failures'].append(f'{theme}/{width}/{name}: no unmistakable green current (15 pixels, 10px extent)')
    if not any(x['time']>24 for x in visible):report['failures'].append(f'{theme}/{width}/{name}: no visible current after 24s')
   report['samples'].append({'width':width,'theme':theme,'frames':samples})
 browser.close()
report['passed']=not report['failures']
(out/f'{tag}-visibility.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'passed':report['passed'],'failures':report['failures']}))
assert report['passed'], 'Electricity is not visible at README size'
