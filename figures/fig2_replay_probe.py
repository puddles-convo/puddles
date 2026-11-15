"""Independent Replay phase-reset check on the data-URI image, not engine HTML."""
from pathlib import Path
from playwright.sync_api import sync_playwright
from PIL import Image,ImageChops,ImageStat
import json,io
root=Path(__file__).parent
with sync_playwright() as pw:
 browser=pw.chromium.launch();page=browser.new_page(viewport={'width':1600,'height':1800})
 page.goto((root/'fig2-player.html').as_uri());page.wait_for_function("document.querySelector('img').complete && document.querySelector('img').naturalWidth>0")
 b=page.locator('img').bounding_box();s=b['width']/2840
 crop={'x':b['x']+660*s,'y':b['y']+404*s,'width':56*s,'height':62*s}
 def shot():return Image.open(io.BytesIO(page.screenshot(clip=crop))).convert('RGB')
 page.wait_for_timeout(500);early=shot()
 page.wait_for_timeout(1300);late=shot()
 page.locator('#replay').click();page.wait_for_function("document.querySelector('img').complete && document.querySelector('img').naturalWidth>0")
 page.wait_for_timeout(500);replay=shot()
 def diff(a,b):return sum(ImageStat.Stat(ImageChops.difference(a,b)).mean)/3
 restart=diff(early,replay);phase=diff(early,late)
 result={'early_vs_replay_mean_difference':restart,'early_vs_late_mean_difference':phase,'replay_resets_phase':restart<phase*.7,'method':'Keyword scanner icon crop, initial 0.5s vs late 1.8s vs replay 0.5s; identical clipping; unmodified exported SVG image.'}
 early.save(root/'fig2-replay-early.png');late.save(root/'fig2-replay-late.png');replay.save(root/'fig2-replay-reset.png')
 (root/'fig2-replay-qa.json').write_text(json.dumps(result,indent=2));print(result);browser.close()
 assert result['replay_resets_phase']
