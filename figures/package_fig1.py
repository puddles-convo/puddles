"""Package unchanged engine SVG bytes as an inline, replayable data-URI image."""
from pathlib import Path
import base64, hashlib, json
root=Path(__file__).parent
svg=(root/'fig1-training.svg').read_bytes()
uri='data:image/svg+xml;base64,'+base64.b64encode(svg).decode()
html='''<!doctype html><meta charset="utf-8"><title>Puddles · implementation-grounded architecture</title>
<div style="background:#080c18;border-radius:12px;overflow:hidden"><img id="figure" alt="Puddles: conversation generation, planned Judge integration, implemented policy improvement and shared state" style="display:block;width:100%;height:auto" src="'''+uri+'''"></div>
<button id="replay" type="button">Replay</button>
<details><summary>Implementation scope and reading guide</summary>
<p>Solid paths are implemented data handoffs. Stationary dashed paths mark planned Judge integration; they carry no particles. The current round script reads pre-scored JSON. No automatic next-batch loop is shown.</p>
<p>The input strips explicitly name state dependencies instead of adding intersecting return wires: policy is consumed by Helper, Critic and Proposer; the decision log supplies prior pattern status, prior queries and earlier edits. Critic recalls supported record types, Researcher consumes earlier queries from ResearchNote, and Proposer consumes GateOutcome reasons. Gatekeeper has no recall path.</p>
<p>Gatekeeper is a development stand-in with a manual keep/revert choice, not measured evaluation. Finish round is orchestration code, not an AI. It logs the round, copies available records into searchable memory after logging, and promotes a policy only when kept. Records are also saved locally during agent execution.</p>
<p>The four cloud groups are record types, not separate stores. Their authors are Critic, Researcher, Proposer and the gate. Searchable memory returns IDs resolved through local, hash-checked authoritative records. The cloud preserves its gather/rotate/split animation; it is illustrative, not a measured embedding. Particles indicate direction, not real execution timing or observed throughput. Early exits (no pattern or no evidence) still finish the round without a gate.</p>
<p>Source trace and checks: fig1-architecture.md and fig1-verification.md. Historical revisions and transient QA screenshots are not included in the published figure bundle. Slide-app playback and a human-approved visual baseline are not tested.</p>
</details><script>const image=document.getElementById('figure');const source=image.src;document.getElementById('replay').onclick=()=>{image.src='';requestAnimationFrame(()=>image.src=source);};</script>'''
(root/'fig1-player.html').write_text(html)
(root/'fig1-package.json').write_text(json.dumps({'svg_sha256':hashlib.sha256(svg).hexdigest(),'embedded_bytes_match':base64.b64decode(uri.split(',')[1])==svg},indent=2))
print(root/'fig1-player.html')
