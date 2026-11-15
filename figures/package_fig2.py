"""Package only the unchanged engine SVG, with a tested image-reset Replay."""
from pathlib import Path
import base64,hashlib,json
root=Path(__file__).parent
svg=(root/'fig2-inference.svg').read_bytes()
uri='data:image/svg+xml;base64,'+base64.b64encode(svg).decode()
html='''<!doctype html><meta charset="utf-8"><title>Puddles · inference</title>
<div style="background:#080c18;border-radius:12px;overflow:hidden"><img id="figure" alt="Puddles inference: independent keyword, crisis and helper branches; Release and audio; device-only offline fallback; session-end private notes" style="display:block;width:100%;height:auto" src="'''+uri+'''"></div>
<button id="replay" type="button">Replay</button>
<details><summary>Architecture, implementation status and motion</summary>
<p>Illustrative target flow, not live execution. Every dashed wire is unintegrated on main. MAIN means an existing function, not a working end-to-end robot. BRANCH means code inspected on the named local branch; PLANNED means intended component or integration. Animated copies describe intended direction. They do not report real execution or throughput.</p>
<p>Listener text independently reaches the deterministic keyword check, crisis checker and Helper. The checker also receives recent turns. Helper receives session history, a policy snapshot fixed for the session, and eligible private notes loaded once. There is no Checker-to-Helper edge.</p>
<p>Release is code, not a model. Keyword match or a valid crisis result takes priority and discards the draft. The category is self, danger or unsure; keyword alone uses unsure. A draft is released only with no keyword and a valid none/concern check for the same session and turn. Checker timeout, failure or unreadable output gives the holding line; so does an unavailable draft with a valid noncrisis check. Two consecutive failed checks give offline text unless keyword/crisis priority applies. Late or wrong-turn checks never clear a draft.</p>
<p>Speaker is deliberately labelled audio output: docs discuss TTS and stored crisis/offline recordings, not a verified implementation that always synthesizes crisis text. The independent device-only cloud-unreachable path plays stored offline audio and bypasses cloud STT, Helper and Speaker.</p>
<p>Green electrical current illustrates the three existing audio-signal handoffs: Microphone → Listener, Speaker → Robot playback, and device-only Offline audio → Robot playback. These are native neon pulse primitives with green glow and a white-hot core, not recolored data particles. They loop indefinitely. The routes remain planned and dashed beneath the current; this is not a claim of physical power/network wiring or a working audio integration. Other semantic data routes are unchanged.</p>
<p>The lower panel is session lifecycle, not per-turn processing. Finished text turns, date and remembering consent reach Note-taker, which reads existing notes. Code applies add, replace, confirm and delete. Only live indexed notes can reach Helper; originals stay fixed. The six categories match NoteKind in schemas.py. Remember off permits deletes only. Forget everything blocks adds, replacements and confirms. No private-memory access is granted to Listener, Checker, Release, Speaker or training agents. Training transcript consent is separate.</p>
<p>Audit snapshot: main e5ea2a8; crisis-path 78377e4; notes 23cdbb8. Audio and session wiring are planned. Full engine baseline approval and slide-app playback are not claimed. See fig2-architecture.md and fig2-qa.json.</p>
</details><script>const image=document.getElementById('figure');const source=image.src;const bytes=Uint8Array.from(atob(source.split(',')[1]),c=>c.charCodeAt(0));let previous=null;document.getElementById('replay').onclick=()=>{const next=URL.createObjectURL(new Blob([bytes],{type:'image/svg+xml'}));image.src=next;if(previous)URL.revokeObjectURL(previous);previous=next;};</script>'''
(root/'fig2-player.html').write_text(html)
(root/'fig2-package.json').write_text(json.dumps({'svg_sha256':hashlib.sha256(svg).hexdigest(),'embedded_bytes_match':base64.b64decode(uri.split(',')[1])==svg},indent=2))
print(root/'fig2-player.html')
