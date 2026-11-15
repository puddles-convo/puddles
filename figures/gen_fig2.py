"""Figure 2: target inference architecture, implementation badges are snapshot evidence.
Motion schedule: looping independent role icons (1.8–4 s), illustrative independent
parallel target handoffs (24 s baked schedule); no implied serial checker→helper.
Private notes never move: opacity eligibility highlights only. Copies on read wires.
All connections are unintegrated on main and stay dashed even when animated.
"""
import json
from pathlib import Path
from fig2_icons import icon_animations
OUT=Path(__file__).with_name('fig2-inference.json')
CYAN,BLUE,LAV,CORAL,GOLD,MINT,TEAL='#27DFFF','#4A8DFF','#B67DFF','#FF777D','#FFE65C','#52FFA8','#2FC4C4'
blocks=[]; edges=[]; anim=[]; icon_ids=[]
def card(bid,x,y,title,rows,col,status=None,icon=None,w=440,h=280):
    b={'id':bid,'type':'glass_block','position':{'x':x,'y':y},'dimensions':{'width':w,'height':h},'styling':{'accent_color':col,'glow_strength':.4},'icon':'none','text':{'title':title},'text_style':{'title_size':34,'subtitle_size':25},'slots':[]}
    if icon:
        # Native icon slot sizing follows glass_block._flow_layout; keep its width
        # fixed across the wide note bank and smaller agents, never stretch icons.
        g=min(w/480,h/220)
        icon_size=1.9*(440/480)/g
        b.update(icon='waveform',icon_style={'color':'#0e1424','size':icon_size,'position':'left'})
        cx=-w/2+34*g+38.4*g*icon_size/2
        cy=-h/2+.13*h+34*39.6/26-.41*34
        anim.extend(icon_animations(bid,icon,col,cx,cy,scale=.85));icon_ids.append(bid)
    if rows:b['slots'].append({'id':bid+'_lines','kind':'list','font_size':25,'items':[{'text':r} for r in rows]})
    if status:b['slots'].append({'id':bid+'_state','kind':'chips','font_size':21,'items':[{'text':status,'color':col}]})
    blocks.append(b);return b

def caption(aid,bid,text,slot='above',size=25,col='#D6E2F5'):
    anim.append({'id':aid,'type':'overlay','anchor':{'component':bid,'slot':slot},'shape':'text','values':[text],'visible':True,'attrs':{'font_size':size,'fill':col,'font_weight':600}})

def edge(eid,src,dst,col,label=None,speed=140,interval=.6,reverse=False):
    edges.append({'id':eid,'type':'spaghetti_edge','source':src,'destination':dst,'bundle':{'strand_count':1,'spread':0},'styling':{'source_color':col,'target_color':col,'stroke_width':2.1,'curvature':.65}})
    target=f'edge_{len(edges)-1}-path-0'
    anim.append({'id':eid+'_dash','type':'track','target':target,'attribute':'stroke_dashoffset','duration':24,'keyframes':[{'t':0,'value':0},{'t':1,'value':0}]})
    anim.append({'id':eid+'_flow','type':'particle_emitter','target':target,'direction':'reverse' if reverse else 'forward','color':col,'speed':speed,'radius':2.4,'launch_interval':interval,'launch_jitter':interval*.4,'max_particles':16,'seed':37+len(edges)*17,'shooter_interval':2.5,'shooter_jitter':.6,'shooter_speed_ratio':4,'shooter_radius_ratio':2})
    if label:caption(eid+'_label',eid,label,'midpoint',23,col)

card('heading',60,70,'Puddles · listening, responding, remembering',[
    'TARGET FLOW · dashed links = not integrated on main · MAIN / BRANCH / PLANNED badges show component status',
    'Motion illustrates intended actions and handoffs, not live execution. Parallel checks never feed the Helper.'],MINT,w=2720,h=210)
card('mic',60,360,'Microphone',['Capture the person’s audio'],'#B67DFF','PLANNED · device','microphone')
card('listener',60,740,'Listener / STT',['Transcribe speech to text'],LAV,'PLANNED · audio stack','listener')
card('keyword',620,360,'Keyword check',['Deterministic phrase match','Match forces crisis path'],GOLD,'BRANCH · crisis-path','keyword')
card('checker',620,740,'Crisis checker',['Text + recent turns','Risk · category · status'],CORAL,'BRANCH · crisis-path','critic')
card('helper',620,1120,'Helper',['Text + session history','Policy fixed for session','Eligible notes loaded once'],CYAN,'MAIN · helper function','helper',h=310)
card('release',1180,650,'Release · code',[
    'Keyword / crisis → fixed line','Discard the Helper draft',
    'None / concern → draft',
    'Failed check / draft → hold',
    '2 failed checks → offline'],BLUE,'BRANCH · crisis-path','gatekeeper',w=480,h=460)
card('speaker',1780,740,'Speaker',['Released text → audio','TTS / stored response'],TEAL,'PLANNED · audio stack','speaker')
card('playback',2340,740,'Robot playback',['Play the selected audio'],LAV,'PLANNED · device','robot')
edge('audio_in','mic.bottom','listener.top',LAV,'audio')
edge('text_keyword','listener.right','keyword.left',GOLD)
edge('text_checker','listener.right','checker.left',CORAL)
edge('text_helper','listener.right','helper.left',CYAN)
edge('match','keyword.right','release.top',GOLD,'match')
edge('risk','checker.right','release.left',CORAL,'check')
edge('draft','helper.right','release.bottom',CYAN,'draft / failure')
edge('released','release.right','speaker.left',BLUE,'text')
edge('audio_out','speaker.right','playback.left',TEAL,'audio',speed=210,interval=.35)
caption('parallel','checker','INDEPENDENT INPUTS',size=25)
caption('turn_panel','mic','PER TURN · AUDIO INPUT',size=25,col=LAV)
# Separate local-only route: never routed through cloud Listener, Helper or Speaker.
card('no_cloud',1780,1240,'No cloud',['Device: cloud unreachable'],BLUE,'PLANNED · device only','disconnected')
card('offline',2340,1240,'Offline audio',['Stored local recording'],LAV,'PLANNED · device only','speaker')
caption('device_panel','no_cloud','DEVICE-ONLY FALLBACK',size=25,col=BLUE)
edge('local_trigger','no_cloud.right','offline.left',BLUE,'local')
edge('local_play','offline.top','playback.bottom',LAV,'bypass cloud',speed=210)
# Space below the turn is a separate session-lifecycle panel, not the next turn.
card('memory',620,1720,'Private memory · fixed live-note bank',[
    'Live-note index + private notes · only eligible records are supplied',
    'Same stored notes → Helper at session start; Note-taker at session end'],MINT,'BRANCH · notes · session wiring planned','notes',w=1160,h=480)
# Native fixed category slots: real NoteKind values, no moving originals.
for i,labels in enumerate([['Situation','People','What helps'],['Goals','Follow-up','Preferences']]):
    sid='bank_categories_'+str(i)
    blocks[-1]['slots'].append({'id':sid,'kind':'chips','font_size':34,'items':[{'text':label,'color':MINT} for label in labels]})
    anim.append({'id':sid+'_eligible','type':'track','target':sid,'attribute':'opacity','duration':8,'phase':i*2,'keyframes':[{'t':0,'value':.65},{'t':.25,'value':1},{'t':.65,'value':.65},{'t':1,'value':.65}]})
edge('recall_start','memory.top','helper.bottom',MINT,'once / session',speed=100,interval=1.3)
card('session_end',60,2420,'Session end',['Finished text turns + date','Remember consent','No raw-audio input'],LAV,'PLANNED · lifecycle wiring','session',h=320)
card('note_taker',620,2420,'Note-taker',['Session + existing notes','Propose note actions'],TEAL,'BRANCH · notes','proposer',h=320)
card('apply',1220,2420,'Apply actions · code',['Add · replace · confirm · delete','Update live-note index first'],BLUE,'BRANCH · notes','gatekeeper',w=520,h=320)
caption('session_panel','session_end','AFTER THE SESSION',size=25,col=LAV)
edge('session_notes','session_end.right','note_taker.left',LAV)
edge('existing_notes','memory.bottom','note_taker.top',MINT,'existing notes',speed=95,interval=1.1)
edge('note_actions','note_taker.right','apply.left',TEAL,'actions',speed=100)
edge('apply_memory','apply.top','memory.bottom',BLUE,'write / remove',speed=95,interval=.8)
card('privacy',1900,1740,'Private means private',[
    'Only Helper + Note-taker read notes',
    'No Checker / Listener / Release access',
    'No Speaker access; no training-memory link',
    'Remember off → deletes only',
    'Forget all → no add / replace / confirm',
    'Training-transcript consent is separate'],MINT,w=880,h=400)
card('scope',1900,2420,'Integration boundary',[
    'Audio + session wiring remain planned',
    'crisis-path: concurrent checker + Helper',
    'notes: storage + note-action code',
    'Branch code is not main integration'],BLUE,w=880,h=320)
card('footer',60,2860,'One turn is parallel · memory changes only after the session',[
    'Fixed responses take priority over drafts. Device offline audio is a separate fallback from Release’s offline text.',
    'Snapshot: main e5ea2a8 · crisis-path 78377e4 · notes 23cdbb8 · animation is illustrative target flow.'],MINT,w=2720,h=210)
# Native neon's pulse primitive runs on these same three audio handoffs.
# This is illustrative audio-signal activity, NOT physical power/network wiring.
# Keep every base route dashed: regular edges accept stroke_dashoffset tracks.
# Two offset pulses leave visible current throughout the indefinitely looping cycle.
for eid in ('audio_in','audio_out','local_play'):
    e=next(e for e in edges if e['id']==eid)
    e['type']='regular_edge'
    del e['bundle']
    e['styling']={'source_color':'#18E978','target_color':'#18E978','stroke_width':3.5,'curvature':.65}
    anim[:]=[a for a in anim if a['id']!=eid+'_flow']
    for k in range(2):
        oid=f'electric_{eid}_pulse_{k}'
        anim.append({'id':oid,'type':'overlay','anchor':{'component':eid,'slot':'midpoint'},
                     'shape':'pulse','visible':True,
                     'attrs':{'length':110,'stroke':'#18E978','stroke_width':14,'opacity':1}})
        anim.append({'id':oid+'_run','type':'track','target':oid,'attribute':'progress',
                     'duration':1.4,'phase':k*.7,
                     'keyframes':[{'t':0,'value':0},{'t':1,'value':1}]})
# Draw the unchanged audio labels above the current, never underneath its glow.
for eid in ('audio_in','audio_out','local_play'):
    label=next(a for a in anim if a['id']==eid+'_label')
    anim.remove(label)
    anim.append(label)
blocks_by_id={b['id']:b for b in blocks}
for bid in ('release','memory'):
    blocks_by_id[bid]['styling']['active_outline']=True
    anim.append({'id':bid+'_glow','type':'glow_pulse','target':bid,'duration':4,'min_opacity':.35,'max_opacity':.8})
recipe={'schema_version':'1.2','diagram_type':'dataflow','motion_mode':'live','seed':42,'global_particle_limit':600,'canvas':{'width':2840,'height':3140},'blocks':blocks,'connections':edges,'animations':anim}
OUT.write_text(json.dumps(recipe,indent=1))
OUT.with_name('fig2-icon-ids.json').write_text(json.dumps(icon_ids))
print(OUT)
