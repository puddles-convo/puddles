"""Figure 1 v6: implemented handoffs, planned scoring, explicit state dependencies.
Motion is illustrative: independent handoff streams; no claimed execution clock.
Cloud retains v3 seed, star geometry and gather/rotate/split schedule (24 s).
Planned edges are stationary dashed tracks, with no emitters. No next-batch loop.
"""
import json
from pathlib import Path
from glass_icons import icon_animations
OUT = Path(__file__).with_name('fig1-training.json')
LAV, CYAN, PINK, BLUE, PEACH, TEAL, YELLOW, MINT = '#B67DFF','#27DFFF','#FF5FA2','#4A8DFF','#FFAB5C','#2FC4C4','#FFE65C','#52FFA8'
blocks=[]; conn=[]; anims=[]

def card(bid,x,y,title,sub,color,pill=None,icon=False,w=400,h=240):
    b={'id':bid,'type':'glass_block','position':{'x':x,'y':y},'dimensions':{'width':w,'height':h},'styling':{'accent_color':color},'text':{'title':title,'subtitle':sub},'text_style':{'title_size':34,'subtitle_size':25}}
    if icon:
        b.update(icon='waveform',icon_style={'color':'#0e1424','size':1.9,'position':'left'})
    if pill:
        b['slots']=[{'id':bid+'_pills','kind':'chips','font_size':21,'items':[{'text':pill,'color':color}]}]
    blocks.append(b)
    return b

def caption(aid,bid,text,slot='above',size=25,color='#D6E2F5'):
    anims.append({'id':aid,'type':'overlay','anchor':{'component':bid,'slot':slot},'shape':'text','values':[text],'visible':True,'attrs':{'font_size':size,'fill':color,'font_weight':600}})

def edge(eid,src,dst,color,label=None,planned=False,bundle=False,reverse=False,**kw):
    c={'id':eid,'type':'spaghetti_edge' if planned or bundle else 'regular_edge','source':src,'destination':dst,**kw}
    if planned or bundle:c['bundle']={'strand_count':1 if planned else 7,'spread':0 if planned else 30}
    conn.append(c); idx=len(conn)-1
    target=f'edge_{idx}-path-0' if planned or bundle else eid
    if planned:
        anims.append({'id':eid+'_dash','type':'track','target':target,'attribute':'stroke_dashoffset','duration':24,'keyframes':[{'t':0,'value':0},{'t':1,'value':0}]})
    else:
        def emit(suffix,direction,speed,interval):
            anims.append({'id':eid+suffix,'type':'particle_emitter','target':target,'direction':direction,'color':color,'speed':speed,'radius':2.5,'launch_interval':interval,'launch_jitter':interval*.45,'max_particles':26,'seed':100+len(anims)*7,'shooter_interval':1.8 if eid=='copy' else 1.25,'shooter_jitter':.5,'shooter_speed_ratio':4,'shooter_radius_ratio':2})
        emit('_flow','forward',85 if eid=='copy' else 150,.22 if eid=='copy' else .38)
        if reverse:emit('_reply','reverse',135,.55)
    if label:caption(eid+'_label',eid,label,'midpoint',23,color)

card('heading',60,20,'Puddles · one policy-improvement round','Solid: implemented data flow    ·    Dashed: planned scoring integration',MINT,w=2160,h=110)
card('sim',60,200,'Simulated user','scenario-driven dialogue',LAV,'Qwen3.6-35B-A3B',True)
card('helper',640,200,'Helper','input: policy in force',CYAN,'Qwen3.5-9B',True)
card('dialogues',1220,200,'Dialogues','completed runner output',LAV,'conversation records')
card('judge',1800,200,'Judge','scoring integration',PINK,'PLANNED',True)
caption('generation','helper','CONVERSATION GENERATION',size=27,color=CYAN)
edge('conversation','sim.right','helper.left',LAV,reverse=True)
edge('completed','helper.right','dialogues.left',CYAN,bundle=True)
edge('planned_score','dialogues.right','judge.left',PINK,planned=True)
card('scores',1800,570,'Scored dialogues','pre-scored JSON input',BLUE,'ACTUAL ROUND INPUT')
edge('planned_scores','judge.bottom','scores.top',PINK,planned=True)
card('exa',1220,570,'Exa','search queries ↔ results',PEACH,'external search')
card('scope',60,570,'Current execution boundary','The round script loads scored JSON. Judge integration is not implemented. No automatic next-batch loop.',MINT,w=980,h=240)

card('critic',1800,960,'Critic','names one failure pattern',BLUE,'OpenAI · configured',True)
card('researcher',1220,960,'Researcher','researches that pattern',PEACH,'OpenAI · configured',True)
card('proposer',640,960,'Proposer','writes one policy edit',TEAL,'OpenAI · configured',True)
card('gate',60,960,'Gatekeeper','manual keep / revert',YELLOW,'DEVELOPMENT STAND-IN',True)
caption('improvement','researcher','POLICY IMPROVEMENT',size=27,color=PEACH)
edge('scored_input','scores.bottom','critic.top',BLUE,'scored input')
edge('search','researcher.top','exa.bottom',PEACH,reverse=True)
edge('pattern_research','critic.left','researcher.right',BLUE,'pattern')
edge('sources','researcher.left','proposer.right',PEACH,'sources')
edge('candidate','proposer.left','gate.right',TEAL,'candidate + edit')
edge('direct_pattern','critic.bottom','proposer.bottom',BLUE,'failure pattern · direct',route='loopback',lane_offset=90)
# Explicit dependency strips replace long state-return wires. These are inputs, not agents.
card('critic_inputs',1800,1350,'Critic inputs','Memory → supported records\nLog → prior pattern status\nPolicy in force',BLUE,w=400,h=150)
card('researcher_inputs',1220,1350,'Researcher inputs','Memory → prior queries\nLog → previous queries',PEACH,w=400,h=150)
card('proposer_inputs',640,1350,'Proposer inputs','Memory → gate reasons\nLog → earlier edits\nPolicy in force',TEAL,w=400,h=150)
card('finish',60,1570,'Finish round','orchestrator · not an AI',MINT,'log / copy / promote')
edge('outcome','gate.bottom','finish.top',YELLOW,'gate outcome')
card('memory_heading',640,1530,'Shared searchable memory','Four record types · copied only after round logging',MINT,w=1000,h=110)
blocks.append({'id':'memory','type':'cluster_cloud','groups':[{'name':'Failure patterns','color':BLUE},{'name':'Research notes','color':PEACH},{'name':'Policy edits','color':TEAL},{'name':'Gate outcomes','color':YELLOW}],'points_per_group':90,'seed':20261114,'loop':24,'star_size':[.6,2.4],'labels':'none','legend':'corners','position':{'x':640,'y':1680},'dimensions':{'width':1000,'height':460},'styling':{'accent_color':MINT,'corner_radius':18,'glow_strength':.6}})
edge('copy','finish.right','memory.left',MINT)
card('log',1800,1570,'Decision log','+ local record files\nAuthoritative records\nWritten by finish_round',BLUE,'IDs resolve here',h=310)
card('authority',1800,1940,'Recall authority','Memory returns IDs; records are loaded and hash-checked locally.',MINT,h=200)
card('policy',60,1950,'Policy in force','→ Helper · Critic · Proposer',CYAN,'promote only if kept')
edge('promote','finish.bottom','policy.top',CYAN,'kept only')
card('footer',640,2200,'Record authors: Critic · Researcher · Proposer · gate','Input strips name state dependencies. Cloud motion is illustrative, not measured embeddings or execution timing.',MINT,w=1560,h=110)
for bid,icon,col in [('sim','person',LAV),('helper','helper',CYAN),('judge','scales',PINK),('critic','critic',BLUE),('researcher','researcher',PEACH),('proposer','proposer',TEAL),('gate','gatekeeper',YELLOW)]:
    anims += icon_animations(bid,icon,col,-141.3,-51,scale=.8)
for bid in ('memory','finish'):
    anims.append({'id':bid+'_glow','type':'glow_pulse','target':bid,'duration':4,'min_opacity':.45,'max_opacity':.9})
anims.append({'id':'gate_ripple','type':'ripple','target':'gate','rings':2,'duration':4,'scale':1.08,'color':YELLOW})
# Use native list slots for multi-line annotations; subtitles are one-line only.
rows = {
 'heading':['Solid: implemented data flow   ·   Dashed: planned scoring integration'],
 'scope':['Round script: pre-scored JSON → Critic','Judge integration is planned. No automatic next-batch loop.'],
 'critic_inputs':['Memory → supported records','Log → prior pattern status','Policy in force'],
 'researcher_inputs':['Memory → prior queries','Log → previous queries'],
 'proposer_inputs':['Memory → gate reasons','Log → earlier edits','Policy in force'],
 'memory_heading':['Four record types · copied after logging'],
 'log':['+ local record files','Authoritative records','Written by finish_round'],
 'authority':['Memory returns IDs','Load local records','Verify content hashes'],
 'footer':['Input strips name state dependencies. Motion is illustrative, not measured embeddings or timing.']}
for b in blocks:
    bid=b['id']
    if b['type']=='glass_block' and bid not in ('sim','helper','judge','critic','researcher','proposer','gate'):
        b['icon']='none'
    if bid in rows:
        b['text'].pop('subtitle',None)
        b['slots']=[{'id':bid+'_lines','kind':'list','font_size':25 if b['dimensions']['width']>400 else 23,'items':[{'text':s} for s in rows[bid]]}]
    if bid=='heading': b['dimensions']['height']=150
    if bid.endswith('_inputs'): b['dimensions']['height']=220; b['position']['y']=1330
    if bid in ('finish','memory_heading','memory','log','authority','policy','footer'): b['position']['y']+=100
    if bid in ('memory_heading','footer'): b['dimensions']['height']=150
    if bid=='authority':b['dimensions']['height']=230
for b in blocks:
    b['position']['y']+=60
    if b['id'] in ('memory','footer'):b['position']['y']+=40
anims[:]=[a for a in anims if a['id'] not in ('generation','improvement')]
recipe={'schema_version':'1.2','diagram_type':'dataflow','motion_mode':'live','seed':27,'global_particle_limit':700,'canvas':{'width':2280,'height':2610},'blocks':blocks,'connections':conn,'animations':anims}
OUT.write_text(json.dumps(recipe,indent=1))
print(OUT)
