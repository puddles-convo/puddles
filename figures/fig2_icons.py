"""Inference-only semantic icons; shared training icon library stays untouched."""
import math
from glass_icons import ICONS, arc, xform, icon_animations

def microphone(u):
    b=math.sin(2*math.pi*u)
    return [arc(0,-9,10,180,360,12)+[(10,4)]+arc(0,4,10,0,180,12)+[(-10,-9)],
            [(-18,0),(-18,7),(-12,20),(0,24),(12,20),(18,7),(18,0)],
            [(0,24),(0,31),(-12,31),(12,31)],
            xform(arc(0,1,27,-35,35,14),s=1+.08*b)]

def listener(u):
    b=math.sin(2*math.pi*u)
    return [[(-25,12),(-25,-12),(-20,-23),(-10,-27),(2,-25),(10,-17),(11,-7),(5,0),(0,4),(-1,13),(-6,21),(-15,21)],
            [(-17,-8),(-12,-14),(-5,-12),(-3,-5),(-10,1)],
            [(16,-12),(30,-12)],[(16,0),(26+4*b,0)],[(16,12),(30,12)]]

def keyword(u):
    x=-20+40*u
    return [[(-25,-21),(25,-21),(25,22),(-25,22),(-25,-21)],
            [(-16,-10),(16,-10),(16,-10)], [(-16,1),(8,1),(8,1)],
            [(x,-27),(x,28)]]

def speaker(u):
    p=1+.12*math.sin(2*math.pi*u)
    return [[(-25,-9),(-14,-9),(1,-23),(1,23),(-14,9),(-25,9),(-25,-9)],
            xform(arc(0,0,15,-55,55,18),s=p),xform(arc(0,0,28,-55,55,18),s=p)]

def robot(u):
    b=3*math.sin(2*math.pi*u)
    return [[(-25,-18),(25,-18),(25,19),(-25,19),(-25,-18)],
            [(-12,-4),(-8,-4),(-8,0),(-12,0),(-12,-4)],[(8,-4),(12,-4),(12,0),(8,0),(8,-4)],
            [(-12,10),(0,13+b),(12,10)],[(0,-18),(0,-29),(-5,-29),(5,-29)]]

def disconnected(u):
    return [arc(0,4,27,220,320,20),arc(0,4,17,220,320,18),arc(0,4,7,220,320,12),
            xform([(-24,25),(25,-25)],dx=1.5*math.sin(2*math.pi*u))]

def notes(u):
    # Fixed original sheets, animated eligibility check (not a moving cloud).
    return [[(-24,-25),(13,-25),(24,-14),(24,25),(-24,25),(-24,-25)],
            [(-14,-12),(7,-12),(7,-12)], [(-14,-1),(13,-1),(13,-1)],
            [(-12,13),(-5,19),(12,7+2*math.sin(2*math.pi*u))]]

def session(u):
    b=2*math.sin(2*math.pi*u)
    return [[(-25,-23),(25,-23),(25,24),(-25,24),(-25,-23)],
            [(-25,-11),(25,-11)], [(-14,-29),(-14,-18)],[(14,-29),(14,-18)],
            [(-12,6),(-3,15),(14,-1+b)]]

def electric_port_animations(edge_id, end, direction, phase=0):
    """Port-local current: green sleeve, bright filament, white-hot core.

    Native neon requires circuit routes; its pulse primitive requires regular
    edges. These planned audio routes are dashed spaghetti. Use supported
    polyline points + opacity tracks instead,
    confined to 28 units at a port. No particles, added wire or cloud-wide current.
    Direction points from the port into the existing connection's gap.
    """
    dx, dy = direction
    def points(jitter):
        # Diamond contact + short changing electric filament, never a new route.
        local = [(0,0),(-4,-4),(0,-8),(4,-4),(0,0),
                 (7,0),(11,3+jitter),(16,-3-jitter),(21,2+jitter),(28,0)]
        return [[round(dx*x-dy*y,2),round(dy*x+dx*y,2)] for x,y in local]
    out=[]
    for layer,color,width,opacity in [('sleeve','#52FFA8',9,.22),
                                       ('current','#52FFA8',3.8,1),
                                       ('core','#E8FFF3',1.2,.95)]:
        oid=f'electric_{edge_id}_{end}_{layer}'
        out.append({'id':oid,'type':'overlay','shape':'polyline','visible':True,
                    'anchor':{'component':edge_id,'slot':end},
                    'attrs':{'points':points(0),'stroke':color,'stroke_width':width,'opacity':opacity}})
        out.append({'id':oid+'_flicker','type':'track','target':oid,'attribute':'opacity',
                    'duration':1.6,'phase':phase,'keyframes':[
                        {'t':t,'value':round(opacity*v,3)} for t,v in
                        [(0,.18),(.12,1),(.2,.35),(.29,.95),(.43,.22),(.56,1),(.7,.5),(.85,.18),(1,.18)]]})
        out.append({'id':oid+'_electric','type':'track','target':oid,'attribute':'points',
                    'duration':1.6,'phase':phase,'keyframes':[
                        {'t':t,'value':' '.join(f'{x},{y}' for x,y in points(j))}
                        for t,j in [(0,0),(.12,2),(.2,-1),(.29,1),(.43,-2),(.56,2),(.7,-1),(.85,0),(1,0)]]})
    return out


ICONS.update(microphone=(microphone,2.4),listener=(listener,2.4),keyword=(keyword,3),speaker=(speaker,1.8),robot=(robot,3),disconnected=(disconnected,4),notes=(notes,3.2),session=(session,4))
