from pathlib import Path
import struct

root=Path(__file__).parent
fixtures=root
def f(x): return struct.unpack('<I',struct.pack('<f',x))[0]
def box_memory(near,far,triangles=False):
    data=[0]*80
    ids=[]
    for i,(lo,hi) in enumerate(zip(near,far)):
        at=(i//4)*32
        kind=([5,0,5,1,5,0,5,1][i] if triangles else 5)
        child=0x100+8*i+kind
        data[at+i%4]=child
        for k in range(3):
            data[at+4+(i%4)*6+k]=f(lo)
            data[at+7+(i%4)*6+k]=f(hi)
        ids.append(child)
    return data,ids

records=[]
def add(name,near,far,order,heuristic=0,triangles=False,wide=True):
    data,ids=box_memory(near,far,triangles)
    desc=[0,0x80000000|(heuristic<<21)|(int(triangles)<<20),0xffffffff,0x81f003ff & ~(0 if wide else 1<<21)]
    inp=[2,f(100),255,0,0,*desc,0,0,0,f(1),f(1),f(1),0,*data]
    out=[*(ids[i] for i in order),0xffffffff,0xffffffff,0,0,0,f(1),f(1),f(1)]
    records.append((name,inp,out))

near=[7,1,5,3,15,9,13,11]
length=[1,4,2,3,20,10,11,5]
far=[a+b for a,b in zip(near,length)]
add('manual_closest',near,far,[1,3,2,0,5,7,6,4])
add('manual_largest_interval',near,far,[4,6,5,7,1,3,2,0],heuristic=1)
add('manual_midpoint',near,far,[1,3,2,0,7,5,6,4],heuristic=2)
add('manual_triangle_priority',near,far,[1,3,5,7,2,0,6,4],triangles=True)
narrow_near=[7,1,5,3,6,0,4,2]
add('manual_narrow_sort',narrow_near,[a+b for a,b in zip(narrow_near,length)],[1,3,2,0,5,7,6,4],wide=False)
add('manual_unclamped_midpoint',[-4,-6,-8,-2,-5,-7,-9,-3],[10,3,12,7,1,5,4,11],[6,4,1,5,2,3,0,7],heuristic=2)

for grow,hit in [(0,False),(255,True)]:
    data=[0]*80
    data[:4]=[0x105,0xffffffff,0xffffffff,0xffffffff]
    data[4:10]=[f(1),f(0),f(0),f(2),f(.99999),f(3)]
    desc=[0,grow<<23,0xffffffff,0x81f003ff]
    inp=[5,f(100),255,0,0,*desc,0,0,0,f(1),f(1),f(1),0,*data]
    out=[0x105 if hit else 0xffffffff,*([0xffffffff]*9),0,0,0,f(1),f(1),f(1)]
    records.append((f'manual_grow_{grow}',inp,out))

for name,inp,out in records: assert len(inp)==96 and len(out)==16,name
(fixtures/'manual-inputs.bin').write_bytes(b''.join(struct.pack('<96I',*inp) for _,inp,_ in records))
(fixtures/'manual-outputs.bin').write_bytes(b''.join(struct.pack('<16I',*out) for _,_,out in records))
(fixtures/'manual-cases.txt').write_text('\n'.join(name for name,_,_ in records)+'\n')
