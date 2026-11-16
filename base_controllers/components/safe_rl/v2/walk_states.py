"""
@author: riccardobussola

Run from v2/: python3 walk_states.py <robot>  (uses data/ and models/, the run_v2.sh defaults).
On held-out mps episodes (the same split as train.py), the walking observations 1-10 steps before the
handover after a push, label = the episode's outcome (handing over Δ steps later recovered).
Compare how each value separates them, and where the paper's V sits on them."""
import glob,sys,numpy as np,onnxruntime as ort
sys.argv=['x','--method','mps','--data']+sorted(glob.glob(f'data/{sys.argv[1]}/mps/batch*/'))+['--out','/dev/null']
robot=sys.argv[4].split('/')[1]
src=open('train.py').read().split('episodes = load_episodes(args.data)')[0]
exec(src)
episodes=load_episodes(args.data)
order=rng.permutation(len(episodes)); n_val=max(1,int(0.15*len(episodes))); val=order[:n_val]
def sess(p): return ort.InferenceSession(p,providers=['CPUExecutionProvider'])
M=f'models/VF_%s_{robot}.onnx'
q,pm,pt,ab=sess(M%'mps'),sess(M%'paper_mc'),sess(M%'paper'),sess(M%'abs')
X=[];D=[];Y=[];H=[]
for i in val:
    s=mps_samples(episodes[i])
    if s is None: continue
    x,d,y,w=s
    X.append(x[w]);D.append(d[w]);Y.append(y[w])
    # handover states (first safe obs)
X=np.concatenate(X);D=np.concatenate(D);Y=np.concatenate(Y)
run=lambda s,x:s.run(None,{'obs':x.astype(np.float32)})[0].ravel()
vq=run(q,np.c_[X,D/10.0]); vm=run(pm,X); vt=run(pt,X); va=run(ab,X)
from sklearn.metrics import roc_auc_score as auc
print(f'{robot}: held-out walking states after the push, {len(Y)} ({int((Y==0).sum())} failed)')
for n,v in [('ours Q(o, Δ)',vq),('paper Monte Carlo V(o)',vm),('paper TD(0) V(o)',vt),('ABS -V_RA',va)]:
    print(f'  {n:24s} AUC {auc(Y,v):.3f}  mean on recovered {v[Y==1].mean():.3f}, on failed {v[Y==0].mean():.3f}', 
          f' failed with V>=0.8: {np.mean(v[Y==0]>=0.8):.2f}' if 'paper' in n else '')
