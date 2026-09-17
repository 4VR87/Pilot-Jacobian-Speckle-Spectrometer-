import numpy as np, pandas as pd, os, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.linalg import norm, cond

OUT='/mnt/data/applied_optics_speckle4'
FIG=os.path.join(OUT,'figures'); DATA=os.path.join(OUT,'data')
os.makedirs(FIG,exist_ok=True); os.makedirs(DATA,exist_ok=True)
rng=np.random.default_rng(20260913)

# Physical/reduced-order spectrometer parameters
P=144          # camera pixels used in reconstruction
M=42           # effective guided-mode basis
lam=np.linspace(1546.0,1554.0,81)  # nm, 100-pm grid
N=len(lam); lam0=1550.0
pilot_lams=np.array([1546.5,1553.5])
pilot_idx=[np.argmin(abs(lam-x)) for x in pilot_lams]
L=2.0          # m effective MMF length
core_d=105e-6; NA=0.22
# Environmental coordinate b is dimensionless; b=1 corresponds to approx 0.5 m^-1 curvature perturbation
b_scale_curv=0.5

# Random but deterministic effective modal mixer
A=(rng.normal(size=(P,M))+1j*rng.normal(size=(P,M)))/np.sqrt(2*M)
b0=(rng.normal(size=M)+1j*rng.normal(size=M)); b0=b0/norm(b0)
phi0=rng.uniform(0,2*np.pi,M)
s_lam=rng.normal(0,22.0,M)     # rad/nm; sets spectral decorrelation
q_lam=rng.normal(0,0.8,M)      # rad/nm^2 weak dispersion curvature
alphaT=rng.normal(0,0.22,M)    # rad/degC differential thermo-optic modal phase
alphaB=rng.normal(0,1.15,M)    # rad per bend-state unit
alphaTB=rng.normal(0,0.035,M)  # weak cross term rad/(degC*unit)
# weak environment-induced mixing perturbation matrices
BT=(rng.normal(size=(P,M))+1j*rng.normal(size=(P,M)))/np.sqrt(2*M)*0.015
BB=(rng.normal(size=(P,M))+1j*rng.normal(size=(P,M)))/np.sqrt(2*M)*0.03


def field_intensity(l_nm, dT=0.0, bend=0.0):
    dl=l_nm-lam0
    ph=phi0+s_lam*dl+0.5*q_lam*dl**2+alphaT*dT+alphaB*bend+alphaTB*dT*bend
    v=b0*np.exp(1j*ph)
    Ae=A+dT*BT+bend*BB
    E=Ae@v
    I=np.abs(E)**2
    # camera offset-free normalized fingerprint; preserve relative spatial pattern
    I=I/(np.sum(I)+1e-15)
    return I


def tmatrix(dT=0.0,bend=0.0):
    return np.column_stack([field_intensity(x,dT,bend) for x in lam])

T0=tmatrix()
# finite-difference Jacobian calibration around nominal state
hT=0.05; hB=0.01
JT=(tmatrix(hT,0)-tmatrix(-hT,0))/(2*hT)
JB=(tmatrix(0,hB)-tmatrix(0,-hB))/(2*hB)
# pilot Jacobian stack (2P x 2)
y0pil=np.concatenate([T0[:,i] for i in pilot_idx])
G=np.column_stack([np.concatenate([JT[:,i] for i in pilot_idx]),
                   np.concatenate([JB[:,i] for i in pilot_idx])])
Gcond=cond(G)

# Reconstruction helper
D=np.zeros((N-1,N))
for i in range(N-1): D[i,i]=-1; D[i,i+1]=1

def reconstruct(T,y,alpha=2e-6,beta=2e-7):
    # Fast nonnegative ridge-smooth inverse for repeated Monte Carlo evaluation.
    H=T.T@T + alpha*np.eye(N) + beta*(D.T@D)
    x=np.linalg.solve(H,T.T@y)
    x=np.maximum(x,0)
    if x.max()>0: x/=x.max()
    return x


def noisy(v,snr_db,rng):
    sig=np.sqrt(np.mean(v**2))
    sigma=sig/(10**(snr_db/20))
    return v+rng.normal(0,sigma,v.shape)


def estimate_state(dT,bend,pilot_snr=40.0,rng=rng,jac_err=0.0):
    yt=[]
    for idx in pilot_idx:
        yy=field_intensity(lam[idx],dT,bend)
        yt.append(noisy(yy,pilot_snr,rng))
    yy=np.concatenate(yt)
    Guse=G.copy()
    if jac_err>0:
        Guse=Guse*(1+rng.normal(0,jac_err,Guse.shape))
    q=np.linalg.solve(Guse.T@Guse+1e-10*np.eye(2),Guse.T@(yy-y0pil))
    return q[0],q[1]


def spec_single(center=1550.0,fwhm=0.12):
    s=np.exp(-4*np.log(2)*(lam-center)**2/fwhm**2); return s/s.max()

def spec_two(sep=0.20, center=1550.0, fwhm=0.09):
    s=0.9*np.exp(-4*np.log(2)*(lam-(center-sep/2))**2/fwhm**2)+np.exp(-4*np.log(2)*(lam-(center+sep/2))**2/fwhm**2)
    return s/s.max()

def spec_broad(center=1550.2,fwhm=1.2):
    s=np.exp(-4*np.log(2)*(lam-center)**2/fwhm**2)*(1+0.12*np.cos(2*np.pi*(lam-center)/0.4)); s=np.maximum(s,0); return s/s.max()

def relerr(a,b): return norm(a-b)/(norm(a)+1e-15)
def centroid(s): return np.sum(lam*s)/(np.sum(s)+1e-15)

# --- spectral/thermal correlation functions
Tref=T0
ref_idx=np.argmin(abs(lam-1550))
corr_l=[]
for i,x in enumerate(lam):
    a=T0[:,ref_idx]-T0[:,ref_idx].mean(); b=T0[:,i]-T0[:,i].mean()
    corr_l.append(np.dot(a,b)/(norm(a)*norm(b)+1e-15))

dTs=np.linspace(-4,4,81); corr_T=[]
for t in dTs:
    a=T0[:,ref_idx]-T0[:,ref_idx].mean(); b=field_intensity(1550,t,0)-field_intensity(1550,t,0).mean()
    corr_T.append(np.dot(a,b)/(norm(a)*norm(b)+1e-15))
bends=np.linspace(-0.8,0.8,81); corr_B=[]
for bb in bends:
    a=T0[:,ref_idx]-T0[:,ref_idx].mean(); c=field_intensity(1550,0,bb); b=c-c.mean()
    corr_B.append(np.dot(a,b)/(norm(a)*norm(b)+1e-15))
pd.DataFrame({'wavelength_nm':lam,'corr_to_1550':corr_l}).to_csv(os.path.join(DATA,'spectral_correlation.csv'),index=False)
pd.DataFrame({'dT_C':dTs,'corr':corr_T}).to_csv(os.path.join(DATA,'thermal_correlation.csv'),index=False)
pd.DataFrame({'bend_state':bends,'curvature_equiv_m-1':bends*b_scale_curv,'corr':corr_B}).to_csv(os.path.join(DATA,'bend_correlation.csv'),index=False)

# --- Pilot state estimation vs SNR
snrs=np.arange(20,46,5); est_rows=[]
for snr in snrs:
    esT=[]; esB=[]
    for _ in range(50):
        t=rng.uniform(-1.0,1.0); b=rng.uniform(-0.20,0.20)
        th,bh=estimate_state(t,b,snr,rng)
        esT.append(th-t); esB.append(bh-b)
    est_rows.append([snr,np.sqrt(np.mean(np.square(esT))),np.sqrt(np.mean(np.square(esB)))])
pd.DataFrame(est_rows,columns=['pilot_snr_db','temperature_rmse_C','bend_state_rmse']).to_csv(os.path.join(DATA,'pilot_state_estimation.csv'),index=False)

# --- Reconstruction baselines on combined drifts
# single pilot baseline: infer only an effective temperature-like scalar from first pilot and use JT only
Gp=G[:P,0:1]
y0p=y0pil[:P]

def single_pilot_update(dT,bend,pilot_snr,rng):
    y=noisy(field_intensity(pilot_lams[0],dT,bend),pilot_snr,rng)
    qt=np.linalg.solve(Gp.T@Gp+1e-10*np.eye(1),Gp.T@(y-y0p))[0]
    return T0+JT*qt

# bank baseline: temperature-only bank, assumes bend=0 and nearest temperature
bank_T=np.arange(-1.5,1.51,0.25)
bank=[tmatrix(t,0) for t in bank_T]

def bank_update(dT): return bank[int(np.argmin(abs(bank_T-dT)))]

# representative example
s_ex=spec_two(0.20)
state_ex=(1.0,0.18)
Ttrue=tmatrix(*state_ex); y=noisy(Ttrue@s_ex,35,rng)
th,bh=estimate_state(*state_ex,40,rng)
That=T0+JT*th+JB*bh
rec_stale=reconstruct(T0,y); rec_single=reconstruct(single_pilot_update(*state_ex,40,rng),y); rec_prop=reconstruct(That,y); rec_oracle=reconstruct(Ttrue,y)
pd.DataFrame({'wavelength_nm':lam,'true':s_ex,'stale':rec_stale,'single_pilot':rec_single,'proposed':rec_prop,'oracle':rec_oracle}).to_csv(os.path.join(DATA,'reconstruction_example.csv'),index=False)

# sweeps of temperature and bend errors averaged over 3 spectrum types and 12 noise repeats
specs=[spec_single(),spec_two(0.20),spec_broad()]
temps=np.linspace(-1.5,1.5,13); bend_fixed=0.15
rows=[]
for t in temps:
    errs=[[],[],[],[],[]]
    for rep in range(3):
        for s in specs:
            Tt=tmatrix(t,bend_fixed); y=noisy(Tt@s,35,rng)
            th,bh=estimate_state(t,bend_fixed,40,rng)
            Tup=T0+JT*th+JB*bh
            mats=[T0,single_pilot_update(t,bend_fixed,40,rng),bank_update(t),Tup,Tt]
            for k,mat in enumerate(mats): errs[k].append(relerr(s,reconstruct(mat,y)))
    rows.append([t]+[np.mean(e) for e in errs])
pd.DataFrame(rows,columns=['dT_C','stale','single_pilot','temperature_bank','proposed_dual_pilot','oracle']).to_csv(os.path.join(DATA,'error_vs_temperature.csv'),index=False)

bgrid=np.linspace(-0.30,0.30,13); temp_fixed=0.8
rows=[]
for b in bgrid:
    errs=[[],[],[],[],[]]
    for rep in range(3):
        for s in specs:
            Tt=tmatrix(temp_fixed,b); y=noisy(Tt@s,35,rng)
            th,bh=estimate_state(temp_fixed,b,40,rng)
            Tup=T0+JT*th+JB*bh
            mats=[T0,single_pilot_update(temp_fixed,b,40,rng),bank_update(temp_fixed),Tup,Tt]
            for k,mat in enumerate(mats): errs[k].append(relerr(s,reconstruct(mat,y)))
    rows.append([b,b*b_scale_curv]+[np.mean(e) for e in errs])
pd.DataFrame(rows,columns=['bend_state','curvature_equiv_m-1','stale','single_pilot','temperature_bank','proposed_dual_pilot','oracle']).to_csv(os.path.join(DATA,'error_vs_bend.csv'),index=False)

# 2D heatmaps proposed vs stale
Tgrid=np.linspace(-1.5,1.5,13); Bgrid=np.linspace(-0.3,0.3,13)
heat_prop=np.zeros((len(Tgrid),len(Bgrid))); heat_stale=np.zeros_like(heat_prop)
s=spec_two(0.20)
for it,t in enumerate(Tgrid):
    for ib,b in enumerate(Bgrid):
        eeP=[]; eeS=[]
        for rep in range(2):
            Tt=tmatrix(t,b); y=noisy(Tt@s,35,rng)
            th,bh=estimate_state(t,b,40,rng); Tup=T0+JT*th+JB*bh
            eeP.append(relerr(s,reconstruct(Tup,y))); eeS.append(relerr(s,reconstruct(T0,y)))
        heat_prop[it,ib]=np.mean(eeP); heat_stale[it,ib]=np.mean(eeS)
np.savetxt(os.path.join(DATA,'heatmap_proposed.csv'),heat_prop,delimiter=',')
np.savetxt(os.path.join(DATA,'heatmap_stale.csv'),heat_stale,delimiter=',')

# pilot separation conditioning: pairs symmetric around 1550
seps=np.arange(1.0,7.1,0.5); sep_rows=[]
for sep in seps:
    ids=[np.argmin(abs(lam-(1550-sep/2))),np.argmin(abs(lam-(1550+sep/2)))]
    Gs=np.column_stack([np.concatenate([JT[:,i] for i in ids]),np.concatenate([JB[:,i] for i in ids])])
    sep_rows.append([sep,cond(Gs),norm(Gs[:,0]),norm(Gs[:,1])])
pd.DataFrame(sep_rows,columns=['pilot_separation_nm','condition_number','thermal_jacobian_norm','bend_jacobian_norm']).to_csv(os.path.join(DATA,'pilot_separation_conditioning.csv'),index=False)

# resolution sweep
seps2=np.arange(0.05,0.51,0.025); res_rows=[]
for sep in seps2:
    s=spec_two(sep,fwhm=0.07)
    Tt=tmatrix(0.8,0.15); y=noisy(Tt@s,38,rng)
    th,bh=estimate_state(0.8,0.15,42,rng); Tup=T0+JT*th+JB*bh
    r=reconstruct(Tup,y)
    # valley ratio near center: lower means resolved
    i0=np.argmin(abs(lam-1550)); valley=r[i0]
    # find peaks either side windows
    lp=r[lam<1550].max(); rp=r[lam>1550].max(); ratio=valley/max(min(lp,rp),1e-12)
    res_rows.append([sep,ratio,relerr(s,r)])
pd.DataFrame(res_rows,columns=['line_separation_nm','valley_to_peak_ratio','relative_error']).to_csv(os.path.join(DATA,'two_line_resolution.csv'),index=False)

# Jacobian calibration tolerance
jerrs=np.array([0,0.0025,0.005,0.01,0.02,0.03,0.05]); jr=[]
for je in jerrs:
    errs=[]; stateerrs=[]
    for rep in range(15):
        t=rng.uniform(-1.0,1.0); b=rng.uniform(-0.2,0.2); s=spec_two(0.2)
        Tt=tmatrix(t,b); y=noisy(Tt@s,35,rng)
        th,bh=estimate_state(t,b,40,rng,jac_err=je)
        # use similarly perturbed full Jacobian calibration
        JTu=JT*(1+rng.normal(0,je,JT.shape)); JBu=JB*(1+rng.normal(0,je,JB.shape))
        r=reconstruct(T0+JTu*th+JBu*bh,y)
        errs.append(relerr(s,r)); stateerrs.append(abs(th-t)+abs(bh-b))
    jr.append([je,np.mean(errs),np.mean(stateerrs)])
pd.DataFrame(jr,columns=['jacobian_fractional_std','mean_reconstruction_error','mean_abs_state_error_sum']).to_csv(os.path.join(DATA,'jacobian_tolerance.csv'),index=False)

# Dynamic tracking over 120 frames
frames=np.arange(120)
trueT=0.9*np.sin(2*np.pi*frames/80)+0.25*np.sin(2*np.pi*frames/23)
trueB=0.16*np.sin(2*np.pi*frames/53+0.7)+0.05*np.cos(2*np.pi*frames/17)
estT=[]; estB=[]; errS=[]; errP=[]
sdyn=spec_two(0.20)
for t,b in zip(trueT,trueB):
    th,bh=estimate_state(t,b,40,rng); estT.append(th); estB.append(bh)
    Tt=tmatrix(t,b); y=noisy(Tt@sdyn,35,rng)
    errS.append(relerr(sdyn,reconstruct(T0,y)))
    errP.append(relerr(sdyn,reconstruct(T0+JT*th+JB*bh,y)))
pd.DataFrame({'frame':frames,'true_dT_C':trueT,'est_dT_C':estT,'true_bend_state':trueB,'est_bend_state':estB,'stale_error':errS,'proposed_error':errP}).to_csv(os.path.join(DATA,'dynamic_tracking.csv'),index=False)

# computational/recalibration burden table - normalized estimates
# Full scan: 161 calibration frames, bank selects 1, single pilot 1 ref frame, dual 2 ref frames
cost=pd.DataFrame([
 ['Full wavelength recalibration',81,81,1.0,0.0],
 ['Temperature-bank lookup',0,1,0.0062,0.0],
 ['Single-pilot correction',1,1,0.0062,1.0],
 ['Proposed dual-pilot Jacobian',2,2,0.0124,2.0],
],columns=['method','reference_frames_per_update','matrix_updates','relative_update_time_vs_fullscan','latent_parameters_tracked'])
cost.to_csv(os.path.join(DATA,'update_cost.csv'),index=False)

# summary key metrics at example
summary={
 'pilot_jacobian_condition_number':Gcond,
 'example_true_dT_C':state_ex[0],
 'example_true_bend_state':state_ex[1],
 'example_est_dT_C':th if False else np.nan,
 'example_stale_relerr':relerr(s_ex,rec_stale),
 'example_single_relerr':relerr(s_ex,rec_single),
 'example_proposed_relerr':relerr(s_ex,rec_prop),
 'example_oracle_relerr':relerr(s_ex,rec_oracle),
 'dynamic_mean_stale_error':float(np.mean(errS)),
 'dynamic_mean_proposed_error':float(np.mean(errP)),
}
with open(os.path.join(DATA,'summary.txt'),'w') as f:
    for k,v in summary.items(): f.write(f'{k}={v}\n')

# FIGURES
# 1 architecture
fig,ax=plt.subplots(figsize=(12,6)); ax.axis('off'); ax.set_xlim(0,12); ax.set_ylim(0,6)
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def bx(x,y,w,h,txt):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.03',fill=False,linewidth=1.5); ax.add_patch(p); ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=11)
def ar(x1,y1,x2,y2): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=14,linewidth=1.3))
bx(.4,3.7,1.8,1.1,'Unknown spectrum\n1547–1553 nm'); bx(.4,1.6,1.8,1.1,'Two gated pilots\n1546.5 / 1553.5 nm')
bx(2.8,2.7,1.6,1.3,'3×1 coupler\n+ mode scrambler'); bx(5.0,2.6,1.8,1.5,'2-m MMF\n105-µm core\nNA = 0.22'); bx(7.4,2.6,1.8,1.5,'InGaAs camera\n256 sampled pixels'); bx(9.8,3.7,1.7,1.1,'Dual-pilot\nstate estimator'); bx(9.8,1.7,1.7,1.1,'Jacobian-updated\ntransmission matrix'); bx(9.8,.25,1.7,.9,'Nonnegative\nspectral inversion')
ar(2.2,4.2,2.8,3.55); ar(2.2,2.15,2.8,3.0); ar(4.4,3.35,5.0,3.35); ar(6.8,3.35,7.4,3.35); ar(9.2,3.55,9.8,4.2); ar(10.65,3.7,10.65,2.8); ar(10.65,1.7,10.65,1.15); ar(9.2,3.0,9.8,2.25)
ax.text(6,5.55,'Pilot-Jacobian self-calibration: physical-state tracking without wavelength rescanning',ha='center',fontsize=14,fontweight='bold')
fig.savefig(os.path.join(FIG,'Fig01_architecture.png'),dpi=300,bbox_inches='tight'); plt.close(fig)

# 2 working principle speckles
fig,ax=plt.subplots(figsize=(8,5));
I0=field_intensity(1550,0,0).reshape(12,12); I1=field_intensity(1550,1.2,0.18).reshape(12,12)
# show difference as three small axes through manual positions
plt.close(fig)
fig=plt.figure(figsize=(11,4))
for k,(arr,title) in enumerate([(I0,'Nominal speckle'),(I1,'Thermal + bend drift'),(I1-I0,'Difference')]):
    a=fig.add_axes([0.03+k*0.32,0.12,0.27,0.75]); im=a.imshow(arr,aspect='equal'); a.set_title(title); a.set_xlabel('camera x'); a.set_ylabel('camera y'); fig.colorbar(im,ax=a,fraction=0.046,pad=0.04)
fig.savefig(os.path.join(FIG,'Fig02_speckle_drift.png'),dpi=300,bbox_inches='tight'); plt.close(fig)

# 3 processing flow
fig,ax=plt.subplots(figsize=(11,5)); ax.axis('off'); ax.set_xlim(0,11); ax.set_ylim(0,5)
steps=[('Pilot frames',.3,3.2),('Subtract nominal\npilot speckles',2.2,3.2),('Solve 2×2\nstate inverse',4.3,3.2),('Update full TM\nT̂=T0+JTΔT+JBb',6.4,3.2),('Acquire sample\nspeckle',.3,1.2),('Nonnegative\nregularized inverse',6.4,1.2),('Recovered spectrum\n+ state report',8.7,1.2)]
for txt,x,y in steps: bx2=FancyBboxPatch((x,y),1.6,.9,boxstyle='round,pad=0.03',fill=False); ax.add_patch(bx2); ax.text(x+.8,y+.45,txt,ha='center',va='center',fontsize=10)
for p in [((1.9,3.65),(2.2,3.65)),((3.8,3.65),(4.3,3.65)),((5.9,3.65),(6.4,3.65)),((8.0,3.65),(8.0,1.65)),((1.9,1.65),(6.4,1.65)),((8.0,1.65),(8.7,1.65))]: ax.add_patch(FancyArrowPatch(*p,arrowstyle='-|>',mutation_scale=14))
fig.savefig(os.path.join(FIG,'Fig03_processing.png'),dpi=300,bbox_inches='tight'); plt.close(fig)

# 4 correlations
fig,ax=plt.subplots(figsize=(7,4)); ax.plot(lam-1550,corr_l,label='spectral shift (nm)'); ax.set_xlabel('Wavelength offset (nm)'); ax.set_ylabel('Speckle correlation'); ax.grid(True); ax.legend(); fig.savefig(os.path.join(FIG,'Fig04_spectral_correlation.png'),dpi=300,bbox_inches='tight'); plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4)); ax.plot(dTs,corr_T,label='temperature'); ax.plot(bends*b_scale_curv,corr_B,label='equiv. curvature'); ax.set_xlabel('Perturbation coordinate (°C or m$^{-1}$)'); ax.set_ylabel('Speckle correlation'); ax.grid(True); ax.legend(); fig.savefig(os.path.join(FIG,'Fig05_environment_correlation.png'),dpi=300,bbox_inches='tight'); plt.close(fig)

# 6 TM and singular values
fig=plt.figure(figsize=(10,4)); a1=fig.add_axes([.07,.15,.50,.75]); im=a1.imshow(T0,aspect='auto',origin='lower'); a1.set_xlabel('Wavelength channel'); a1.set_ylabel('Camera pixel'); a1.set_title('Nominal transmission matrix'); cax=fig.add_axes([.585,.15,.018,.75]); cb=fig.colorbar(im,cax=cax); a2=fig.add_axes([.73,.15,.24,.75]); sv=np.linalg.svd(T0,compute_uv=False); a2.semilogy(sv/sv[0]); a2.set_xlabel('Index'); a2.set_ylabel('Normalized singular value',labelpad=6); a2.grid(True); fig.savefig(os.path.join(FIG,'Fig06_tm_singular.png'),dpi=300,bbox_inches='tight'); plt.close(fig)

# 7 state estimation
pe=pd.read_csv(os.path.join(DATA,'pilot_state_estimation.csv')); fig,ax=plt.subplots(figsize=(7,4)); ax.plot(pe.pilot_snr_db,pe.temperature_rmse_C,marker='o',label='Temperature RMSE (°C)'); ax.plot(pe.pilot_snr_db,pe.bend_state_rmse,marker='s',label='Bend-state RMSE'); ax.set_xlabel('Pilot SNR (dB)'); ax.set_ylabel('RMSE'); ax.grid(True); ax.legend(); fig.savefig(os.path.join(FIG,'Fig07_state_estimation.png'),dpi=300,bbox_inches='tight'); plt.close(fig)

# 8 example reconstruction
re=pd.read_csv(os.path.join(DATA,'reconstruction_example.csv')); fig,ax=plt.subplots(figsize=(8,4.5));
for c in ['true','stale','single_pilot','proposed','oracle']: ax.plot(re.wavelength_nm,re[c],label=c.replace('_',' '))
ax.set_xlabel('Wavelength (nm)'); ax.set_ylabel('Normalized spectral power'); ax.grid(True); ax.legend(ncol=2); fig.savefig(os.path.join(FIG,'Fig08_reconstruction_example.png'),dpi=300,bbox_inches='tight'); plt.close(fig)

# 9 temperature sweep
et=pd.read_csv(os.path.join(DATA,'error_vs_temperature.csv')); fig,ax=plt.subplots(figsize=(7,4.5));
for c in ['stale','single_pilot','temperature_bank','proposed_dual_pilot','oracle']: ax.plot(et.dT_C,et[c],label=c.replace('_',' '))
ax.set_xlabel('Temperature drift ΔT (°C)'); ax.set_ylabel('Mean relative reconstruction error'); ax.grid(True); ax.legend(fontsize=8); fig.savefig(os.path.join(FIG,'Fig09_error_temperature.png'),dpi=300,bbox_inches='tight'); plt.close(fig)

# 10 bend sweep
eb=pd.read_csv(os.path.join(DATA,'error_vs_bend.csv')); fig,ax=plt.subplots(figsize=(7,4.5));
for c in ['stale','single_pilot','temperature_bank','proposed_dual_pilot','oracle']: ax.plot(eb.curvature_equiv_m_1 if 'curvature_equiv_m_1' in eb else eb['curvature_equiv_m-1'],eb[c],label=c.replace('_',' '))
ax.set_xlabel('Equivalent bend-curvature perturbation (m$^{-1}$)'); ax.set_ylabel('Mean relative reconstruction error'); ax.grid(True); ax.legend(fontsize=8); fig.savefig(os.path.join(FIG,'Fig10_error_bend.png'),dpi=300,bbox_inches='tight'); plt.close(fig)

# 11 heatmap
fig=plt.figure(figsize=(10,4));
for k,(hm,title) in enumerate([(heat_stale,'Stale calibration'),(heat_prop,'Dual-pilot Jacobian')]):
    a=fig.add_axes([.07+k*.49,.16,.37,.72]); im=a.imshow(hm,origin='lower',aspect='auto',extent=[Bgrid[0]*b_scale_curv,Bgrid[-1]*b_scale_curv,Tgrid[0],Tgrid[-1]]); a.set_xlabel('Equivalent curvature (m$^{-1}$)'); a.set_ylabel('ΔT (°C)'); a.set_title(title); fig.colorbar(im,ax=a,fraction=.046,pad=.04,label='Relative error')
fig.savefig(os.path.join(FIG,'Fig11_combined_drift_heatmap.png'),dpi=300,bbox_inches='tight'); plt.close(fig)

# 12 pilot separation
ps=pd.read_csv(os.path.join(DATA,'pilot_separation_conditioning.csv')); fig,ax=plt.subplots(figsize=(7,4)); ax.plot(ps.pilot_separation_nm,ps.condition_number,marker='o'); ax.set_xlabel('Pilot wavelength separation (nm)'); ax.set_ylabel('Jacobian condition number'); ax.grid(True); fig.savefig(os.path.join(FIG,'Fig12_pilot_conditioning.png'),dpi=300,bbox_inches='tight'); plt.close(fig)

# 13 line resolution
rr=pd.read_csv(os.path.join(DATA,'two_line_resolution.csv')); fig,ax=plt.subplots(figsize=(7,4)); ax.plot(rr.line_separation_nm,rr.valley_to_peak_ratio,marker='o',label='Valley/peak'); ax.plot(rr.line_separation_nm,rr.relative_error,marker='s',label='Relative error'); ax.set_xlabel('Two-line separation (nm)'); ax.set_ylabel('Metric'); ax.grid(True); ax.legend(); fig.savefig(os.path.join(FIG,'Fig13_resolution.png'),dpi=300,bbox_inches='tight'); plt.close(fig)

# 14 tolerance
jt=pd.read_csv(os.path.join(DATA,'jacobian_tolerance.csv')); fig,ax=plt.subplots(figsize=(7,4)); ax.plot(100*jt.jacobian_fractional_std,jt.mean_reconstruction_error,marker='o'); ax.set_xlabel('Jacobian calibration error, 1σ (%)'); ax.set_ylabel('Mean relative reconstruction error'); ax.grid(True); fig.savefig(os.path.join(FIG,'Fig14_jacobian_tolerance.png'),dpi=300,bbox_inches='tight'); plt.close(fig)

# 15 dynamic
dy=pd.read_csv(os.path.join(DATA,'dynamic_tracking.csv')); fig=plt.figure(figsize=(9,6)); a1=fig.add_axes([.10,.57,.82,.33]); a1.plot(dy.frame,dy.true_dT_C,label='true ΔT'); a1.plot(dy.frame,dy.est_dT_C,label='estimated ΔT'); a1.plot(dy.frame,dy.true_bend_state,label='true bend'); a1.plot(dy.frame,dy.est_bend_state,label='estimated bend'); a1.set_ylabel('State coordinate'); a1.grid(True); a1.legend(ncol=2,fontsize=8); a2=fig.add_axes([.10,.12,.82,.33]); a2.plot(dy.frame,dy.stale_error,label='stale'); a2.plot(dy.frame,dy.proposed_error,label='proposed'); a2.set_xlabel('Frame'); a2.set_ylabel('Relative reconstruction error'); a2.grid(True); a2.legend(); fig.savefig(os.path.join(FIG,'Fig15_dynamic_tracking.png'),dpi=300,bbox_inches='tight'); plt.close(fig)

print('done')
print(open(os.path.join(DATA,'summary.txt')).read())
