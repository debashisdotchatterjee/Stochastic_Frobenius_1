#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STOCHASTIC FROBENIUS: exact verification + journal-grade numerical study.

Colab: upload/run this script, or open the companion .ipynb notebook.
Python >=3.10; numpy, scipy, pandas, matplotlib, seaborn not required.
All numerical output is seeded, audited, written to disk, and zipped.

MODEL: a=(5,6,7,8); p=(2/5,3/10,1/5,1/10).
F_eta = max{n>=1 : P(exists k>=0, S_k=n)<eta}, 0<eta<1/E[X].
The exact rational threshold computations are backed by a global finite
minorization certificate. Approximate simulation-based inference is explicitly
separated from finite-sample certified bounds.

Non-experimental results use no real data; simulated outcomes are labeled.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, platform, shutil, sys, time, warnings, zipfile
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction as Q
from functools import reduce
from heapq import heappop, heappush
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
if 'google.colab' not in sys.modules:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm, binom

A = (5, 6, 7, 8)
P = (Q(2,5), Q(3,10), Q(1,5), Q(1,10))
LEVELS = (Q(1,10), Q(12,100), Q(14,100), Q(15,100), Q(16,100))
EXPECTED = (9, 15, 21, 27, 33)
MASTER_SEED = 20261023
OUT_NAME = 'stochastic_frobenius_outputs'
COLORS = {'ink':'#14243B','blue':'#155E92','teal':'#007F86','gold':'#C88C2C',
          'red':'#B34B51','gray':'#66768A','pale':'#EDF3F7','green':'#397B57'}


def make_styles():
    plt.rcParams.update({
        'font.family':'DejaVu Sans', 'font.size':10.5, 'axes.titlesize':13,
        'axes.titleweight':'bold', 'axes.labelsize':10.5,
        'axes.spines.right':False, 'axes.spines.top':False,
        'axes.edgecolor':'#BAC6D2', 'axes.labelcolor':COLORS['ink'],
        'text.color':COLORS['ink'], 'xtick.color':COLORS['gray'],
        'ytick.color':COLORS['gray'], 'figure.facecolor':'white',
        'axes.facecolor':'white', 'savefig.facecolor':'white',
        'axes.grid':True, 'grid.color':'#E7ECF1', 'grid.linewidth':0.65,
        'grid.alpha':0.8, 'legend.frameon':False, 'pdf.fonttype':42,
        'ps.fonttype':42, 'lines.linewidth':2.0,
    })


def save_fig(fig, label: str, out: Path):
    for ext in ('png','pdf','svg'):
        fig.savefig(out/'figures'/f'{label}.{ext}', dpi=320, bbox_inches='tight', pad_inches=.18)
    plt.close(fig)


def as_number(v):
    return float(v) if isinstance(v,Q) else v


def residue_apery(a=A):
    a = tuple(map(int,a))
    if reduce(math.gcd,a)!=1: raise ValueError('gcd must be one')
    m=a[0]; d=[None]*m; d[0]=0; heap=[(0,0)]
    while heap:
        v,r=heappop(heap)
        if v!=d[r]: continue
        for z in a[1:]:
            rr=(r+z)%m; w=v+z
            if d[rr] is None or w<d[rr]: d[rr]=w; heappush(heap,(w,rr))
    assert all(t is not None for t in d)
    return d


def validate_generators(a):
    assert tuple(sorted(set(a)))==tuple(a) and a[0]>=2
    assert reduce(math.gcd,a)==1
    for i,x in enumerate(a):
        others=[v for j,v in enumerate(a) if j!=i]
        d=[False]*(x+1); d[0]=True
        for k in range(1,x+1): d[k]=any(k>=v and d[k-v] for v in others)
        assert not d[x], f'Generator {x} is redundant'


def renewal_q(a, p, nmax):
    assert len(a)==len(p)
    u=[Q(1)]+[Q(0)]*nmax
    for n in range(1,nmax+1):
        u[n]=sum((p[j]*u[n-x] for j,x in enumerate(a) if x<=n),Q(0))
    return u


def renewal_float(a,p,nmax):
    p=np.asarray(p,float)
    u=np.zeros(nmax+1,float); u[0]=1.
    for n in range(1,nmax+1):
        u[n]=sum(p[j]*u[n-a[j]] for j in range(len(a)) if n>=a[j])
    return u


def renewal_grad(a,p,nmax):
    p=np.asarray(p,float); d=len(a)
    u=np.zeros(nmax+1); g=np.zeros((nmax+1,d)); u[0]=1.
    for n in range(1,nmax+1):
        for j,z in enumerate(a):
            if n>=z:
                u[n]+=p[j]*u[n-z]
                g[n]+=p[j]*g[n-z]
                g[n,j]+=u[n-z]
    return u,g


def renewal_replicates(a,prob_mat,k):
    # Rows are independent fitted probability vectors.
    prob_mat=np.asarray(prob_mat,dtype=float)
    R=prob_mat.shape[0]
    u=np.zeros((R,k+1),dtype=float); u[:,0]=1
    for n in range(1,k+1):
        for j,z in enumerate(a):
            if n>=z: u[:,n]+=prob_mat[:,j]*u[:,n-z]
    return u


def markov_P(a,p):
    L=max(a)
    Z=Q(0) if isinstance(p[0],Q) else 0.
    M=[[Z for _ in range(L)] for _ in range(L)]
    for i in range(1,L): M[i][i-1]=Q(1) if isinstance(Z,Q) else 1.
    for z,v in zip(a,p): M[0][z-1]+=v
    return M


def mat_mul(U,V):
    m=len(U); Z=Q(0)
    return [[sum((U[i][k]*V[k][j] for k in range(m)),Z)
             for j in range(m)] for i in range(m)]


def mat_pow(M,e):
    d=len(M); T=[[Q(int(i==j)) for j in range(d)] for i in range(d)]
    while e:
        if e&1: T=mat_mul(T,M)
        e//=2
        if e: M=mat_mul(M,M)
    return T


def minorization(a, lower_probs, B):
    # Works when lower probabilities sum <=1. P_lower is substochastic:
    # P(p)^B >= P_lower^B entrywise whenever p_j >= lower_probs[j].
    assert all(isinstance(x,Q) for x in lower_probs)
    assert min(lower_probs)>0
    assert sum(lower_probs,Q(0))<=1
    T=mat_pow(markov_P(a,lower_probs),B)
    delta=min(t for row in T for t in row)
    assert 0<delta<=Q(1,max(a))
    q=1-max(a)*delta
    assert Q(0)<=q<Q(1)
    return delta,q


def certified_h(q, margin):
    assert isinstance(q,Q) and isinstance(margin,Q)
    assert 0<margin<1 and 0<=q<1
    if q==0: return 1
    # FLOAT logs are only an initial guess; final inequality verified EXACTLY.
    # Avoid thousands of expensive Fraction multiplications by exponentiation.
    h=max(1, int(math.floor(math.log(float(margin))/math.log(float(q))))-2)
    while q**h>=margin: h+=1
    while h>1 and q**(h-1)<margin: h-=1
    assert q**h<margin and (h==1 or q**(h-1)>=margin)
    return h


def certificate(a, lower_probs, mu_upper, eta, B):
    assert isinstance(mu_upper,Q) and isinstance(eta,Q)
    gap=1/mu_upper-eta
    if min(lower_probs)<=0 or gap<=0:
        return None
    d,q=minorization(a,lower_probs,B)
    h=certified_h(q,gap)
    return {'delta':d,'q':q,'h':h,'K':B*h-1,'tail_from':B*h,
            'mean_upper':mu_upper, 'margin':gap}


def decimal_floor_q(x,d=10000):
    # Float value x is rounded DOWN one extra grid cell for robustness.
    return Q(max(0,int(math.floor(float(x)*d))-1),d)


def decimal_ceil_q(x,d=10000):
    return Q(int(math.ceil(float(x)*d))+1,d)


def fraction_str(x):
    return f'{x.numerator}/{x.denominator}' if isinstance(x,Q) else str(x)


def write_df(df,path,out):
    target=out/'tables'/f'{path}.csv'; df.to_csv(target,index=False, float_format='%.12g')
    # Escape table fields before generating booktabs LaTeX with pandas.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',FutureWarning)
        df.to_latex(out/'tables'/f'{path}.tex',index=False,escape=True,
                    caption=f'{path.replace("_"," ").capitalize()}.',label=f'tab:{path}',
                    float_format=lambda x:f'{x:.6g}')
    return target


def exact_combinatorial_u(n,a,p):
    # Independent non-recursive coefficient computation:
    # sum over integer vectors z>=0 with a*z=n of multinomial(k;z) prod p_j^z_j.
    ans=Q(0); solutions=0
    for z0 in range(n//a[0]+1):
        for z1 in range((n-a[0]*z0)//a[1]+1):
            for z2 in range((n-a[0]*z0-a[1]*z1)//a[2]+1):
                r=n-a[0]*z0-a[1]*z1-a[2]*z2
                if r<0 or r%a[3]: continue
                z3=r//a[3]; zs=(z0,z1,z2,z3); k=sum(zs)
                term=Q(math.factorial(k),math.prod(math.factorial(z) for z in zs))
                for v,ww in zip(p,zs): term*=v**ww
                ans+=term; solutions+=1
    return ans,solutions


def verify_exact(out):
    validate_generators(A)
    apery=residue_apery(A); F=max(apery)-A[0]
    assert apery==[0,6,7,8,14] and F==9
    mu=sum((z*w for z,w in zip(A,P)),Q(0))
    assert mu==6
    B=F+2*max(A)-1
    assert B==24
    certs={}
    for eta in LEVELS:
        certs[eta]=certificate(A,P,mu,eta,B)
    assert all(c is not None for c in certs.values())
    delta,q=minorization(A,P,B)
    assert delta==Q(83,6250) and q==Q(2793,3125)
    K=max(c['K'] for c in certs.values())
    u=renewal_q(A,P,K)
    rows=[]
    for eta,expected in zip(LEVELS,EXPECTED):
        c=certs[eta]
        last=max((n for n in range(1,c['K']+1) if u[n]<eta),default=0)
        assert last==expected, (eta,last,expected)
        assert all(u[n]>=eta for n in range(last+1,c['K']+1))
        # n >= K+1 certified by exact Doeblin bound, NOT extrapolated numerically.
        rows.append({'eta':float(eta),'eta_fraction':fraction_str(eta),
                     'classical_F':F,'stochastic_F_eta':last,
                     'global_certified_from':c['tail_from'],'B':B,
                     'minorization_delta':float(c['delta']),
                     'minorization_delta_fraction':fraction_str(c['delta']),
                     'geometric_q':float(c['q']), 'h':c['h'],
                     'difference_limit_minus_eta':float(c['margin'])})
    table=pd.DataFrame(rows); write_df(table,'01_exact_thresholds',out)
    sel=[0,1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,26,27,28,32,33,34,35,40,60,80,100]
    assert all(n<=K for n in sel)
    hit=pd.DataFrame({'n':sel,'reachable':[int(u[n]>0) for n in sel],
                      'u_n_exact':[fraction_str(u[n]) for n in sel],
                      'u_n':[float(u[n]) for n in sel]})
    write_df(hit,'02_selected_hitting_probabilities',out)
    known={9:Q(0),10:Q(4,25),15:Q(13,125),21:Q(83,625),27:Q(37,250),33:Q(2456,15625)}
    assert all(u[n]==v for n,v in known.items())
    # Independent combinatorial check (exact), for moderate n.
    checks=[]
    for n in [9,10,15,21,27,33,40,50]:
        alt,k=exact_combinatorial_u(n,A,P)
        assert alt==u[n]
        checks.append({'n':n,'renewal_fraction':fraction_str(u[n]),
                       'independent_compositions_fraction':fraction_str(alt),
                       'num_vectors_z':k,'agree':True})
    write_df(pd.DataFrame(checks),'03_independent_exact_checks',out)
    # Independent Markov-power check at finite n (in exact rational arithmetic).
    transition=markov_P(A,P)
    powers={0:[[Q(int(i==j)) for j in range(max(A))] for i in range(max(A))]}
    vals=[]; Mk=powers[0]
    for n in range(0,101):
        assert Mk[0][0]==u[n], f'Markov check fails at {n}'
        if n in [0,5,9,10,15,21,27,33,60,100]:
            vals.append({'n':n,'u_n':float(u[n]),'P_power_00':float(Mk[0][0]),'exact_agreement':True})
        if n<100: Mk=mat_mul(Mk,transition)
    write_df(pd.DataFrame(vals),'04_markov_crosscheck',out)
    report={'generator_set':list(A),'selection_probabilities':[fraction_str(t) for t in P],
      'mean_step':fraction_str(mu),'renewal_limit':fraction_str(1/mu),
      'embedding_dimension':4,'apery_set':apery,'classical_frobenius':F,
      'B':B,'P_to_B_min':fraction_str(delta),'q':fraction_str(q),
      'last_thresholds':dict(zip([str(float(x)) for x in LEVELS],EXPECTED)),
      'proof': 'For n>=B*h, abs(u_n-1/mu)<=q^h < (1/mu-eta); all n<=B*h-1 evaluated in exact fractions.',
      'status':'All exact identities, thresholds and independent checks passed.'}
    (out/'verification'/'exact_certificate.json').write_text(json.dumps(report,indent=2))
    with (out/'verification'/'exact_certificate.txt').open('w') as fh:
        for key,val in report.items(): fh.write(f'{key}: {val}\n')
    (out/'verification'/'exact_tables_full.json').write_text(json.dumps(
        {'certificates':[{k:(fraction_str(v) if isinstance(v,Q) else v) for k,v in c.items()}
                         for c in certs.values()]},indent=2))
    return {'F':F,'B':B,'mu':mu,'u':u,'K':K,'certificate':certs,'apery':apery,'q':q}


def plot_renewal(base,out):
    n=np.arange(1,141)
    u=np.array([float(base['u'][j]) for j in n])
    gaps=[j for j in range(1,56) if base['u'][j]==0]
    fig,axs=plt.subplots(1,2,figsize=(14.5,4.65),
                         gridspec_kw={'width_ratios':[1,1.13]})
    for ax in axs:
        ax.plot(n,u,color=COLORS['blue'],lw=1.75,label=r'Exact $u_n$')
        ax.axhline(1/6,color=COLORS['gold'],ls='--',lw=2,label=r'Renewal limit $1/\mu=1/6$')
        ax.axhline(.14,color=COLORS['teal'],ls=':',lw=2,label=r'Reliability level $\eta=0.14$')
        ax.axvline(21,color=COLORS['red'],alpha=.72,ls='--',lw=1.35)
        ax.set_xlabel('Target integer n')
    axs[0].scatter(n[:40],u[:40],s=18,color=COLORS['blue'],zorder=3)
    axs[0].scatter(gaps,np.zeros(len(gaps)),marker='x',s=42,color=COLORS['red'],
                   label='Classical gaps',zorder=4)
    axs[0].set(xlim=(1,42),ylim=(-.022,.44),ylabel=r'Hitting probability $u_n$',
               title='Early targets: structural gaps and overshoot')
    axs[1].set(xlim=(18,140),ylim=(.118,.205),
               title='Longer targets: convergence and last undershoot')
    axs[1].annotate(r'$F_{0.14}=21$',(21,float(base['u'][21])),xytext=(44,.127),
             arrowprops=dict(arrowstyle='->',color=COLORS['red']),color=COLORS['red'])
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,loc='lower center',bbox_to_anchor=(.5,-.045),
               fontsize=9,frameon=False)
    fig.suptitle('Exact stochastic reachability: probability oscillations and renewal equilibrium',
                 fontsize=14.4,fontweight='bold')
    fig.tight_layout(rect=[0,.052,1,.95],w_pad=2.2)
    save_fig(fig,'Fig01_exact_renewal_probabilities',out)


def plot_gaps(base,out):
    xx=np.arange(1,101)
    class_rep=np.array([int(base['u'][i]>0) for i in xx])
    data=np.vstack([class_rep]+[np.array([float(base['u'][i])>=e for i in xx],int)
                                  for e in [.10,.12,.14,.15,.16]])
    fig,ax=plt.subplots(figsize=(11,3.7))
    from matplotlib.colors import ListedColormap
    ax.imshow(data,aspect='auto',cmap=ListedColormap(['#F1D5D1','#CDE5DA']),
              interpolation='nearest',vmin=0,vmax=1,extent=[.5,100.5,5.5,-.5])
    ax.set_yticks(range(6),['Semigroup member',r'$u_n\geq0.10$',r'$u_n\geq0.12$',
                                r'$u_n\geq0.14$',r'$u_n\geq0.15$',r'$u_n\geq0.16$'])
    ax.set_xticks(range(5,101,5)); ax.set(xlabel='Target integer n',
                title='Structural attainability versus probabilistic reliability')
    ax.grid(False)
    save_fig(fig,'Fig02_attainability_reliability_map',out)


def plot_staircase(base,out):
    etas=np.linspace(.001,.164,220)
    u=np.array([float(t) for t in base['u']])
    f=np.array([np.where(u[1:]<e)[0].max()+1 for e in etas])
    # Baseline K certifies ALL eta<=0.16, but for 0.164 a new certificate is necessary.
    # Use the same q to certify the largest eta on the grid.
    max_eta=decimal_ceil_q(etas.max(),1000000)
    c=certificate(A,P,base['mu'],max_eta,base['B'])
    assert c is not None
    if c['K']>base['K']:
        tail=renewal_float(A,np.array([float(t) for t in P]),c['K'])
        u=tail
        f=np.array([np.where(u[1:]<e)[0].max()+1 for e in etas])
    assert len(u)-1>=c['K']
    fig,ax=plt.subplots(figsize=(9.3,4.8))
    ax.step(etas,f,where='post',color=COLORS['teal'],label=r'$F_\eta$')
    ax.scatter([float(e) for e in LEVELS],EXPECTED,zorder=4,color=COLORS['red'],s=38,
               label='Exactly certified table entries')
    ax.set(xlabel=r'Reliability threshold $\eta$',ylabel=r'Stochastic Frobenius threshold $F_\eta$',
           title='Monotone, discontinuous stochastic Frobenius frontier')
    ax.legend(loc='upper left'); ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    save_fig(fig,'Fig03_probability_threshold_staircase',out)
    return pd.DataFrame({'eta':etas,'F_eta_numeric':f,
           'global_tail_certified_for_grid':True,'scan_K':len(u)-1})


def plot_convergence(base,out):
    idx=np.arange(0,base['K']+1)
    err=np.array([abs(float(t)-1/6) for t in base['u']])
    bound=np.array([float(base['q'])**(n//base['B']) for n in idx])
    assert np.all(err<=bound+1e-12)
    fig,ax=plt.subplots(figsize=(9.5,4.7))
    ax.semilogy(idx,np.maximum(err,1e-17),color=COLORS['blue'],lw=1.2,
                label=r'$|u_n-1/\mu|$ (exact recursion)')
    ax.semilogy(idx,np.maximum(bound,1e-17),color=COLORS['red'],ls='--',lw=2,
                label=r'Certified bound $q^{\lfloor n/B\rfloor}$')
    ax.set(xlabel='Target integer n',ylabel='Absolute discrepancy (log scale)',
           title='Convergence to renewal equilibrium: observed and certified')
    ax.legend()
    save_fig(fig,'Fig04_renewal_convergence_certificate',out)
    df=pd.DataFrame({'n':idx,'absolute_error':err,'rigorous_error_bound':bound})
    write_df(df,'05_convergence_and_bounds',out)


def trajectories_mc(base,out,rng,profile):
    num=15000 if profile=='quick' else 120000
    max_target=110
    state=np.zeros(num,dtype=np.int32)
    counts=np.zeros(max_target+1,dtype=np.int64); counts[0]=num
    p=np.array([float(x) for x in P]); a=np.array(A)
    # One independent path per row, simulated until it crosses max_target;
    # every path can visit any given n at most once (Bernoulli outcome).
    while True:
        active=np.flatnonzero(state<=max_target)
        if active.size==0: break
        increments=rng.choice(a,size=active.size,p=p)
        state[active]+=increments
        visited=state[active]
        visited=visited[visited<=max_target]
        counts+=np.bincount(visited,minlength=max_target+1)
    n=np.array([5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,27,33,40,60,80,100])
    successes=counts[n]; empirical=successes/num
    z=norm.ppf(.975)
    denom=1+z*z/num
    center=(empirical+z*z/(2*num))/denom
    half=(z/denom)*np.sqrt(empirical*(1-empirical)/num+z*z/(4*num*num))
    lo=np.maximum(0,center-half); hi=np.minimum(1,center+half)
    exact=np.array([float(base['u'][j]) for j in n])
    assert np.all((lo>=0)&(hi<=1))
    table=pd.DataFrame({'n':n,'M_paths':num,'hits':successes,'empirical_u':empirical,
            'exact_u':exact,'mc_error':empirical-exact,'wilson_95_lower':lo,
            'wilson_95_upper':hi,'theory_within_MC_Wilson':(lo<=exact)&(exact<=hi)})
    write_df(table,'06_pathwise_monte_carlo',out)
    fig,ax=plt.subplots(figsize=(10.8,4.8))
    ax.errorbar(n,empirical,yerr=np.maximum(0.,np.vstack((empirical-lo,hi-empirical))),fmt='o',ms=4,
                capsize=2.3,color=COLORS['teal'],label='Monte Carlo ± Wilson 95% interval')
    ax.plot(n,exact,'s-',ms=4,lw=1.25,color=COLORS['red'],label='Exact renewal recurrence')
    ax.axhline(1/6,color=COLORS['gold'],ls='--',alpha=.8)
    ax.set(xlabel='Target n',ylabel=r'$\Pr(\text{hit }n)$',
           title=f'Independent path simulation versus exact probabilities (M={num:,})')
    ax.legend()
    save_fig(fig,'Fig05_independent_path_validation',out)
    return table


def numeric_thresholds_many(prob_matrix,eta,reference_certificate,exact_near=True):
    # Global finite tail bound is obtained from a COMMON rational lower
    # transition matrix and rational upper bound on means.
    M=np.asarray(prob_matrix,float)
    R,d=M.shape
    eta_q=Q(str(eta)) if not isinstance(eta,Q) else eta
    a=np.array(A); B=24
    l=tuple(decimal_floor_q(M[:,j].min()) for j in range(4))
    mu_upper=decimal_ceil_q(np.max(M @ a))
    cert=certificate(A,l,mu_upper,eta_q,B)
    if cert is None: raise RuntimeError('No common tail certificate; narrow the parameter region.')
    if cert['K']>120000:
        raise RuntimeError('Certified computation exceeds 120000 steps; narrow parameter ranges or strengthen minorization.')
    # Vectorized recurrence (float). Tail certificate itself is exact.
    u=renewal_replicates(A,M,cert['K'])
    last=np.zeros(R,dtype=int)
    ambiguous=[]
    for n in range(1,cert['K']+1):
        close=np.flatnonzero(np.abs(u[:,n]-float(eta_q))<2e-11)
        if exact_near and close.size:
            for row in close:
                # Exact rational data whenever probabilities are rational grid or
                # multinomial counts. Decimal str(float) is rational, and we
                # normalize only if the input row sums exactly after this conversion.
                pp=tuple(Q(str(x)) for x in M[row])
                ss=sum(pp,Q(0)); pp=tuple(x/ss for x in pp)
                eq=renewal_q(A,pp,n)[n]
                is_less=(eq<eta_q)
                if is_less: last[row]=n
                ambiguous.append((int(row),n,bool(is_less)))
        values=(u[:,n]<float(eta_q))
        if close.size: values[close]=False
        last[values]=n
    return last,cert,len(ambiguous)


def sensitivity_study(out,profile):
    # Fixed p2=.3, p4=.1, p3=.6-p1.
    theta=np.linspace(.15,.50,31 if profile=='quick' else 81)
    mat=np.stack([theta,np.full_like(theta,.3),.6-theta,np.full_like(theta,.1)],axis=1)
    f,c,amb=numeric_thresholds_many(mat,Q(14,100),None)
    df=pd.DataFrame({'p1':theta,'p2':.3,'p3':.6-theta,'p4':.1,
                    'mean_step':mat @ np.array(A),'renewal_limit':1/(mat @ np.array(A)),
                    'F_eta_0_14':f,'global_K_for_family':c['K']})
    write_df(df,'07_probability_sensitivity',out)
    fig,ax=plt.subplots(figsize=(9.8,4.8))
    ax.step(theta,f,where='mid',color=COLORS['blue'],lw=2.1)
    ax.scatter([.4],[21],color=COLORS['red'],s=80,zorder=5,label='Baseline: (0.4, 0.3, 0.2, 0.1)')
    ax.set(xlabel=r'$p_1$  (with $p_2=0.3$, $p_3=0.6-p_1$, $p_4=0.1$)',
           ylabel=r'$F_{0.14}$',title='Sensitivity of the reliability frontier to selection probabilities')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True)); ax.legend()
    save_fig(fig,'Fig06_probability_sensitivity',out)

    xx=np.linspace(.20,.40,11 if profile=='quick' else 25)
    yy=np.linspace(.20,.35,9 if profile=='quick' else 21)
    gx,gy=np.meshgrid(xx,yy)
    m=np.column_stack([gx.ravel(),gy.ravel(),.9-gx.ravel()-gy.ravel(),np.full(gx.size,.1)])
    assert (m>0).all()
    ff,c2,amb2=numeric_thresholds_many(m,Q(14,100),None)
    grid=ff.reshape(gx.shape)
    fig,ax=plt.subplots(figsize=(9.6,5.7))
    im=ax.imshow(grid,origin='lower',aspect='auto',interpolation='nearest',
        extent=[xx.min(),xx.max(),yy.min(),yy.max()],cmap='viridis')
    cb=fig.colorbar(im,ax=ax,pad=.02); cb.set_label(r'$F_{0.14}$ (integer)')
    ax.scatter([.4],[.3],s=80,marker='x',c='white',lw=2.5,label='Baseline')
    ax.set(xlabel=r'$p_1$',ylabel=r'$p_2$  (with $p_3=0.9-p_1-p_2$, $p_4=0.1$)',
           title='Two-dimensional sensitivity surface for the stochastic threshold')
    ax.legend(loc='lower left')
    save_fig(fig,'Fig07_two_parameter_sensitivity',out)
    write_df(pd.DataFrame({'p1':m[:,0],'p2':m[:,1],'p3':m[:,2],
                           'p4':m[:,3],'F_eta_0_14':ff}),
             '08_two_parameter_sensitivity',out)
    return {'line_family_K':c['K'],'grid_family_K':c2['K'],
            'near_boundary_exact_fallbacks':amb+amb2}


def delta_method_study(out,rng,profile):
    targets=(15,21,27,33)
    _,G=renewal_grad(A,[float(x) for x in P],max(targets))
    p=np.array([float(x) for x in P])
    Sigma=np.diag(p)-np.outer(p,p)
    theory={n:float(G[n]@Sigma@G[n]) for n in targets}
    assert all(theory[n]>0 for n in targets)
    R=400 if profile=='quick' else 2400
    sizes=[500,2000] if profile=='quick' else [500,2000,10000]
    rows=[]; z=norm.ppf(.975); standardized=None
    for N in sizes:
        counts=rng.multinomial(N,p,size=R)
        phat=counts/N
        U=renewal_replicates(A,phat,max(targets))
        for target in targets:
            u_true=float(renewal_float(A,p,target)[target])
            errors=U[:,target]-u_true
            # Plug-in SE with estimated multinomial covariance and gradient
            Ug=np.zeros((R,target+1)); Gg=np.zeros((R,target+1,4)); Ug[:,0]=1
            for n in range(1,target+1):
                for j,a in enumerate(A):
                    if n>=a:
                        Ug[:,n]+=phat[:,j]*Ug[:,n-a]
                        Gg[:,n,:]+=phat[:,j,None]*Gg[:,n-a,:]
                        Gg[:,n,j]+=Ug[:,n-a]
            grads=Gg[:,target]
            # g^T[diag(phat)-phat phat^T]g
            sigma_sq=(phat*grads**2).sum(axis=1)-(phat*grads).sum(axis=1)**2
            sigma_sq=np.maximum(sigma_sq,0)
            half=z*np.sqrt(sigma_sq/N)
            cover=np.abs(errors)<=half
            rows.append({'n':target,'N':N,'R_replications':R,'true_u':u_true,
                'bias':np.mean(errors),'rmse':np.sqrt(np.mean(errors**2)),
                'empirical_sd':np.std(errors,ddof=1),
                'theoretical_asymptotic_sd':np.sqrt(theory[target]/N),
                'empirical_delta_95_coverage':np.mean(cover),
                'coverage_mc_se':np.sqrt(np.mean(cover)*(1-np.mean(cover))/R),
                'variance_factor_gSigmag':theory[target]})
            if target==21 and N==sizes[-1]:
                standardized=errors/np.sqrt(theory[target]/N)
    tab=pd.DataFrame(rows); write_df(tab,'09_delta_method_sampling_properties',out)
    fig,ax=plt.subplots(figsize=(9.4,4.8))
    ax.hist(standardized,bins=38,density=True,color=COLORS['teal'],alpha=.55,
            edgecolor='white',label='Standardized simulated estimates')
    X=np.linspace(-4,4,200)
    ax.plot(X,norm.pdf(X),color=COLORS['red'],lw=2.7,label='Standard normal density')
    ax.set(xlabel=r'$(\widehat u_{21}-u_{21})/\sqrt{V_{21}/N}$',ylabel='Density',
           title=f'Delta-method approximation: n=21, N={sizes[-1]:,}, R={R:,}')
    ax.legend(); save_fig(fig,'Fig08_delta_method_normality',out)
    fig,ax=plt.subplots(figsize=(9.4,4.8))
    for n in targets:
        dd=tab[tab.n==n]
        ax.plot(dd.N,dd.rmse,'o-',label=f'n={n}')
    ax.set_xscale('log');ax.set_yscale('log')
    ax.set(xlabel='Number of observed packet draws N (log)',ylabel='RMSE of plug-in hitting probability (log)',
           title='Sampling accuracy across targets')
    ax.legend(ncol=2); save_fig(fig,'Fig09_sampling_accuracy',out)
    return tab


def threshold_estimation_study(out,rng,profile):
    R=300 if profile=='quick' else 1200
    sizes=[500,2000] if profile=='quick' else [500,2000,10000]
    p=np.array([float(x) for x in P]); result=[]; est_tables=[]
    for N in sizes:
        counts=rng.multinomial(N,p,size=R)
        phat=counts/N
        f,c,amb=numeric_thresholds_many(phat,Q(14,100),None)
        correct=f==21
        result.append({'N':N,'R_replications':R,'actual_F':21,
                       'exact_recovery_rate':float(correct.mean()),
                       'recovery_mc_se':float(np.sqrt(correct.mean()*(1-correct.mean())/R)),
                       'mean_estimated_F':float(f.mean()),
                       'median_estimated_F':float(np.median(f)),
                       'MAE':float(np.abs(f-21).mean()),
                       'prob_underestimate':float((f<21).mean()),
                       'prob_overestimate':float((f>21).mean()),
                       'certified_common_tail_K':c['K'],
                       'numerically_near_boundaries_resolved_exactly':amb})
        tab=pd.DataFrame({'replicate':np.arange(R)+1,'N':N,'p1_hat':phat[:,0],
                     'p2_hat':phat[:,1],'p3_hat':phat[:,2],'p4_hat':phat[:,3],
                     'Fhat_0_14':f})
        write_df(tab,f'10a_threshold_estimates_N{N}',out)
        est_tables.append(tab)
    results=pd.DataFrame(result); write_df(results,'10_threshold_estimation_performance',out)
    fig,axs=plt.subplots(1,len(sizes),figsize=(13,4.5),sharey=True)
    for ax,N,df in zip(axs,sizes,est_tables):
        cnt=Counter(df.Fhat_0_14)
        keys=np.array(sorted(cnt))
        ax.bar(keys,[cnt[k]/R for k in keys],width=1.4,color=COLORS['blue'],alpha=.85)
        ax.axvline(21,color=COLORS['red'],ls='--',lw=2,label=r'True $F_{0.14}=21$')
        ax.set_title(f'N = {N:,}'); ax.set_xlabel(r'Plug-in $\widehat F_{0.14}$')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].set_ylabel('Relative frequency')
    axs[0].legend(fontsize=8)
    fig.suptitle('Sampling distribution of the integer-valued plug-in threshold',fontweight='bold')
    save_fig(fig,'Fig10_integer_threshold_sampling_distribution',out)
    return results


def certified_interval_one(counts,eta,confidence=0.95):
    # Implementation of the manuscript's simultaneous finite-N CI with a
    # SHARPER (but still mathematically valid) substochastic-matrix bound.
    # The event max_j |phat_j-p_j| <= t has probability >= 1-beta.
    N=int(sum(counts)); phat=np.array(counts,float)/N
    beta=1-confidence; t=math.sqrt(math.log(8/beta)/(2*N))
    l=tuple(decimal_floor_q(max(0,x-t)) for x in phat)
    muhat=float(np.dot(phat,np.array(A)))
    mu_plus=min(float(max(A)),muhat+2*(max(A)-min(A))*t)
    mu_upper=decimal_ceil_q(mu_plus,1000000)
    if mu_upper>max(A): mu_upper=Q(max(A))
    eta=Q(str(eta)) if not isinstance(eta,Q) else eta
    if min(l)<=0 or not (eta<1/mu_upper):
        return {'N':N,'reported':False,'reason':'no strict positive probability/renewal-limit certificate',
                't_N':t,'lower':None,'upper':None,'K':None}
    cert=certificate(A,l,mu_upper,eta,24)
    if cert['K']>50000:
        # This is a computational guard; never label as a valid confidence
        # interval if the original theorem's entire finite prefix is unscanned.
        return {'N':N,'reported':False,'reason':f'computationally inconclusive (K={cert["K"]})',
                't_N':t,'lower':None,'upper':None,'K':cert['K']}
    u=renewal_float(A,phat,cert['K'])
    # Bounds use float arithmetic for classification. If within 1e-12 of
    # eta, make outward-safe choice (ambiguous -> cannot assert lower bound;
    # ambiguous -> retain in upper). This preserves conservative meaning.
    lo=0;hi=0
    for n in range(1,cert['K']+1):
        e=min(1.,2*(n//min(A))*t)
        if u[n]+e<float(eta)-1e-12: lo=n
        if u[n]-e<float(eta)+1e-12: hi=n
    return {'N':N,'reported':True,'reason':'finite-sample selective false-report bound <= beta',
            't_N':t,'lower':lo,'upper':hi,'K':cert['K'],
            'confidence':confidence,'q':float(cert['q']),
            'delta':float(cert['delta']),'eta':float(eta)}


def confidence_bound_demo(out,rng):
    p=np.array([float(x) for x in P]); Ns=[25000,100000,1000000]
    rows=[]
    for N in Ns:
        counts=rng.multinomial(N,p)
        rec=certified_interval_one(counts,Q(14,100))
        rec.update({'n_5':int(counts[0]),'n_6':int(counts[1]),
                    'n_7':int(counts[2]),'n_8':int(counts[3]),
                    'true_F_inside_reported_interval':(
                        rec['lower']<=21<=rec['upper'] if rec['reported'] else None)})
        rows.append(rec)
    t=pd.DataFrame(rows);write_df(t,'11_rigorous_confidence_interval_examples',out)
    fig,ax=plt.subplots(figsize=(9.8,4.3))
    for i,row in enumerate(rows):
        if row['reported']:
            ax.plot([row['lower'],row['upper']],[i,i],'-',lw=6,color=COLORS['teal'])
            ax.scatter([row['lower'],row['upper']],[i,i],s=33,color=COLORS['ink'])
        else:
            ax.text(0,i,'Inconclusive (no interval reported)',va='center',fontsize=9)
    ax.axvline(21,color=COLORS['red'],ls='--',lw=2,label=r'True $F_{0.14}=21$')
    ax.set_yticks(range(len(Ns)),[f'N = {N:,}' for N in Ns]); ax.invert_yaxis()
    ax.set(xlabel='Certified interval for last threshold violation',
           title='Conservative nonasymptotic inference: finite-sample reporting')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.22),ncol=1)
    save_fig(fig,'Fig11_finite_sample_confidence_bounds',out)
    return t


def selective_reporting_operating_study(out,rng,profile):
    # An honest simulation diagnostic for the paper's selective-reporting
    # finite-sample theorem; never confuse empirical frequencies with proof.
    eta=Q(16,100); truth=33; p=np.array([float(x) for x in P])
    cases=[500,2000,10000]
    R=30 if profile=='quick' else 100
    summary=[]; raw=[]
    for N in cases:
        for rep in range(R):
            counts=rng.multinomial(N,p)
            res=certified_interval_one(counts,eta)
            reported=bool(res['reported'])
            miss=bool(reported and not (res['lower']<=truth<=res['upper']))
            raw.append({'N':N,'replicate':rep+1,'reported':reported,
                        'false_report':miss,'interval_lower':res['lower'],
                        'interval_upper':res['upper'],'K':res['K'],
                        'reason':res['reason']})
        slice_rows=raw[-R:]
        report_rate=sum(r['reported'] for r in slice_rows)/R
        false_rate=sum(r['false_report'] for r in slice_rows)/R
        widths=[r['interval_upper']-r['interval_lower'] for r in slice_rows if r['reported']]
        summary.append({'N':N,'R':R,'eta':float(eta),'true_F_eta':truth,
             'report_probability_estimate':report_rate,
             'joint_false_report_estimate':false_rate,
             'mean_reported_interval_width':np.mean(widths) if widths else np.nan,
             'theoretical_joint_error_upper_bound':0.05,
             'report_rate_mc_se':math.sqrt(report_rate*(1-report_rate)/R),
             'false_rate_mc_se':math.sqrt(false_rate*(1-false_rate)/R)})
    summary=pd.DataFrame(summary)
    write_df(summary,'14_selective_reporting_operating_characteristics',out)
    write_df(pd.DataFrame(raw),'15_selective_reporting_replications',out)
    fig,ax=plt.subplots(figsize=(9.3,4.6))
    ax.errorbar(summary.N,summary.report_probability_estimate,
       yerr=1.96*summary.report_rate_mc_se,fmt='o-',color=COLORS['blue'],capsize=3,
       label='Probability of reporting (Monte Carlo ±1.96 MCSE)')
    ax.plot(summary.N,summary.joint_false_report_estimate,'s--',color=COLORS['red'],
       label='Joint false-report frequency (Monte Carlo)')
    ax.axhline(.05,ls=':',color=COLORS['gold'],
       label='Theoretical bound on joint false-report probability')
    ax.set_ylim(-.05,1.08)
    ax.set(xlabel='Number of observed packet draws N',ylabel='Estimated probability',
        title=r'Finite-sample selective reporting, $\eta=0.16$, true $F_{\eta}=33$')
    ax.legend(loc='center right',fontsize=8.5)
    save_fig(fig,'Fig13_selective_reporting_operating_characteristics',out)
    return summary


def bootstrap_example(out,rng,profile):
    # This is an APPROXIMATE conditional nonparametric/bootstrap interval.
    # It is NOT the manuscript's guaranteed-coverage interval.
    N=3000; Bboot=350 if profile=='quick' else 2000
    p=np.array([float(x) for x in P]); observed=rng.multinomial(N,p)
    obs_p=observed/N
    boot=rng.multinomial(N,obs_p,size=Bboot)/N
    f,c,amb=numeric_thresholds_many(np.vstack([obs_p,boot]),Q(14,100),None)
    obs_F=int(f[0]); boot_F=f[1:]
    lo=float(np.quantile(boot_F,.025)); hi=float(np.quantile(boot_F,.975))
    table=pd.DataFrame({'replicate':np.arange(Bboot)+1,'F_boot':boot_F,
             'N_original':N,'observed_Fhat':obs_F})
    write_df(table,'12_bootstrap_threshold_draws',out)
    summary=pd.DataFrame([{'N':N,'B_boot':Bboot,'observed_counts':str(observed.tolist()),
       'observed_Fhat':obs_F,'bootstrap_percentile_2_5':lo,
       'bootstrap_percentile_97_5':hi,'bootstrap_true_F_covered_in_this_run':lo<=21<=hi,
       'boot_finite_tail_K':c['K'],
       'interpretation':'Approximate/bootstrap: NO unconditional finite-sample coverage guarantee'}])
    write_df(summary,'13_bootstrap_summary',out)
    fig,ax=plt.subplots(figsize=(9.2,4.6))
    hist=Counter(boot_F); keys=np.array(sorted(hist))
    ax.bar(keys,[hist[k]/Bboot for k in keys],color=COLORS['blue'],width=1.6,alpha=.85)
    ax.axvline(21,color=COLORS['red'],ls='--',label='True model value')
    ax.axvline(obs_F,color=COLORS['gold'],ls=':',lw=2,label='Original sample plug-in')
    ax.set(xlabel=r'Bootstrap $\widehat F_{0.14}^{*}$',ylabel='Bootstrap relative frequency',
      title='Parametric multinomial bootstrap (approximate discrete-functional inference)')
    ax.legend();save_fig(fig,'Fig12_bootstrap_threshold_distribution',out)
    return summary


def write_readme(out,profile,seed):
    text=f'''STOCHASTIC FROBENIUS: REPRODUCIBLE EXPERIMENTAL PACKAGE\n
Analysis date (UTC): {datetime.now(timezone.utc).isoformat()}
Run profile: {profile}
Master seed: {seed}

MODEL
- Four fixed, minimal generators: 5,6,7,8.
- IID packet sizes with probabilities .4,.3,.2,.1.
- Exact hitting event: at least one partial sum equals target n.
- True renewal limit: 1/6.
- True deterministic Frobenius number: 9.
- Exactly verified F_eta values: eta (.10,.12,.14,.15,.16) => (9,15,21,27,33).

DESIGN AND CLAIM STRENGTH
A. Tables 01-04 use exact Fraction rational arithmetic and independent
   recurrence / combinatorial enumeration / Markov matrix verification.
   Global matrix-minorization bounds certify that NO LARGER n violates each
   specified threshold. No finite-window extrapolation is used.
B. Figures 01-04 are deterministic model illustrations (not observations).
C. Figure 05 and Table 06 come from independent simulated renewal paths;
   Wilson limits describe Bernoulli Monte Carlo sampling, not uncertainty in p.
D. Figures 06-07 use floating-point recurrence with exact rational universal
   global tail bounds for the stated parameter families. Near-boundary
   classifications are re-evaluated using Fraction arithmetic.
E. Figures 08-10 are independent multinomial simulated training datasets.
   The delta-method normal approximation is asymptotic; threshold sampling
   frequencies do not constitute a guaranteed confidence interval.
F. Table 11 and Figure 13 evaluate finite-sample confidence reporting.
   Table 11 gives a finite-sample confidence reporting interval with the
   manuscript's Hoeffding+coupling guarantee and a sharpened rational
   sub-stochastic matrix tail certificate. Coverage guarantee means
   P(report interval AND miss true F) <= beta; it does NOT guarantee
   P(cover | reported) >= 1-beta. When no interval is reported, the answer
   is INCONCLUSIVE. Numerical endpoint comparisons are outward-safe.
G. Figure 13 estimates the probability of issuing an interval and the
   JOINT false-report probability over simulated datasets. Empirical
   frequencies are not a substitute for mathematical error control.
H. Figure 12 bootstrap percentiles are APPROXIMATE. They do NOT have a
   guaranteed finite-sample coverage theorem for the discontinuous F_eta.

OUTPUTS
- figures/ : 13 manuscript-grade plots as PNG (320dpi), PDF and SVG.
- tables/ : numbered CSV datasets and journal-compatible LaTeX tables.
- verification/: exact rational tail certificate and automatic assertions.
- run_manifest.json: software, parameters, random seeds, file hashes.
- source/: exact execution source and (if available) supplementary manuscript.
- source ZIP: all above in a single archive outside this directory.

NOTE ON SCIENTIFIC INTERPRETATION
This is a synthetic illustration of a precisely specified probabilistic model,
not a measurement of actual McDonald's ordering patterns. Confidence reporting
in the preceding theorem is deliberately conservative. Any proposed journal
submission should include further generator families, theory-audit, sensitivity
analysis, and a careful literature-based novelty statement.
'''
    (out/'README.txt').write_text(text)


def run_pipeline(profile='full',output_root=None,seed=MASTER_SEED,download=False):
    t_start=time.perf_counter()
    root=Path(output_root) if output_root is not None else Path.cwd()
    out=root/OUT_NAME
    if out.exists(): shutil.rmtree(out)
    for directory in ('figures','tables','verification','source'):
        (out/directory).mkdir(parents=True,exist_ok=True)
    make_styles()
    # Separate independent PCG64 streams; adding one experiment does not
    # silently change the random numbers of the other experiments.
    streams=np.random.SeedSequence(seed).spawn(6)
    rngs=[np.random.default_rng(s) for s in streams]
    manifest={'timestamp_utc':datetime.now(timezone.utc).isoformat(),
              'profile':profile,'seed':int(seed),'rng':'numpy PCG64 SeedSequence',
              'python_version':sys.version.split()[0],'numpy_version':np.__version__,
              'scipy_version':__import__('scipy').__version__,
              'pandas_version':pd.__version__,'matplotlib_version':matplotlib.__version__,
              'platform':platform.platform(),
              'model':{'a':list(A),'p':[float(t) for t in P],
                       'etas':[float(t) for t in LEVELS]},'sections':{}}
    print('\n'+'='*78)
    print('STOCHASTIC FROBENIUS: EXACT + SIMULATION + INFERENCE')
    print(f'profile={profile}; seed={seed}; output={out.resolve()}')
    print('='*78)
    print('[1/9] Exact Apéry, rational recurrence, Doeblin certificate, 3 crosschecks')
    base=verify_exact(out)
    print('PASS — 5 exact F_eta values: ',EXPECTED)
    print('[2/9] Deterministic figures, reliability staircase and convergence')
    plot_renewal(base,out);plot_gaps(base,out)
    step=plot_staircase(base,out);write_df(step,'14_staircase_grid',out)
    plot_convergence(base,out)
    print('[3/9] Independent renewal-path Monte Carlo')
    mct=trajectories_mc(base,out,rngs[0],profile)
    print('   Maximum MC absolute error:',f'{np.max(np.abs(mct.mc_error)):.5f}')
    print('[4/9] Probability sensitivity + two-dimensional response surface')
    manifest['sections']['sensitivity']=sensitivity_study(out,profile)
    print('[5/9] Multinomial plug-in estimation and delta-method validation')
    dmt=delta_method_study(out,rngs[1],profile)
    print('[6/9] Sampling distribution of the last below-threshold integer')
    fest=threshold_estimation_study(out,rngs[2],profile)
    print(fest[['N','exact_recovery_rate','MAE']].to_string(index=False))
    print('[7/9] Nonasymptotic certified confidence intervals')
    cint=confidence_bound_demo(out,rngs[3])
    print(cint[['N','reported','lower','upper','K']].to_string(index=False))
    print('[8/9] Empirical operating characteristics of selective reporting')
    reporting_tab=selective_reporting_operating_study(out,rngs[5],profile)
    manifest['sections']['selective_reporting']=reporting_tab.astype(object).where(pd.notnull(reporting_tab),None).to_dict('records')
    print('[9/9] Approximate bootstrap illustration (explicitly NOT certified)')
    boot=bootstrap_example(out,rngs[4],profile)
    write_readme(out,profile,seed)
    source=Path(__file__).resolve() if '__file__' in globals() else None
    if source and source.exists():
        shutil.copy2(source,out/'source'/'stochastic_frobenius_colab.py')
    # Optionally include original paper as supplementary source (if local).
    paper_base=Path('/mnt/data/stochastic_frobenius_4')
    if paper_base.is_dir():
        for name in ('main.tex','ref.bib','main.pdf'):
            if (paper_base/name).exists(): shutil.copy2(paper_base/name,out/'source'/name)
    manifest['elapsed_seconds']=round(time.perf_counter()-t_start,2)
    manifest['figure_count']=len(list((out/'figures').glob('*.png')))
    manifest['table_count']=len(list((out/'tables').glob('*.csv')))
    manifest['sha256_files']={}
    for fp in sorted(out.rglob('*')):
        if fp.is_file() and fp.name!='run_manifest.json':
            manifest['sha256_files'][str(fp.relative_to(out))]=hashlib.sha256(fp.read_bytes()).hexdigest()
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
    zip_path=root/(OUT_NAME+'.zip')
    if zip_path.exists(): zip_path.unlink()
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=8) as z:
        for fp in sorted(out.rglob('*')):
            if fp.is_file(): z.write(fp,arcname=str(fp.relative_to(root)))
    with zipfile.ZipFile(zip_path) as z:
        assert z.testzip() is None
        assert len(z.namelist())>=45
    print('\n'+'='*78)
    print('COMPLETE: All assertions passed.')
    print(f'FIGURES: {manifest["figure_count"]} x (PNG/PDF/SVG)')
    print(f'TABLES : {manifest["table_count"]} x (CSV/LaTeX)')
    print(f'ZIP    : {zip_path.resolve()}  ({zip_path.stat().st_size/1e6:.2f} MB)')
    print('Elapsed seconds:',manifest['elapsed_seconds'])
    print('='*78)
    if download:
        try:
            from google.colab import files
            files.download(str(zip_path))
        except ImportError:
            print('Colab auto-download unavailable; ZIP is saved at',zip_path)
    return zip_path


# ========= SINGLE PLACE TO SET THE COLAB STUDY PROFILE ====================
RUN_PROFILE = 'full'             # 'quick' for code checks; 'full' for paper run
RANDOM_SEED = MASTER_SEED
AUTO_DOWNLOAD_ON_COLAB = True
# ===========================================================================

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=('quick','full'),default=RUN_PROFILE)
    parser.add_argument('--output-root',type=str,default=None)
    parser.add_argument('--seed',type=int,default=RANDOM_SEED)
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--no-download',action='store_true')
    args,unrecognized=parser.parse_known_args()
    in_colab=('google.colab' in sys.modules)
    run_pipeline(args.profile,args.output_root,args.seed,
                 download=(args.download or (in_colab and AUTO_DOWNLOAD_ON_COLAB)) and not args.no_download)
