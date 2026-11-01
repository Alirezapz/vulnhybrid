"""Fit registered graph/text/static combinations on validation fitting only.
Labels from test and external frames are consulted only after every recipe fits.
"""
from pathlib import Path
import json,time
import numpy as np
from scipy.special import expit,logit,softmax
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier


def logits(p):return logit(np.clip(p,1e-5,1-1e-5))


def compose(frames,fit,labels,quality):
    phases=list(frames);v=frames['validation'];base=list(v.columns)
    neural=[n for n in base if n.startswith('g_') and n.endswith('_s42')]
    graph=neural+[n for n in base if n.startswith('k_')]
    extended=graph+[n for n in base if n.startswith('x_')]
    text=[n for n in ['v1_token_lr','v1_char_lr','v1_token_svm_platt','v1_token_nb'] if n in base]
    static=['v1_static_lr']
    informative=[n for n in base if not n.startswith('repeat_') and np.std(v[n].to_numpy()[fit])>1e-8]
    # Repeat seeds remain separate measurements and an explicitly named ensemble.
    output={key:{n:df[n].to_numpy().copy() for n in base} for key,df in frames.items()};recipes={}
    def register(name,ps,detail):
        for key in phases:output[key][name]=np.asarray(ps[key],float)
        recipes[name]=detail
    def mean(names):return {key:frames[key][names].to_numpy().mean(axis=1) for key in phases}
    register('fusion_neural_mean',mean(neural),dict(kind='mean',experts=neural))
    register('fusion_all_graph_mean',mean(extended),dict(kind='mean',experts=extended))
    register('fusion_all_expert_mean',mean(informative),dict(kind='mean',experts=informative))
    pg=mean(extended);pl=mean(text);ps=mean(static)
    family={key:np.column_stack([pg[key],pl[key],ps[key]]) for key in phases}
    register('fusion_family_balanced_mean',{key:z.mean(axis=1) for key,z in family.items()},dict(kind='equal_family_mean'))
    # Fixed meta capacity prevents tuning to new test outcomes.
    groups={'fusion_graph_stack':graph,'fusion_extended_graph_stack':extended,
            'fusion_graph_text_stack':extended+text,'fusion_graph_static_stack':extended+static,
            'fusion_all_logit_stack':informative,'fusion_all_probability_stack':informative,
            'fusion_all_rank_stack':informative}
    for name,names in groups.items():
        raw={key:frames[key][names].to_numpy() for key in phases}
        if 'rank' in name:
            reference=np.sort(raw['validation'][fit],axis=0)
            z={key:np.column_stack([(np.searchsorted(reference[:,j],x[:,j],side='right')+.5)/(len(reference)+1) for j in range(x.shape[1])]) for key,x in raw.items()}
        elif 'probability' in name:z=raw
        else:z={key:logits(x) for key,x in raw.items()}
        scaler=StandardScaler().fit(z['validation'][fit]);model=LogisticRegression(C=.1,solver='liblinear',max_iter=1000,random_state=42)
        model.fit(scaler.transform(z['validation'][fit]),labels[fit])
        register(name,{key:model.predict_proba(scaler.transform(x))[:,1] for key,x in z.items()},
            dict(kind='heldout_logistic',C=.1,experts=names,coefficients=model.coef_.tolist(),intercept=model.intercept_.tolist()))
    hgb=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,max_depth=3,l2_regularization=10,early_stopping=False,random_state=42)
    hgb.fit(v[informative].to_numpy()[fit],labels[fit])
    register('fusion_all_boosting',{key:hgb.predict_proba(frames[key][informative].to_numpy())[:,1] for key in phases},dict(kind='heldout_hist_gradient_boost',experts=informative))
    # Learned convex mixture is a control for flexible stacking.
    x=v[informative].to_numpy()[fit];y=labels[fit]
    def convex_loss(w):
        p=np.clip(x@w,1e-6,1-1e-6)
        loss=-(y*np.log(p)+(1-y)*np.log1p(-p)).mean()+.01*np.dot(w,w)
        grad=x.T@((p-y)/(p*(1-p)))/len(y)+.02*w
        return loss,grad
    opt=minimize(convex_loss,np.full(len(informative),1/len(informative)),jac=True,method='SLSQP',
        bounds=[(0,1)]*len(informative),constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones_like(w)},options={'maxiter':500,'ftol':1e-9})
    if not opt.success:raise RuntimeError('Convex fusion failed: '+opt.message)
    register('fusion_convex_all',{key:frames[key][informative].to_numpy()@opt.x for key in phases},
             dict(kind='convex_logloss',weights=dict(zip(informative,opt.x.tolist()))))
    # Fixed lexical anchor plus shrunken graph corrections, with intercept-only control.
    anchor={key:logits(frames[key]['v1_mean_token_char'].to_numpy()) for key in phases}
    for name,experts in [('fusion_graph_residual',extended),('fusion_all_residual',informative),('fusion_intercept_control',[])]:
        design={key:np.column_stack([np.ones(len(df)),*(logits(df[e].to_numpy())-anchor[key] for e in experts)]) for key,df in frames.items()}
        z=design['validation'][fit];a=anchor['validation'][fit];lam=10.
        def residual_loss(beta):
            eta=a+z@beta;p=expit(eta)
            return np.logaddexp(0,eta).sum()-y@eta+.5*lam*np.dot(beta,beta),z.T@(p-y)+lam*beta
        opt=minimize(residual_loss,np.zeros(z.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':1000})
        if not opt.success:raise RuntimeError('Residual fusion did not converge')
        register(name,{key:expit(anchor[key]+z@opt.x) for key,z in design.items()},dict(kind='anchored_residual',l2=lam,experts=experts,coefficients=opt.x.tolist()))
    # Gating uses confidence, disagreement and observed frontend quality only.
    features={}
    for key,f in family.items():
        features[key]=np.column_stack([np.ones(len(f)),f*(1-f),np.abs(f[:,0]-f[:,1]),
                                      np.std(f,axis=1),quality[key]])
    z=features['validation'][fit];experts=family['validation'][fit];shape=(z.shape[1],3)
    def gate_loss(flat):
        w=flat.reshape(shape);g=softmax(z@w,axis=1);p=np.clip(np.sum(g*experts,axis=1),1e-6,1-1e-6)
        loss=-(y*np.log(p)+(1-y)*np.log1p(-p)).sum()+5*np.dot(flat,flat)
        dp=((p-y)/(p*(1-p)))[:,None]
        gradient=z.T@(dp*g*(experts-p[:,None]))+10*w
        return loss,gradient.ravel()
    opt=minimize(gate_loss,np.zeros(np.prod(shape)),jac=True,method='L-BFGS-B',options={'maxiter':1000})
    if not opt.success:raise RuntimeError('Confidence/quality gate did not converge')
    register('fusion_confidence_quality_gate',
        {key:np.sum(softmax(features[key]@opt.x.reshape(shape),axis=1)*family[key],axis=1) for key in phases},
        dict(kind='confidence_quality_mixture',families=['graph','text','static'],coefficients=opt.x.reshape(shape).tolist()))
    register('fusion_quality_fallback',{key:np.where(np.any(quality[key]>0,axis=1),pl[key],.5*(pg[key]+pl[key])) for key in phases},
             dict(kind='quality_fallback',rule='text on parse/truncation flags, graph/text mean otherwise'))
    return output,recipes
