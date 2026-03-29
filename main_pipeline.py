#!/usr/bin/env python3
"""
================================================================================
PIPELINE CON DATASET SINTÉTICO - DETECCIÓN DE ESPASTICIDAD EN NEONATOS
================================================================================
TFM - Universitat Oberta de Catalunya (UOC)
Autor: Máximo Fernández Riera
Fecha: 07/02/2026

Aborda la limitación principal: ausencia de datos patológicos en el corpus Kaggle.
Genera dataset sintético con movimiento espástico mediante perturbaciones
clínicamente motivadas sobre características reales de movimiento normal.

Clasificación binaria: Normal (0) vs Patológico sintético (1)
================================================================================
"""

import os, sys, time, json, gc, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import joblib
from pathlib import Path
from datetime import datetime
from scipy import stats

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate, learning_curve
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, confusion_matrix,
    classification_report, cohen_kappa_score, matthews_corrcoef,
    log_loss, brier_score_loss, precision_recall_curve,
    average_precision_score
)
from sklearn.calibration import calibration_curve
import xgboost as xgb
import shap

warnings.filterwarnings('ignore')
np.random.seed(42)

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models' / 'synthetic'
FIGURES_DIR = PROJECT_ROOT / 'reports' / 'figures'
RESULTS_DIR = PROJECT_ROOT / 'reports' / 'results'
for d in [MODELS_DIR, FIGURES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_CV_FOLDS = 5
CLASS_NAMES = ['Normal', 'Patológico']
FEATURE_NAMES = [
    'of_mean_mag','of_std_mag','of_max_mag','of_mean_ang','of_std_ang','of_motion_ratio',
    'tmp_mean_int','tmp_std_int','tmp_range_int','tmp_mean_diff','tmp_max_diff','tmp_std_diff',
    'tmp_fft_peak1','tmp_fft_energy_low','tmp_fft_dom_freq','tmp_win10_std','tmp_win10_range',
    'tmp_win25_std','tmp_win25_range',
    'sp_q1_mean','sp_q1_std','sp_q2_mean','sp_q2_std','sp_q3_mean','sp_q3_std',
    'sp_q4_mean','sp_q4_std','sp_symmetry','sp_com_y','sp_com_x'
]
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

def pr(t): print("\n" + "="*78 + f"\n  {t}\n" + "="*78)
def ps(t): print(f"\n--- {t} ---")


# ============================================================================
# FASE 1: GENERACIÓN DEL DATASET SINTÉTICO
# ============================================================================

def generate_synthetic_pathological(X_normal, n_synthetic=None, severity_levels=3):
    """Genera muestras sintéticas de movimiento espástico basadas en literatura clínica."""
    if n_synthetic is None:
        n_synthetic = len(X_normal)
    rng = np.random.RandomState(RANDOM_STATE)
    sigma = X_normal.std(axis=0) + 1e-8
    X_synth = np.zeros((n_synthetic, X_normal.shape[1]))
    spl = n_synthetic // severity_levels

    for lv in range(severity_levels):
        s, e = lv * spl, (lv + 1) * spl if lv < severity_levels - 1 else n_synthetic
        n = e - s
        sev = 1.0 + (lv / max(severity_levels - 1, 1)) * 2.0
        base = X_normal[rng.choice(len(X_normal), n, replace=True)].copy()

        # Flujo óptico (0-5): mov. más bruscos
        base[:, 0] *= (1 + 0.5*sev + rng.normal(0, 0.2, n))
        base[:, 1] *= (1 + 0.8*sev + rng.normal(0, 0.3, n))
        base[:, 2] *= (1 + 0.6*sev + rng.normal(0, 0.25, n))
        base[:, 3] += rng.normal(0, 0.5*sev*sigma[3], n)
        base[:, 4] *= (1 + 0.4*sev + rng.normal(0, 0.2, n))
        base[:, 5] = np.clip(base[:, 5]*(1 + 0.3*sev + rng.normal(0, 0.15, n)), 0, X_normal[:, 5].max()*2)

        # Temporales (6-18): pérdida periodicidad
        base[:, 6] *= (1 + rng.normal(0.2*sev, 0.3, n))
        base[:, 7] *= (1 + 0.6*sev + rng.normal(0, 0.25, n))
        base[:, 8] *= (1 + 0.5*sev + rng.normal(0, 0.2, n))
        for c in [9, 10, 11]:
            base[:, c] *= (1 + 0.7*sev + rng.normal(0, 0.3, n))
        base[:, 12] *= (1 + 0.3*sev + rng.normal(0, 0.2, n))
        base[:, 13] = np.clip(base[:, 13]*(1 - 0.3*sev/3 + rng.normal(0, 0.15, n)), 0, None)
        base[:, 14] *= (1 + 0.4*sev + rng.normal(0, 0.2, n))
        for c in [15, 16, 17, 18]:
            base[:, c] *= (1 + 0.5*sev + rng.normal(0, 0.2, n))

        # Espaciales (19-29): asimetría bilateral
        asym = rng.uniform(0.5, 1.5, n)
        base[:, 19] *= asym
        base[:, 20] *= (1 + 0.4*sev*rng.uniform(0, 1, n))
        base[:, 21] *= (2 - asym)
        base[:, 22] *= (1 + 0.4*sev*rng.uniform(0, 1, n))
        base[:, 23] *= asym*(1 + rng.normal(0, 0.1, n))
        base[:, 24] *= (1 + 0.3*sev*rng.uniform(0, 1, n))
        base[:, 25] *= (2 - asym)*(1 + rng.normal(0, 0.1, n))
        base[:, 26] *= (1 + 0.3*sev*rng.uniform(0, 1, n))
        base[:, 27] = np.clip(base[:, 27]*(1 - 0.3*sev/3 + rng.normal(0, 0.1, n)), 0, None)
        base[:, 28] += rng.normal(0.2*sev*sigma[28], 0.5*sigma[28], n)
        base[:, 29] += rng.normal(0.3*sev*sigma[29], 0.5*sigma[29], n)

        X_synth[s:e] = base

    for c in [0,1,2,5,7,8,10,11,13,15,16,17,18,20,22,24,26]:
        X_synth[:, c] = np.abs(X_synth[:, c])
    return X_synth


def load_and_generate_dataset():
    pr("FASE 1: CARGA DE DATOS Y GENERACIÓN SINTÉTICA")
    features_path = DATA_DIR / 'features.npz'
    if not features_path.exists():
        print("ERROR: features.npz no encontrado"); sys.exit(1)
    data = np.load(features_path)
    X_norm = data['X']; y_orig = data['y']
    print(f"  Normales: {X_norm.shape[0]}, Features: {X_norm.shape[1]}")

    ps("Generación de muestras patológicas sintéticas")
    X_patho = generate_synthetic_pathological(X_norm, n_synthetic=len(X_norm), severity_levels=3)
    print(f"  Sintéticas generadas: {X_patho.shape[0]} (3 niveles severidad)")

    X = np.vstack([X_norm, X_patho])
    y = np.concatenate([np.zeros(len(X_norm), dtype=int), np.ones(len(X_patho), dtype=int)])
    idx = np.random.RandomState(RANDOM_STATE).permutation(len(X))
    X, y = X[idx], y[idx]

    print(f"  Dataset combinado: {X.shape[0]} muestras")
    print(f"  Normal: {np.sum(y==0)}, Patológico: {np.sum(y==1)}")

    sig = sum(1 for i in range(30) if stats.ttest_ind(X[y==0,i], X[y==1,i]).pvalue < 0.001)
    print(f"  Features con p<0.001: {sig}/30")

    np.savez(DATA_DIR / 'features_synthetic_07022026.npz', X=X, y=y,
             X_normal=X_norm, X_pathological=X_patho)
    return X, y, X_norm, X_patho


# ============================================================================
# FASE 2-3: PREPROCESAMIENTO Y ENTRENAMIENTO
# ============================================================================

def preprocess(X, y):
    pr("FASE 2: PREPROCESAMIENTO")
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RANDOM_STATE)
    print(f"  Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    for nm, ys in [('Train',y_train),('Val',y_val),('Test',y_test)]:
        d = np.bincount(ys, minlength=2)/len(ys)
        print(f"  {nm}: Normal={d[0]:.3f}, Pato={d[1]:.3f}")

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_va_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_test)

    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
    X_tr_p = pca.fit_transform(X_tr_s)
    X_va_p = pca.transform(X_va_s)
    X_te_p = pca.transform(X_te_s)
    print(f"  PCA: {X.shape[1]} → {pca.n_components_} ({sum(pca.explained_variance_ratio_)*100:.2f}%)")

    joblib.dump(scaler, MODELS_DIR / 'scaler_synthetic.pkl')
    joblib.dump(pca, MODELS_DIR / 'pca_synthetic.pkl')

    return {'X_train':X_tr_p,'X_val':X_va_p,'X_test':X_te_p,
            'y_train':y_train,'y_val':y_val,'y_test':y_test,
            'X_train_raw':X_train,'X_test_raw':X_test,
            'X_train_scaled':X_tr_s,'X_test_scaled':X_te_s}, scaler, pca


def train_models(splits):
    pr("FASE 3: ENTRENAMIENTO (CLASIFICACIÓN BINARIA)")
    X_tr, y_tr, X_va, y_va = splits['X_train'], splits['y_train'], splits['X_val'], splits['y_val']
    models, times = {}, {}
    cfgs = [
        ('Logistic Regression', LogisticRegression(solver='saga',max_iter=2000,random_state=RANDOM_STATE,n_jobs=-1,class_weight='balanced')),
        ('Random Forest', RandomForestClassifier(n_estimators=200,max_depth=15,min_samples_split=5,min_samples_leaf=2,random_state=RANDOM_STATE,n_jobs=-1,class_weight='balanced')),
        ('SVM', SVC(kernel='rbf',probability=True,random_state=RANDOM_STATE,class_weight='balanced')),
        ('XGBoost', xgb.XGBClassifier(n_estimators=200,learning_rate=0.1,max_depth=6,subsample=0.8,colsample_bytree=0.8,objective='binary:logistic',random_state=RANDOM_STATE,verbosity=0,use_label_encoder=False,eval_metric='logloss'))
    ]
    for name, model in cfgs:
        ps(name)
        t0 = time.time()
        if name == 'XGBoost':
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        else:
            model.fit(X_tr, y_tr)
        times[name] = time.time() - t0
        models[name] = model
        print(f"  Train: {model.score(X_tr,y_tr):.4f}, Val: {model.score(X_va,y_va):.4f}, t={times[name]:.3f}s")
        joblib.dump(model, MODELS_DIR / f'{name.lower().replace(" ","_")}_synthetic.pkl')
    return models, times


# ============================================================================
# FASE 4-5: EVALUACIÓN Y VALIDACIÓN CRUZADA
# ============================================================================

def evaluate_models(models, splits, train_times):
    pr("FASE 4: EVALUACIÓN EN TEST SET")
    X_te, y_te = splits['X_test'], splits['y_test']
    all_res, det = [], {}
    for name, model in models.items():
        ps(name)
        yp = model.predict(X_te)
        ypr = model.predict_proba(X_te)
        acc = accuracy_score(y_te, yp)
        sens = recall_score(y_te, yp)
        spec = recall_score(y_te, yp, pos_label=0)
        prec = precision_score(y_te, yp, zero_division=0)
        f1 = f1_score(y_te, yp)
        f1m = f1_score(y_te, yp, average='macro')
        kappa = cohen_kappa_score(y_te, yp)
        mcc = matthews_corrcoef(y_te, yp)
        try: auc_roc = roc_auc_score(y_te, ypr[:, 1])
        except: auc_roc = 0.0
        ll = log_loss(y_te, ypr)
        brier = brier_score_loss(y_te, ypr[:, 1])
        cm = confusion_matrix(y_te, yp)
        tn,fp,fn,tp = cm.ravel()
        npv = tn/(tn+fn) if (tn+fn)>0 else 0
        ppv = tp/(tp+fp) if (tp+fp)>0 else 0
        mx = np.max(ypr, axis=1)
        errs = y_te != yp
        cgap = (np.mean(mx[~errs]) - np.mean(mx[errs])) if np.any(errs) else np.mean(mx)

        print(f"  Acc={acc:.4f} Sens={sens:.4f} Spec={spec:.4f} F1={f1:.4f}")
        print(f"  AUC={auc_roc:.4f} MCC={mcc:.4f} Brier={brier:.4f}")
        print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")

        row = {'Model':name,'Accuracy':acc,'Sensitivity':sens,'Specificity':spec,
               'Precision':prec,'F1-score':f1,'F1-macro':f1m,'Cohen-Kappa':kappa,
               'MCC':mcc,'AUC-ROC':auc_roc,'Log-Loss':ll,'Brier-Score':brier,
               'PPV':ppv,'NPV':npv,'Avg-Confidence':np.mean(mx),
               'Confidence-Gap':cgap,'Train-Time-s':train_times[name]}
        all_res.append(row)
        det[name] = {'y_pred':yp,'y_proba':ypr,'cm':cm,'metrics':row,
                     'classification_report':classification_report(y_te,yp,target_names=CLASS_NAMES,output_dict=True)}

    df = pd.DataFrame(all_res).sort_values('AUC-ROC', ascending=False)
    df['Rank'] = range(1, len(df)+1)
    ps("RANKING")
    print(df[['Rank','Model','Accuracy','Sensitivity','Specificity','F1-score','AUC-ROC','MCC']].to_string(index=False))
    df.to_csv(RESULTS_DIR / 'model_comparison.csv', index=False)
    return df, det


def cross_validation_analysis(X_scaled, y, pca):
    pr("FASE 5: VALIDACIÓN CRUZADA (5-fold)")
    X_pca = pca.transform(X_scaled)
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ['accuracy','f1','roc_auc','precision','recall']
    cfgs = {
        'Logistic Regression': LogisticRegression(solver='saga',max_iter=2000,random_state=RANDOM_STATE,n_jobs=-1,class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=200,max_depth=15,min_samples_split=5,min_samples_leaf=2,random_state=RANDOM_STATE,n_jobs=-1,class_weight='balanced'),
        'SVM': SVC(kernel='rbf',probability=True,random_state=RANDOM_STATE,class_weight='balanced'),
        'XGBoost': xgb.XGBClassifier(n_estimators=200,learning_rate=0.1,max_depth=6,subsample=0.8,colsample_bytree=0.8,random_state=RANDOM_STATE,verbosity=0,use_label_encoder=False,eval_metric='logloss')
    }
    cv_res = {}
    for name, model in cfgs.items():
        ps(name)
        sc = cross_validate(model, X_pca, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=True)
        cv_res[name] = sc
        print(f"  Acc={sc['test_accuracy'].mean():.4f}±{sc['test_accuracy'].std():.4f}  "
              f"F1={sc['test_f1'].mean():.4f}±{sc['test_f1'].std():.4f}  "
              f"AUC={sc['test_roc_auc'].mean():.4f}  "
              f"Gap={sc['train_accuracy'].mean()-sc['test_accuracy'].mean():.4f}")

    rows = []
    for name, sc in cv_res.items():
        rows.append({'Model':name,
            'CV-Acc-Mean':sc['test_accuracy'].mean(),'CV-Acc-Std':sc['test_accuracy'].std(),
            'CV-F1-Mean':sc['test_f1'].mean(),'CV-F1-Std':sc['test_f1'].std(),
            'CV-AUC-Mean':sc['test_roc_auc'].mean(),'CV-AUC-Std':sc['test_roc_auc'].std(),
            'CV-Sens-Mean':sc['test_recall'].mean(),
            'Train-Acc-Mean':sc['train_accuracy'].mean(),
            'Overfit-Gap':sc['train_accuracy'].mean()-sc['test_accuracy'].mean()})
    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(RESULTS_DIR / 'cross_validation.csv', index=False)
    return cv_res, cv_df


# ============================================================================
# FASE 6: VISUALIZACIONES (14 figuras)
# ============================================================================

def generate_visualizations(models, splits, results_df, det, pca, scaler,
                            cv_res, X_raw, y_all, X_norm, X_patho):
    pr("FASE 6: VISUALIZACIONES")
    X_te, y_te = splits['X_test'], splits['y_test']
    plt.rcParams.update({'font.size':11,'axes.titlesize':13,'axes.labelsize':11,
        'figure.dpi':150,'savefig.dpi':300,'savefig.bbox':'tight','font.family':'serif',
        'axes.spines.top':False,'axes.spines.right':False})
    C = ['#1976D2','#388E3C','#F57C00','#D32F2F']

    # FIG 1: Distribuciones Normal vs Patológico
    print("  [1/14] Distribuciones...")
    fig,axes = plt.subplots(2,3,figsize=(16,10))
    feats = {'of_mean_mag':0,'of_std_mag':1,'tmp_std_diff':11,
             'tmp_fft_dom_freq':14,'sp_symmetry':27,'sp_com_x':29}
    for ax,(fn,ci) in zip(axes.flat,feats.items()):
        ax.hist(X_norm[:,ci],bins=40,alpha=0.6,color=C[0],label='Normal',density=True,edgecolor='white',lw=0.5)
        ax.hist(X_patho[:,ci],bins=40,alpha=0.6,color=C[3],label='Patológico',density=True,edgecolor='white',lw=0.5)
        ax.set_title(fn,fontweight='bold',fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.2)
    plt.suptitle('Distribución Normal vs. Patológico sintético',fontsize=14,fontweight='bold')
    plt.tight_layout(); plt.savefig(FIGURES_DIR/'fig01_feature_distributions.png'); plt.close()

    # FIG 2: Comparación métricas
    print("  [2/14] Métricas...")
    fig,axes = plt.subplots(1,2,figsize=(15,5.5))
    df_p = results_df.sort_values('Rank'); x = np.arange(len(df_p)); w = 0.18
    for i,m in enumerate(['Accuracy','Sensitivity','Specificity','F1-score']):
        axes[0].bar(x+i*w,df_p[m],w,label=m,color=C[i],alpha=0.85)
    axes[0].set_xticks(x+w*1.5); axes[0].set_xticklabels(df_p['Model'],rotation=20,ha='right')
    axes[0].set_ylim(0.5,1.05); axes[0].legend(fontsize=8); axes[0].grid(axis='y',alpha=0.2)
    axes[0].set_title('Métricas de clasificación binaria')
    for i,m in enumerate(['AUC-ROC','Cohen-Kappa','MCC']):
        axes[1].bar(x+i*0.22,df_p[m],0.22,label=m,color=['#7B1FA2','#00796B','#5D4037'][i],alpha=0.85)
    axes[1].set_xticks(x+0.22); axes[1].set_xticklabels(df_p['Model'],rotation=20,ha='right')
    axes[1].set_ylim(0.5,1.05); axes[1].legend(fontsize=8); axes[1].grid(axis='y',alpha=0.2)
    axes[1].set_title('Métricas avanzadas')
    plt.tight_layout(); plt.savefig(FIGURES_DIR/'fig02_model_comparison.png'); plt.close()

    # FIG 3: Matrices confusión
    print("  [3/14] Matrices confusión...")
    fig,axes = plt.subplots(2,2,figsize=(12,10))
    for name,ax in zip(models.keys(),axes.flat):
        cm = det[name]['cm']
        pct = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]*100
        ann = np.array([[f'{cm[i,j]}\n({pct[i,j]:.1f}%)' for j in range(2)] for i in range(2)])
        sns.heatmap(cm,annot=ann,fmt='',cmap='Blues',ax=ax,xticklabels=CLASS_NAMES,
                    yticklabels=CLASS_NAMES,linewidths=1,linecolor='white')
        ax.set_title(name,fontweight='bold'); ax.set_xlabel('Predicción'); ax.set_ylabel('Real')
    plt.suptitle('Matrices de confusión',fontsize=14,fontweight='bold',y=1.02)
    plt.tight_layout(); plt.savefig(FIGURES_DIR/'fig03_confusion_matrices.png'); plt.close()

    # FIG 4: ROC
    print("  [4/14] ROC...")
    fig,ax = plt.subplots(figsize=(8,7))
    for i,(name,_) in enumerate(models.items()):
        fpr,tpr,_ = roc_curve(y_te,det[name]['y_proba'][:,1])
        ax.plot(fpr,tpr,color=C[i],lw=2.5,label=f'{name} (AUC={auc(fpr,tpr):.4f})')
    ax.plot([0,1],[0,1],'k--',lw=1.5,alpha=0.4); ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('Curvas ROC',fontweight='bold'); ax.legend(fontsize=9); ax.grid(alpha=0.2)
    plt.tight_layout(); plt.savefig(FIGURES_DIR/'fig04_roc_curves.png'); plt.close()

    # FIG 5: Precision-Recall
    print("  [5/14] PR...")
    fig,ax = plt.subplots(figsize=(8,7))
    for i,(name,_) in enumerate(models.items()):
        p,r,_ = precision_recall_curve(y_te,det[name]['y_proba'][:,1])
        ap = average_precision_score(y_te,det[name]['y_proba'][:,1])
        ax.plot(r,p,color=C[i],lw=2.5,label=f'{name} (AP={ap:.4f})')
    bl = np.sum(y_te==1)/len(y_te)
    ax.axhline(bl,color='gray',ls='--',alpha=0.5,label=f'Prevalencia ({bl:.3f})')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_title('Precision-Recall',fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.2)
    plt.tight_layout(); plt.savefig(FIGURES_DIR/'fig05_precision_recall.png'); plt.close()

    # FIG 6: PCA
    print("  [6/14] PCA...")
    fig,axes = plt.subplots(1,2,figsize=(14,5.5))
    cv_ = np.cumsum(pca.explained_variance_ratio_)
    axes[0].bar(range(1,len(pca.explained_variance_ratio_)+1),pca.explained_variance_ratio_,color=C[0],alpha=0.7)
    axes[0].plot(range(1,len(cv_)+1),cv_,'ro-',ms=5,lw=2); axes[0].axhline(0.95,color='green',ls='--')
    axes[0].set_xlabel('PC'); axes[0].set_ylabel('Varianza'); axes[0].set_title('Varianza PCA')
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True)); axes[0].grid(alpha=0.2)
    X2 = splits['X_test'][:,:2]
    axes[1].scatter(X2[y_te==0,0],X2[y_te==0,1],c=C[0],alpha=0.6,s=30,label='Normal')
    axes[1].scatter(X2[y_te==1,0],X2[y_te==1,1],c=C[3],alpha=0.6,s=30,label='Patológico')
    axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2'); axes[1].legend(); axes[1].grid(alpha=0.2)
    axes[1].set_title('Proyección PCA')
    plt.tight_layout(); plt.savefig(FIGURES_DIR/'fig06_pca_analysis.png'); plt.close()

    # FIG 7: Learning curves
    print("  [7/14] Learning curves...")
    X_all_pca = pca.transform(scaler.transform(X_raw))
    fig,axes = plt.subplots(2,2,figsize=(14,10))
    lc_cfgs = [
        ('Logistic Regression',LogisticRegression(solver='saga',max_iter=2000,random_state=RANDOM_STATE,n_jobs=-1,class_weight='balanced')),
        ('Random Forest',RandomForestClassifier(n_estimators=200,max_depth=15,random_state=RANDOM_STATE,n_jobs=-1,class_weight='balanced')),
        ('SVM',SVC(kernel='rbf',probability=True,random_state=RANDOM_STATE,class_weight='balanced')),
        ('XGBoost',xgb.XGBClassifier(n_estimators=200,learning_rate=0.1,max_depth=6,random_state=RANDOM_STATE,verbosity=0,use_label_encoder=False,eval_metric='logloss'))
    ]
    for idx,(name,model) in enumerate(lc_cfgs):
        ax = axes.flat[idx]
        try:
            tsz,trsc,vlsc = learning_curve(model,X_all_pca,y_all,train_sizes=np.linspace(0.1,1,8),cv=3,scoring='accuracy',n_jobs=-1,random_state=RANDOM_STATE)
            ax.plot(tsz,trsc.mean(1),'o-',color=C[0],lw=2,label='Train')
            ax.fill_between(tsz,trsc.mean(1)-trsc.std(1),trsc.mean(1)+trsc.std(1),alpha=0.12,color=C[0])
            ax.plot(tsz,vlsc.mean(1),'o-',color=C[3],lw=2,label='Val')
            ax.fill_between(tsz,vlsc.mean(1)-vlsc.std(1),vlsc.mean(1)+vlsc.std(1),alpha=0.12,color=C[3])
        except Exception as e:
            ax.text(0.5,0.5,str(e)[:50],transform=ax.transAxes,ha='center')
        ax.set_title(name,fontweight='bold'); ax.set_xlabel('Muestras'); ax.set_ylabel('Accuracy')
        ax.legend(fontsize=8); ax.grid(alpha=0.2); ax.set_ylim(0.5,1.05)
    plt.suptitle('Curvas de aprendizaje',fontsize=14,fontweight='bold')
    plt.tight_layout(); plt.savefig(FIGURES_DIR/'fig07_learning_curves.png'); plt.close()

    # FIG 8: CV boxplots
    print("  [8/14] CV boxplots...")
    fig,axes = plt.subplots(1,3,figsize=(16,5.5))
    for ax,(k,lb) in zip(axes,[('test_accuracy','Accuracy'),('test_f1','F1'),('test_roc_auc','AUC-ROC')]):
        data = [cv_res[n][k] for n in cv_res]
        bp = ax.boxplot(data,labels=list(cv_res.keys()),patch_artist=True,widths=0.5,showmeans=True,
                        meanprops=dict(marker='D',markerfacecolor='red',markersize=6))
        for p,c in zip(bp['boxes'],C): p.set_facecolor(c); p.set_alpha(0.5)
        ax.set_title(lb,fontweight='bold'); ax.grid(axis='y',alpha=0.2)
        plt.setp(ax.get_xticklabels(),rotation=20,ha='right')
    plt.suptitle('Validación cruzada 5-fold',fontsize=14,fontweight='bold')
    plt.tight_layout(); plt.savefig(FIGURES_DIR/'fig08_cv_boxplots.png'); plt.close()

    # FIG 9: Confianza
    print("  [9/14] Confianza...")
    fig,axes = plt.subplots(2,2,figsize=(14,10))
    for name,ax in zip(models.keys(),axes.flat):
        mx = np.max(det[name]['y_proba'],axis=1); ok = y_te==det[name]['y_pred']
        ax.hist(mx[ok],bins=30,alpha=0.6,color='#388E3C',label='Correctas',density=True)
        if np.any(~ok): ax.hist(mx[~ok],bins=15,alpha=0.6,color=C[3],label='Errores',density=True)
        ax.axvline(mx.mean(),color='k',ls='--',label=f'μ={mx.mean():.3f}')
        ax.set_title(name,fontweight='bold'); ax.legend(fontsize=8); ax.grid(alpha=0.2)
    plt.suptitle('Distribución de confianza',fontsize=14,fontweight='bold')
    plt.tight_layout(); plt.savefig(FIGURES_DIR/'fig09_confidence.png'); plt.close()

    # FIG 10: Calibración
    print("  [10/14] Calibración...")
    fig,ax = plt.subplots(figsize=(8,7))
    ax.plot([0,1],[0,1],'k--',alpha=0.4)
    for i,(name,_) in enumerate(models.items()):
        yp1 = det[name]['y_proba'][:,1]
        try:
            pt,pp = calibration_curve(y_te,yp1,n_bins=10)
            ax.plot(pp,pt,'o-',color=C[i],lw=2,label=f'{name} (Brier={brier_score_loss(y_te,yp1):.4f})')
        except: pass
    ax.set_xlabel('P predicha'); ax.set_ylabel('Fracción positivos'); ax.set_title('Calibración',fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.2)
    plt.tight_layout(); plt.savefig(FIGURES_DIR/'fig10_calibration.png'); plt.close()

    # FIG 11: Heatmap
    print("  [11/14] Heatmap...")
    fig,ax = plt.subplots(figsize=(14,4.5))
    cols = ['Accuracy','Sensitivity','Specificity','F1-score','Cohen-Kappa','MCC','AUC-ROC','Brier-Score']
    hd = results_df.set_index('Model')[cols].sort_values('AUC-ROC',ascending=False)
    sns.heatmap(hd,annot=True,fmt='.4f',cmap='RdYlGn',ax=ax,linewidths=1,vmin=0,vmax=1)
    ax.set_title('Resumen de métricas',fontweight='bold')
    plt.tight_layout(); plt.savefig(FIGURES_DIR/'fig11_heatmap.png'); plt.close()

    # FIG 12: Umbral
    print("  [12/14] Umbral...")
    fig,axes = plt.subplots(1,2,figsize=(14,5.5))
    best = results_df.iloc[0]['Model']
    yp1 = det[best]['y_proba'][:,1]
    thr = np.linspace(0.01,0.99,100)
    se = [recall_score(y_te,(yp1>=t).astype(int),zero_division=0) for t in thr]
    sp = [recall_score(y_te,(yp1>=t).astype(int),pos_label=0,zero_division=0) for t in thr]
    f1t = [f1_score(y_te,(yp1>=t).astype(int),zero_division=0) for t in thr]
    axes[1].plot(thr,se,color=C[3],lw=2,label='Sensibilidad')
    axes[1].plot(thr,sp,color=C[0],lw=2,label='Especificidad')
    axes[1].plot(thr,f1t,color=C[1],lw=2,ls='--',label='F1')
    oi = np.argmax(f1t)
    axes[1].axvline(thr[oi],color='gray',ls=':',label=f'Óptimo ({thr[oi]:.2f})')
    axes[1].set_xlabel('Umbral'); axes[1].set_title(f'Sens/Spec vs Umbral - {best}',fontweight='bold')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.2)
    # Terciles
    pm = y_te==1; pp = det[best]['y_proba'][pm,1]
    tc = np.percentile(pp,[33.3,66.6])
    for g,lb in enumerate(['Baja','Media','Alta']):
        mg = np.digitize(pp,tc)==g
        if mg.sum()>0: axes[0].hist(pp[mg],bins=15,alpha=0.5,label=f'{lb} (n={mg.sum()})',density=True)
    axes[0].set_xlabel('P(Patológico)'); axes[0].set_title('Distribución por tercil',fontweight='bold')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.2)
    plt.tight_layout(); plt.savefig(FIGURES_DIR/'fig12_threshold.png'); plt.close()

    # FIG 13: Métricas clínicas
    print("  [13/14] Clínicas...")
    fig,ax = plt.subplots(figsize=(10,6))
    cm_ = ['Sensitivity','Specificity','PPV','NPV']; xc = np.arange(len(cm_))
    for i,(_,row) in enumerate(results_df.sort_values('Rank').iterrows()):
        ax.bar(xc+i*0.18,[row[m] for m in cm_],0.18,label=row['Model'],color=C[i],alpha=0.85)
    ax.set_xticks(xc+0.27); ax.set_xticklabels(['Sensibilidad','Especificidad','VPP','VPN'])
    ax.set_ylim(0.5,1.05); ax.legend(fontsize=8); ax.grid(axis='y',alpha=0.2)
    ax.set_title('Métricas clínicas',fontweight='bold')
    plt.tight_layout(); plt.savefig(FIGURES_DIR/'fig13_clinical.png'); plt.close()

    # FIG 14: Tiempos
    print("  [14/14] Tiempos...")
    fig,ax = plt.subplots(figsize=(8,5))
    df_s = results_df.sort_values('Train-Time-s')
    bars = ax.barh(df_s['Model'],df_s['Train-Time-s'],color=C[:len(df_s)],alpha=0.85)
    for b,t in zip(bars,df_s['Train-Time-s']): ax.text(b.get_width()+0.01,b.get_y()+b.get_height()/2,f'{t:.3f}s',va='center')
    ax.set_xlabel('Tiempo (s)'); ax.set_title('Tiempos de entrenamiento',fontweight='bold'); ax.grid(axis='x',alpha=0.2)
    plt.tight_layout(); plt.savefig(FIGURES_DIR/'fig14_times.png'); plt.close()

    print(f"\n  ✅ 14 figuras en {FIGURES_DIR}/")


# ============================================================================
# FASE 7: SHAP
# ============================================================================

def shap_analysis(models, splits, pca):
    pr("FASE 7: ANÁLISIS SHAP")
    X_te = splits['X_test']; n_pcs = X_te.shape[1]
    pc_names = [f'PC{i+1}' for i in range(n_pcs)]
    shap_res = {}
    C = ['#1976D2','#388E3C']
    for name in ['Random Forest','XGBoost']:
        ps(f"SHAP - {name}")
        try:
            exp = shap.TreeExplainer(models[name])
            Xe = X_te[:min(100,len(X_te))]
            sv = exp.shap_values(Xe)
            if isinstance(sv, list): sv1 = sv[1] if len(sv)>1 else sv[0]
            elif sv.ndim == 3: sv1 = sv[:,:,1]
            else: sv1 = sv
            imp = np.abs(sv1).mean(0); imp_n = imp/imp.sum()
            top = np.argsort(imp_n)[-10:][::-1]
            for r,idx in enumerate(top,1): print(f"    {r}. PC{idx+1}: {imp_n[idx]:.4f}")
            shap_res[name] = {'sv1':sv1,'importance':imp_n,'top':top,'Xe':Xe}
        except Exception as e: print(f"  Error: {e}")

    if shap_res:
        fig,axes = plt.subplots(1,2,figsize=(14,6))
        for i,(name,res) in enumerate(shap_res.items()):
            ax = axes[i]; top = res['top']
            ax.barh(range(len(top)),res['importance'][top][::-1],color=C[i],alpha=0.8)
            ax.set_yticks(range(len(top))); ax.set_yticklabels([f'PC{j+1}' for j in top[::-1]])
            ax.set_xlabel('SHAP'); ax.set_title(name,fontweight='bold'); ax.grid(axis='x',alpha=0.2)
        plt.suptitle('Importancia SHAP - Clase Patológico',fontsize=14,fontweight='bold')
        plt.tight_layout(); plt.savefig(FIGURES_DIR/'fig15_shap_importance.png'); plt.close()

        best_t = list(shap_res.keys())[0]; res = shap_res[best_t]
        try:
            fig,_ = plt.subplots(figsize=(10,8))
            se = shap.Explanation(values=res['sv1'],data=res['Xe'],feature_names=pc_names[:res['Xe'].shape[1]])
            shap.plots.beeswarm(se,show=False,max_display=15)
            plt.title(f'SHAP Beeswarm - {best_t}',fontweight='bold')
            plt.tight_layout(); plt.savefig(FIGURES_DIR/'fig16_shap_beeswarm.png'); plt.close()
        except Exception as e: print(f"  Beeswarm error: {e}")
        print("  ✅ SHAP figs guardadas")
    return shap_res


# ============================================================================
# FASE 8: EXPORTAR
# ============================================================================

def export_results(results_df, det, cv_df, pca, splits, train_times, total_time):
    pr("FASE 8: EXPORTACIÓN")
    best = results_df.iloc[0]
    summary = {
        'timestamp': TIMESTAMP, 'experiment': 'Synthetic pathological', 'date': '07/02/2026',
        'dataset': {
            'total': len(splits['y_train'])+len(splits['y_val'])+len(splits['y_test']),
            'n_features_raw': 30, 'n_features_pca': int(pca.n_components_),
            'pca_variance': float(sum(pca.explained_variance_ratio_)),
            'synthetic_method': '3-level clinical perturbation'
        },
        'splits': {k: len(splits[f'y_{k}']) for k in ['train','val','test']},
        'best_model': {k: float(best[k]) if isinstance(best[k],float) else best[k]
                       for k in ['Model','Accuracy','Sensitivity','Specificity','F1-score','AUC-ROC','MCC']},
        'all_models': results_df.to_dict(orient='records'),
        'cross_validation': cv_df.to_dict(orient='records'),
        'total_time_s': total_time, 'train_times': train_times,
        'best_per_class': det[best['Model']]['classification_report']
    }
    with open(RESULTS_DIR/'full_results.json','w') as f: json.dump(summary,f,indent=2,default=str)
    for name in det:
        cm = det[name]['cm']
        pd.DataFrame(cm,index=['Real_Normal','Real_Pato'],columns=['Pred_Normal','Pred_Pato']).to_csv(
            RESULTS_DIR/f'cm_{name.lower().replace(" ","_")}.csv')
    pd.DataFrame(pca.components_,columns=FEATURE_NAMES[:pca.components_.shape[1]],
                 index=[f'PC{i+1}' for i in range(pca.n_components_)]).to_csv(RESULTS_DIR/'pca_loadings.csv')
    print(f"  ✅ Resultados en {RESULTS_DIR}/")
    for f in sorted(RESULTS_DIR.iterdir()): print(f"    - {f.name}")
    return summary


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("\n" + "█"*78)
    print("█  PIPELINE SINTÉTICO - ML DETECCIÓN ESPASTICIDAD INFANTIL              █")
    print("█"*78)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    X, y, Xn, Xp = load_and_generate_dataset()
    splits, scaler, pca = preprocess(X, y)
    models, times = train_models(splits)
    results_df, det = evaluate_models(models, splits, times)
    X_sc = scaler.transform(X)
    cv_res, cv_df = cross_validation_analysis(X_sc, y, pca)
    generate_visualizations(models, splits, results_df, det, pca, scaler, cv_res, X, y, Xn, Xp)
    shap_res = shap_analysis(models, splits, pca)
    total = time.time() - t0
    summary = export_results(results_df, det, cv_df, pca, splits, times, total)

    pr("RESUMEN EJECUTIVO")
    b = results_df.iloc[0]
    print(f"""
  DATASET: Sintético (767 normales + 767 patológicos)
    Features: 30 → PCA: {pca.n_components_} | Varianza: {sum(pca.explained_variance_ratio_)*100:.1f}%

  DIVISIÓN: Train={len(splits['y_train'])} | Val={len(splits['y_val'])} | Test={len(splits['y_test'])}

  MEJOR MODELO: {b['Model']}
    Accuracy:      {b['Accuracy']:.4f}
    Sensibilidad:  {b['Sensitivity']:.4f}
    Especificidad: {b['Specificity']:.4f}
    F1-score:      {b['F1-score']:.4f}
    AUC-ROC:       {b['AUC-ROC']:.4f}
    MCC:           {b['MCC']:.4f}

  VALIDACIÓN CRUZADA:
{cv_df[['Model','CV-Acc-Mean','CV-Acc-Std','CV-AUC-Mean','Overfit-Gap']].to_string(index=False)}

  TIEMPO: {total:.1f}s ({total/60:.1f}min)
  FIGURAS: {len(list(FIGURES_DIR.glob('*.png')))}
  ✅ COMPLETADO - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")


if __name__ == '__main__':
    main()
