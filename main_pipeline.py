#!/usr/bin/env python3
"""
================================================================================
PIPELINE CON DATASET SINTÉTICO - DETECCIÓN DE ESPASTICIDAD EN NEONATOS
================================================================================
TFM - Universitat Oberta de Catalunya (UOC)
Autor: Máximo Fernández Riera
Fecha: 03/04/2026

Versión centrada exclusivamente en perturbación sobre vídeo crudo.
Genera dataset sintético con movimiento espástico mediante perturbaciones
clínicamente motivadas aplicadas sobre vídeos normales antes de la extracción
de características.

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
import cv2
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
MODELS_BASE_DIR = PROJECT_ROOT / 'models'
REPORTS_DIR = PROJECT_ROOT / 'reports'
MODELS_DIR = MODELS_BASE_DIR / 'synthetic_video'
FIGURES_DIR = REPORTS_DIR / 'figures_video'
RESULTS_DIR = REPORTS_DIR / 'results_video'
RAW_VIDEO_CANDIDATES = [
    PROJECT_ROOT / 'data' / 'raw',
    PROJECT_ROOT.parent / 'data' / 'raw',
    PROJECT_ROOT.parent / 'FUENTE' / 'kaggle_data'
]
CURRENT_MODE = 'raw_video'
for d in [MODELS_BASE_DIR, REPORTS_DIR, MODELS_DIR, FIGURES_DIR, RESULTS_DIR]:
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


def configure_output_dirs():
    global MODELS_DIR, FIGURES_DIR, RESULTS_DIR, CURRENT_MODE
    CURRENT_MODE = 'raw_video'
    MODELS_DIR = MODELS_BASE_DIR / 'synthetic_video'
    FIGURES_DIR = REPORTS_DIR / 'figures_video'
    RESULTS_DIR = REPORTS_DIR / 'results_video'
    for d in [MODELS_DIR, FIGURES_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def find_raw_video_dir():
    for candidate in RAW_VIDEO_CANDIDATES:
        if (candidate / 'data_100_50_50.npz').exists() and (candidate / 'target_100_50_50.npz').exists():
            return candidate
    return None


# ============================================================================
# FASE 1: GENERACIÓN DEL DATASET SINTÉTICO
# ============================================================================

def load_raw_video_corpus():
    raw_dir = find_raw_video_dir()
    if raw_dir is None:
        raise FileNotFoundError('No se encontró el corpus de vídeo crudo en ninguna ruta candidata.')
    print(f"  Directorio de vídeo crudo: {raw_dir}")
    videos = np.load(raw_dir / 'data_100_50_50.npz')['arr_0'].astype(np.uint8)
    labels = np.load(raw_dir / 'target_100_50_50.npz')['arr_0']
    print(f"  Vídeos cargados: {videos.shape} | Labels originales: {labels.shape}")
    return videos, labels


def ensure_uint8_video(video):
    if video.dtype == np.uint8:
        return video
    if np.max(video) <= 1.0:
        video = video * 255.0
    return np.clip(video, 0, 255).astype(np.uint8)


def extract_optical_flow(video):
    frames_gray = []
    for frame in video:
        f = ensure_uint8_video(frame)
        if len(f.shape) == 3 and f.shape[2] == 3:
            gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        else:
            gray = f if len(f.shape) == 2 else f[:, :, 0]
        frames_gray.append(gray)
    if len(frames_gray) < 2:
        return np.zeros(6)
    flows = []
    for i in range(len(frames_gray) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            frames_gray[i], frames_gray[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(flow)
    flows = np.array(flows)
    mag = np.sqrt(flows[:,:,:,0]**2 + flows[:,:,:,1]**2)
    ang = np.arctan2(flows[:,:,:,1], flows[:,:,:,0])
    return np.array([
        np.mean(mag), np.std(mag), np.max(mag),
        np.mean(np.abs(ang)), np.std(ang),
        np.sum(mag > 1.0) / mag.size
    ])


def extract_temporal(video):
    intensity = np.mean(video, axis=(1, 2, 3))
    diffs = np.diff(intensity)
    fft = np.fft.fft(intensity - np.mean(intensity))
    fft_mag = np.abs(fft)[:len(fft)//2]
    features = [
        np.mean(intensity), np.std(intensity),
        np.max(intensity) - np.min(intensity),
        np.mean(np.abs(diffs)), np.max(np.abs(diffs)), np.std(diffs),
        fft_mag[1] if len(fft_mag) > 1 else 0,
        np.sum(fft_mag[:5]) if len(fft_mag) >= 5 else np.sum(fft_mag),
        np.argmax(fft_mag[1:]) + 1 if len(fft_mag) > 1 else 0
    ]
    for ws in [10, 25]:
        if len(intensity) >= ws:
            windows = [intensity[i:i+ws] for i in range(0, len(intensity)-ws+1, max(ws//2, 1))]
            if windows:
                features.extend([
                    np.mean([np.std(w) for w in windows]),
                    np.mean([np.max(w) - np.min(w) for w in windows])
                ])
            else:
                features.extend([0, 0])
        else:
            features.extend([0, 0])
    return np.array(features)


def extract_spatial(video):
    avg = np.mean(video, axis=0)
    h, w = avg.shape[:2]
    features = []
    for q in [avg[:h//2,:w//2], avg[:h//2,w//2:], avg[h//2:,:w//2], avg[h//2:,w//2:]]:
        features.extend([np.mean(q), np.std(q)])
    left = avg[:, :w//2]
    right = np.flip(avg[:, w//2:], axis=1)
    min_w = min(left.shape[1], right.shape[1])
    if min_w > 0:
        lf, rf = left[:, :min_w].flatten(), right[:, :min_w].flatten()
        if len(lf) > 1 and np.std(lf) > 0 and np.std(rf) > 0:
            sym = np.corrcoef(lf, rf)[0, 1]
        else:
            sym = 0
    else:
        sym = 0
    features.append(sym if not np.isnan(sym) else 0)
    var = np.mean(np.var(video, axis=0), axis=2) if len(video.shape) > 3 else np.var(video, axis=0)
    total = np.sum(var)
    if total > 0:
        y_c, x_c = np.meshgrid(range(h), range(w), indexing='ij')
        features.extend([np.sum(y_c * var) / total / h, np.sum(x_c * var) / total / w])
    else:
        features.extend([0.5, 0.5])
    return np.array(features)


def extract_features_single(video):
    video = ensure_uint8_video(video)
    return np.concatenate([extract_optical_flow(video), extract_temporal(video), extract_spatial(video)])


def apply_common_acquisition_artifacts(video, rng):
    video_f = ensure_uint8_video(video).astype(np.float32)
    gain = rng.uniform(0.97, 1.03)
    bias = rng.uniform(-4.0, 4.0)
    video_f = video_f * gain + bias
    if rng.rand() < 0.85:
        video_f += rng.normal(0, rng.uniform(0.8, 2.2), video_f.shape)
    if rng.rand() < 0.25:
        gamma = rng.uniform(0.97, 1.03)
        video_f = 255.0 * np.power(np.clip(video_f / 255.0, 0, 1), gamma)
    return np.clip(video_f, 0, 255).astype(np.uint8)


def apply_tremor(video, severity, rng):
    h, w = video.shape[1], video.shape[2]
    amp = rng.uniform(0.4, 1.6) * max(severity, 0.1)
    freq = rng.uniform(3.0, 6.0)
    phase = rng.uniform(0, 2*np.pi)
    out = np.empty_like(video)
    for i in range(len(video)):
        dx = amp * np.sin(2*np.pi*freq*i/len(video) + phase)
        dy = 0.5 * amp * np.cos(2*np.pi*freq*i/len(video) + phase)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        out[i] = cv2.warpAffine(video[i], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out


def apply_asymmetry(video, severity, rng):
    out = video.astype(np.float32).copy()
    mean_frame = np.mean(out, axis=0)
    h, w = out.shape[1], out.shape[2]
    atten = 1.0 - rng.uniform(0.10, 0.35) * severity
    left_side = rng.rand() < 0.5
    cols = slice(0, w//2) if left_side else slice(w//2, w)
    out[:, :, cols, :] = atten * out[:, :, cols, :] + (1 - atten) * mean_frame[:, cols, :]
    gradient = np.linspace(1.0, atten, w//2, dtype=np.float32).reshape(1, 1, -1, 1)
    if left_side:
        out[:, :, :w//2, :] *= gradient
    else:
        out[:, :, w//2:, :] *= gradient[:, :, ::-1, :]
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_motion_blur(video, severity, rng):
    k = int(3 + 4 * severity)
    if k % 2 == 0:
        k += 1
    kernel = np.zeros((k, k), dtype=np.float32)
    if rng.rand() < 0.5:
        kernel[k//2, :] = 1.0
    else:
        kernel[:, k//2] = 1.0
    kernel /= kernel.sum()
    out = video.copy()
    step = max(2, int(6 - 3 * severity))
    for i in range(0, len(video), step):
        out[i] = cv2.filter2D(out[i], -1, kernel)
    return out


def apply_temporal_jitter(video, severity, rng):
    out = video.copy()
    n_changes = max(1, int(len(video) * rng.uniform(0.02, 0.06) * max(severity, 0.2)))
    valid_idx = np.arange(1, len(video) - 1)
    chosen = rng.choice(valid_idx, size=min(n_changes, len(valid_idx)), replace=False)
    for idx in chosen:
        out[idx] = out[idx - 1] if rng.rand() < 0.5 else out[idx + 1]
    return out


def apply_global_noise(video, severity, rng):
    sigma = rng.uniform(1.5, 6.0) * max(severity, 0.15)
    noisy = video.astype(np.float32) + rng.normal(0, sigma, video.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def generate_synthetic_pathological_video(base_video, rng, exclude_set=None):
    """
    Genera vídeo patológico sintético con perturbaciones clínicas.
    
    Args:
        base_video: vídeo base normal
        rng: generador aleatorio
        exclude_set: set de strings con perturbaciones a excluir
                    {'artifacts', 'rigidez', 'temblor', 'asimetria', 
                     'motion_blur', 'jitter', 'ruido'}
    """
    if exclude_set is None:
        exclude_set = set()
    
    video = apply_common_acquisition_artifacts(base_video, rng).astype(np.float32) if 'artifacts' not in exclude_set else ensure_uint8_video(base_video).astype(np.float32)
    mean_frame = np.mean(video, axis=0, keepdims=True)
    severity = rng.uniform(0.20, 0.65)
    if rng.rand() < 0.35:
        severity *= rng.uniform(0.35, 0.65)
    
    if 'rigidez' not in exclude_set:
        amplitude_scale = 1.0 - rng.uniform(0.12, 0.42) * severity
        video = mean_frame + amplitude_scale * (video - mean_frame)
    
    video = np.clip(video, 0, 255).astype(np.uint8)
    
    if 'temblor' not in exclude_set and rng.rand() < 0.80:
        video = apply_tremor(video, severity, rng)
    if 'asimetria' not in exclude_set and rng.rand() < 0.65:
        video = apply_asymmetry(video, severity, rng)
    if 'motion_blur' not in exclude_set and rng.rand() < 0.45:
        video = apply_motion_blur(video, severity, rng)
    if 'jitter' not in exclude_set and rng.rand() < 0.35:
        video = apply_temporal_jitter(video, severity, rng)
    if 'ruido' not in exclude_set:
        video = apply_global_noise(video, severity, rng)
    
    if rng.rand() < 0.50:
        original = apply_common_acquisition_artifacts(base_video, rng).astype(np.float32) if 'artifacts' not in exclude_set else ensure_uint8_video(base_video).astype(np.float32)
        alpha = rng.uniform(0.08, 0.22)
        video = np.clip((1 - alpha) * video.astype(np.float32) + alpha * original, 0, 255).astype(np.uint8)
    return video


def extract_feature_batch_from_videos(videos, desc, apply_common_noise=True):
    rng = np.random.RandomState(RANDOM_STATE)
    X = np.zeros((len(videos), 30), dtype=np.float32)
    for i, video in enumerate(videos):
        try:
            current = apply_common_acquisition_artifacts(video, rng) if apply_common_noise else ensure_uint8_video(video)
            X[i] = extract_features_single(current)
        except Exception:
            X[i] = np.zeros(30, dtype=np.float32)
        if (i + 1) % 100 == 0 or i == len(videos) - 1:
            print(f"  {desc}: {i+1}/{len(videos)}")
            gc.collect()
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def generate_synthetic_pathological_from_videos(videos, n_synthetic=None, exclude_set=None):
    if n_synthetic is None:
        n_synthetic = len(videos)
    rng = np.random.RandomState(RANDOM_STATE)
    X_patho = np.zeros((n_synthetic, 30), dtype=np.float32)
    desc_suffix = f" [excluye: {', '.join(sorted(exclude_set))}]" if exclude_set else ""
    for i in range(n_synthetic):
        try:
            base_video = videos[rng.randint(0, len(videos))]
            synth_video = generate_synthetic_pathological_video(base_video, rng, exclude_set=exclude_set)
            X_patho[i] = extract_features_single(synth_video)
        except Exception:
            X_patho[i] = np.zeros(30, dtype=np.float32)
        if (i + 1) % 100 == 0 or i == n_synthetic - 1:
            print(f"  Patológicas sintéticas (vídeo crudo){desc_suffix}: {i+1}/{n_synthetic}")
            gc.collect()
    return np.nan_to_num(X_patho, nan=0.0, posinf=0.0, neginf=0.0)


def load_and_generate_dataset(exclude_set=None):
    pr("FASE 1: CARGA DE DATOS Y GENERACIÓN SINTÉTICA [raw_video]")
    videos, raw_labels = load_raw_video_corpus()
    ps("Extracción de features de los vídeos normales con ruido de adquisición compartido")
    X_norm = extract_feature_batch_from_videos(videos, 'Normales')
    ps("Generación de patología sintética a partir de vídeo crudo")
    X_patho = generate_synthetic_pathological_from_videos(videos, n_synthetic=len(videos), exclude_set=exclude_set)
    dataset_note = 'Perturbación sobre vídeo crudo antes de extracción de features'
    if exclude_set:
        dataset_note += f' [excluye: {", ".join(sorted(exclude_set))}]'
    save_name = DATA_DIR / f'features_synthetic_raw_video_{TIMESTAMP}.npz'
    y_orig = raw_labels

    X = np.vstack([X_norm, X_patho])
    y = np.concatenate([np.zeros(len(X_norm), dtype=int), np.ones(len(X_patho), dtype=int)])
    idx = np.random.RandomState(RANDOM_STATE).permutation(len(X))
    X, y = X[idx], y[idx]

    print(f"  Dataset combinado: {X.shape[0]} muestras")
    print(f"  Normal: {np.sum(y==0)}, Patológico: {np.sum(y==1)}")

    sig = sum(1 for i in range(30) if stats.ttest_ind(X[y==0, i], X[y==1, i]).pvalue < 0.001)
    print(f"  Features con p<0.001: {sig}/30")

    np.savez(save_name, X=X, y=y, X_normal=X_norm, X_pathological=X_patho, source_mode='raw_video')
    print(f"  Dataset guardado: {save_name.name}")
    return X, y, X_norm, X_patho, dataset_note


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

def export_results(results_df, det, cv_df, pca, splits, train_times, total_time, mode_tag, dataset_note):
    pr("FASE 8: EXPORTACIÓN")
    best = results_df.iloc[0]
    summary = {
        'timestamp': TIMESTAMP, 'experiment': mode_tag, 'date': datetime.now().strftime('%d/%m/%Y'),
        'dataset': {
            'total': len(splits['y_train'])+len(splits['y_val'])+len(splits['y_test']),
            'n_features_raw': 30, 'n_features_pca': int(pca.n_components_),
            'pca_variance': float(sum(pca.explained_variance_ratio_)),
            'synthetic_method': dataset_note
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


def run_experiment(exclude_set=None):
    experiment_start = time.time()
    configure_output_dirs()
    X, y, Xn, Xp, dataset_note = load_and_generate_dataset(exclude_set=exclude_set)
    splits, scaler, pca = preprocess(X, y)
    models, times = train_models(splits)
    results_df, det = evaluate_models(models, splits, times)
    X_sc = scaler.transform(X)
    cv_res, cv_df = cross_validation_analysis(X_sc, y, pca)
    generate_visualizations(models, splits, results_df, det, pca, scaler, cv_res, X, y, Xn, Xp)
    shap_analysis(models, splits, pca)
    total = time.time() - experiment_start
    summary = export_results(results_df, det, cv_df, pca, splits, times, total, 'raw_video', dataset_note)
    return {
        'results_df': results_df,
        'det': det,
        'cv_df': cv_df,
        'summary': summary,
        'dataset_note': dataset_note
    }


# ============================================================================
# EXPERIMENTO B: ABLACIÓN SISTEMÁTICA
# ============================================================================

def run_ablation_experiment():
    """
    Ejecuta el estudio de ablación sistemática (Experimento B).
    Genera 8 datasets con diferentes perturbaciones excluidas,
    entrena modelos y compara resultados.
    """
    t0 = time.time()
    pr("EXPERIMENTO B: ESTUDIO DE ABLACIÓN SISTEMÁTICA")
    
    # Definir las 8 condiciones de ablación
    ablation_conditions = [
        ('A_completo', None, 'Completo (todas las perturbaciones)'),
        ('B_sin_rigidez', {'rigidez'}, 'Sin rigidez (amplitude_scale)'),
        ('C_sin_temblor', {'temblor'}, 'Sin temblor'),
        ('D_sin_asimetria', {'asimetria'}, 'Sin asimetría'),
        ('E_sin_motion_blur', {'motion_blur'}, 'Sin motion blur'),
        ('F_sin_jitter', {'jitter'}, 'Sin fluctuación temporal'),
        ('G_sin_ruido', {'ruido'}, 'Sin ruido gaussiano'),
        ('H_sin_artifacts', {'artifacts'}, 'Sin artifacts de adquisición')
    ]
    
    # Crear directorio para resultados de ablación
    ablation_dir = REPORTS_DIR / f'ablation_{TIMESTAMP}'
    ablation_dir.mkdir(parents=True, exist_ok=True)
    
    results_summary = []
    all_results = {}
    
    for condition_id, exclude_set, description in ablation_conditions:
        ps(f"Condición {condition_id}: {description}")
        condition_start = time.time()
        
        # Configurar directorios específicos para esta condición
        global MODELS_DIR, FIGURES_DIR, RESULTS_DIR
        MODELS_DIR = ablation_dir / condition_id / 'models'
        FIGURES_DIR = ablation_dir / condition_id / 'figures'
        RESULTS_DIR = ablation_dir / condition_id / 'results'
        for d in [MODELS_DIR, FIGURES_DIR, RESULTS_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Ejecutar experimento para esta condición (solo modo raw_video)
        try:
            X, y, Xn, Xp, dataset_note = load_and_generate_dataset(exclude_set=exclude_set)
            splits, scaler, pca = preprocess(X, y)
            models, times = train_models(splits)
            results_df, det = evaluate_models(models, splits, times)
            
            # Guardar resultados de esta condición
            for _, row in results_df.iterrows():
                results_summary.append({
                    'Condicion': condition_id,
                    'Descripcion': description,
                    'Modelo': row['Model'],
                    'Accuracy': row['Accuracy'],
                    'AUC-ROC': row['AUC-ROC'],
                    'F1-Score': row['F1-score'],
                    'Tiempo_s': times.get(row['Model'], 0)
                })
            
            all_results[condition_id] = {
                'results_df': results_df,
                'description': description,
                'exclude_set': exclude_set,
                'elapsed_s': time.time() - condition_start
            }
            
            print(f"  ✅ Condición {condition_id} completada en {time.time() - condition_start:.1f}s")
        except Exception as e:
            print(f"  ❌ Error en condición {condition_id}: {e}")
            continue
    
    # Crear tabla resumen
    pr("GENERANDO ANÁLISIS COMPARATIVO")
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(ablation_dir / 'summary_table.csv', index=False)
    
    # Calcular impacto de cada perturbación (usando SVM como referencia)
    baseline_acc = summary_df[(summary_df['Condicion'] == 'A_completo') & (summary_df['Modelo'] == 'SVM')]['Accuracy'].values[0]
    
    impact_analysis = []
    for condition_id, exclude_set, description in ablation_conditions[1:]:  # Skip completo
        cond_acc = summary_df[(summary_df['Condicion'] == condition_id) & (summary_df['Modelo'] == 'SVM')]['Accuracy'].values
        if len(cond_acc) > 0:
            delta = (cond_acc[0] - baseline_acc) * 100
            impact_rel = (delta / baseline_acc) * 100 if baseline_acc > 0 else 0
            perturbation = list(exclude_set)[0] if exclude_set else 'ninguna'
            impact_analysis.append({
                'Perturbacion_excluida': perturbation,
                'Condicion': condition_id,
                'Accuracy_SVM': cond_acc[0],
                'Delta_pp': delta,
                'Impacto_relativo_%': impact_rel
            })
    
    impact_df = pd.DataFrame(impact_analysis)
    impact_df = impact_df.sort_values('Delta_pp', ascending=True)
    impact_df.to_csv(ablation_dir / 'impact_analysis.csv', index=False)
    
    # Generar visualización comparativa
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gráfico 1: Accuracy por condición (todos los modelos)
    pivot = summary_df.pivot(index='Condicion', columns='Modelo', values='Accuracy')
    pivot.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Accuracy por condición y modelo', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Condición')
    ax1.legend(title='Modelo', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0.5, 1.0)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Gráfico 2: Impacto relativo de cada perturbación
    colors = ['#D32F2F' if x < -10 else '#FF9800' if x < -5 else '#4CAF50' for x in impact_df['Delta_pp']]
    ax2.barh(impact_df['Perturbacion_excluida'], impact_df['Delta_pp'], color=colors)
    ax2.set_xlabel('Δ Accuracy (pp)', fontsize=12)
    ax2.set_title('Impacto de excluir cada perturbación (SVM)', fontsize=14, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(ablation_dir / 'ablation_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Generar tabla LaTeX
    latex_lines = [
        '\\begin{table}[H]',
        '\\centering',
        '\\caption{Resultados del estudio de ablación sistemática (SVM).}',
        '\\label{tab:ablation_results}',
        '\\begin{tabular}{lcccc}',
        '\\toprule',
        '\\textbf{Condición} & \\textbf{Perturbación excluida} & \\textbf{Accuracy} & \\textbf{$\\Delta$ (pp)} & \\textbf{Impacto rel.} \\\\',
        '\\midrule',
        f'Completo & Ninguna & {baseline_acc:.4f} & - & 100\\% \\\\'
    ]
    
    for _, row in impact_df.iterrows():
        latex_lines.append(
            f"{row['Condicion'].replace('_', ' ')} & {row['Perturbacion_excluida'].capitalize()} & "
            f"{row['Accuracy_SVM']:.4f} & {row['Delta_pp']:.2f} & "
            f"{100 + row['Impacto_relativo_%']:.1f}\\% \\\\"
        )
    
    latex_lines.extend([
        '\\bottomrule',
        '\\end{tabular}',
        '\\end{table}'
    ])
    
    (ablation_dir / 'tabla_latex.tex').write_text('\n'.join(latex_lines), encoding='utf-8')
    
    # Guardar análisis completo en JSON
    analysis_json = {
        'timestamp': TIMESTAMP,
        'baseline_accuracy': float(baseline_acc),
        'total_time_s': time.time() - t0,
        'conditions': {cid: {'description': desc, 'exclude': list(exc) if exc else []} 
                      for cid, exc, desc in ablation_conditions},
        'impact_ranking': impact_df.to_dict('records')
    }
    
    with open(ablation_dir / 'ablation_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(analysis_json, f, indent=2, ensure_ascii=False)
    
    # Resumen final
    pr("RESUMEN DEL EXPERIMENTO DE ABLACIÓN")
    print(f"\n  Baseline (completo): Accuracy = {baseline_acc:.4f}")
    print(f"\n  Ranking de impacto (de mayor a menor):")
    for i, row in impact_df.iterrows():
        print(f"    {i+1}. {row['Perturbacion_excluida']:15s}: Δ = {row['Delta_pp']:6.2f} pp")
    
    print(f"\n  Tiempo total: {time.time() - t0:.1f}s ({(time.time() - t0)/60:.1f} min)")
    print(f"  Resultados guardados en: {ablation_dir}/")
    print(f"\n  ✅ EXPERIMENTO B COMPLETADO")
    
    return ablation_dir, summary_df, impact_df


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Pipeline ML Detección Espasticidad Infantil - Modo B')
    parser.add_argument('--experiment', type=str, choices=['video', 'ablation'], default='video',
                       help='Tipo de experimento: video (Modo B) o ablation (Experimento B)')
    args = parser.parse_args()
    
    t0 = time.time()
    print("\n" + "█"*78)
    print("█  PIPELINE MODO B - ML DETECCIÓN ESPASTICIDAD INFANTIL                █")
    print("█"*78)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.experiment == 'ablation':
        ablation_dir, summary_df, impact_df = run_ablation_experiment()
        total = time.time() - t0
        print(f"\n  ✅ EXPERIMENTO B COMPLETADO - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  TIEMPO TOTAL: {total:.1f}s ({total/60:.1f}min)")
    else:
        pr('MODO B: PERTURBACIÓN SOBRE VÍDEO CRUDO')
        experiment = run_experiment()
        total = time.time() - t0
        best = experiment['results_df'].iloc[0]
        print(f"""
  MODO B (raw_video):
    Mejor modelo: {best['Model']}
    Accuracy:      {best['Accuracy']:.4f}
    Sensibilidad:  {best['Sensitivity']:.4f}
    Especificidad: {best['Specificity']:.4f}
    AUC-ROC:       {best['AUC-ROC']:.4f}

  TIEMPO TOTAL: {total:.1f}s ({total/60:.1f}min)
  RESULTADOS GUARDADOS EN: {RESULTS_DIR}
  ✅ COMPLETADO - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")


if __name__ == '__main__':
    main()
