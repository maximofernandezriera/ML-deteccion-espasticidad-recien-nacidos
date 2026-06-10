# ML Detección de Espasticidad en Recién Nacidos

Pipeline de Machine Learning tradicional para el apoyo al diagnóstico temprano de espasticidad en neonatos a partir de análisis de movimiento en vídeo.

> **Repositorio anexo a la memoria del TFM.** Contiene el código del pipeline, las instrucciones detalladas de reproducción y la documentación técnica complementaria que no se incluye íntegramente en el documento principal.

## Tabla de contenidos

- [Descripción](#descripción)
- [Resultados principales](#resultados-principales)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Requisitos y configuración](#requisitos-y-configuración)
- [Instrucciones de reproducción](#instrucciones-de-reproducción)
- [Configuración centralizada del experimento](#configuración-centralizada-del-experimento)
- [Mapa de figuras generadas](#mapa-de-figuras-generadas)
- [Documentación complementaria](#documentación-complementaria)
- [Dataset](#dataset)
- [Autor y licencia](#autor-y-licencia)

## Descripción

Este proyecto aborda la detección temprana de espasticidad mediante el análisis de patrones de movimiento infantil. Utiliza algoritmos de ML tradicional (Regresión Logística, Random Forest, SVM, XGBoost) sobre características cinemáticas extraídas de vídeos.

**Características principales:**
- Clasificación binaria: Normal vs. Patológico (sintético)
- 30 características cinemáticas (flujo óptico, temporales, espaciales)
- Reducción dimensional mediante PCA (30 → 12 componentes, 96.34% varianza)
- Aumentación sobre vídeo crudo con 7 perturbaciones clínicamente motivadas
- Explicabilidad mediante análisis SHAP
- Pipeline reproducible con semilla fija (`random_state=42`)
- Experimento de ablación sistemática (contribución individual de cada perturbación)

## Resultados principales

Resultados sobre el conjunto de test independiente (n=307), con las perturbaciones aplicadas **sobre vídeo crudo** (las que se defienden en la memoria):

| Modelo | Accuracy | Sensibilidad | Especificidad | AUC-ROC |
|--------|----------|--------------|---------------|---------|
| **SVM** | **82.74%** | 77.12% | **88.31%** | **0.9117** |
| XGBoost | 82.08% | **80.39%** | 83.77% | 0.9011 |
| Random Forest | 80.46% | 78.43% | 82.47% | 0.8900 |
| Logistic Regression | 78.83% | 75.82% | 81.82% | 0.8489 |

El SVM lidera en accuracy, especificidad, AUC-ROC, MCC (0.659) y Brier Score (0.1184); XGBoost destaca en sensibilidad y F1. Estos valores, más conservadores que los de perturbar directamente las características ya extraídas, reflejan un escenario más realista porque las perturbaciones sobre vídeo crudo preservan las correlaciones naturales entre familias de características.

> **Nota sobre la calibración:** el SVM utiliza Platt scaling y alcanza un Brier Score de 0.1184 (Brier Skill Score 0.526, ECE 0.0568), lo que indica probabilidades razonablemente calibradas.

## Estructura del repositorio

```
ML-deteccion-espasticidad-recien-nacidos/
├── main_pipeline.py                # Pipeline principal sobre vídeo crudo (entrenamiento + evaluación + ablación)
├── test_ablation.py                # Test de verificación del experimento de ablación
├── requirements.txt                # Dependencias Python
├── README.md                       # Este archivo
├── README_EXPERIMENTO_B.md         # Documentación del experimento de ablación sistemática
├── docs/
│   └── pca_loadings.md             # Tabla completa de ponderaciones PCA → features originales
├── data/                           # (parcialmente ignorado por git, ver instrucciones)
│   └── raw/
│       ├── data_100_50_50.npz      # Vídeos normales (767 × 100 × 50 × 50)
│       └── target_100_50_50.npz    # Etiquetas originales
├── reports/                        # (ignorado por git, generado al ejecutar)
│   ├── figures_video/              # 16 figuras PNG del experimento principal
│   ├── results_video/              # CSVs y JSON con métricas
│   └── ablation_<timestamp>/       # Resultados de ablación
└── models/                         # (ignorado por git)
    └── synthetic_video/            # Modelos serializados (.pkl)
```

## Requisitos y configuración

- **Python**: 3.10 o superior
- **CPU**: ≥4 núcleos recomendado
- **RAM**: ≥8 GB
- **GPU**: no requerida
- **Tiempo total esperado**: ~2-3 min (experimento principal) + ~17 min (ablación)

Dependencias principales: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `opencv-python`. Versiones exactas en `requirements.txt`.

## Instrucciones de reproducción

```bash
# 1. Clonar el repositorio
git clone https://github.com/maximofernandezriera/ML-deteccion-espasticidad-recien-nacidos.git
cd ML-deteccion-espasticidad-recien-nacidos

# 2. Crear entorno virtual e instalar dependencias
python3 -m venv venv
source venv/bin/activate     # En Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Descargar el dataset original (ver sección Dataset) y colocarlo en data/raw/
#    data/raw/data_100_50_50.npz  y  data/raw/target_100_50_50.npz

# 4. Ejecutar el experimento principal (perturbación sobre vídeo crudo)
python3 main_pipeline.py --experiment video

# 5. Ejecutar el experimento de ablación sistemática
python3 main_pipeline.py --experiment ablation
```

Los resultados se generan automáticamente en:

| Carpeta | Contenido |
|---------|-----------|
| `reports/figures_video/` | 16 figuras PNG del experimento principal |
| `reports/results_video/` | CSVs y JSON con todas las métricas |
| `reports/ablation_<timestamp>/` | Resultados detallados de la ablación |
| `models/synthetic_video/` | Modelos serializados (`.pkl`) por modelo |

## Configuración centralizada del experimento

Todos los hiperparámetros del experimento se centralizan al inicio de `main_pipeline.py`. Los valores adoptados en el TFM son:

| Parámetro | Valor |
|-----------|-------|
| `random_state` | `42` (semilla fija para todos los procesos estocásticos) |
| Partición de datos | 60 % train · 20 % val · 20 % test (estratificada) |
| Umbral PCA | varianza explicada ≥ 95 % → 12 componentes (96.34 %) |
| Validación cruzada | 5-fold estratificada |

**Configuración de los 4 modelos** (todos con `class_weight='balanced'`):

| Modelo | Hiperparámetros |
|--------|-----------------|
| Logistic Regression | solver `saga`, `max_iter=2000`, regularización L2 |
| Random Forest | `n_estimators=200`, `max_depth=15` |
| SVM | kernel RBF, `C=1.0`, `gamma='scale'`, `probability=True` (Platt scaling) |
| XGBoost | `n_estimators=200`, `learning_rate=0.1`, `max_depth=6` |

## Mapa de figuras generadas

El pipeline produce 16 figuras PNG numeradas en `reports/figures_video/`:

| Nº | Archivo | Contenido |
|----|---------|-----------|
| 01 | `fig01_feature_distributions.png` | Distribuciones de características: normal vs alterado |
| 02 | `fig02_model_comparison.png` | Comparativa de métricas por modelo |
| 03 | `fig03_confusion_matrices.png` | Matrices de confusión de los 4 modelos |
| 04 | `fig04_roc_curves.png` | Curvas ROC con AUC |
| 05 | `fig05_precision_recall.png` | Curvas Precision-Recall |
| 06 | `fig06_pca_analysis.png` | Varianza explicada PCA y proyección 2D |
| 07 | `fig07_learning_curves.png` | Curvas de aprendizaje |
| 08 | `fig08_cv_boxplots.png` | Boxplots de validación cruzada 5-fold |
| 09 | `fig09_confidence.png` | Distribución de confianza de las predicciones |
| 10 | `fig10_calibration.png` | Curvas de calibración (con Brier) |
| 11 | `fig11_heatmap.png` | Heatmap comparativo de todas las métricas |
| 12 | `fig12_threshold.png` | Análisis de umbral (cribado vs confirmación) |
| 13 | `fig13_clinical.png` | Métricas clínicas (Sensibilidad, Especificidad, VPP, VPN) |
| 14 | `fig14_times.png` | Tiempos de entrenamiento por modelo |
| 15 | `fig15_shap_importance.png` | Importancia SHAP por componente (RF y XGBoost) |
| 16 | `fig16_shap_beeswarm.png` | Diagrama SHAP beeswarm |

## Documentación complementaria

- [`docs/pca_loadings.md`](docs/pca_loadings.md) — **Tabla completa** de ponderaciones de las 30 características originales sobre los 12 componentes principales, agrupada por importancia SHAP.
- [`README_EXPERIMENTO_B.md`](README_EXPERIMENTO_B.md) — Descripción del experimento de ablación sistemática (perturbaciones aplicadas, condiciones y justificación clínica).

## Dataset

El corpus original contiene **767 vídeos de movimiento espontáneo normal** procedentes del [Infant Movements Dataset (Kaggle)](https://www.kaggle.com/datasets/hansamaldharmananda/infants-movements-kicking-patterns-data-set) (resolución 50×50, 100 frames por muestra).

Se generan **767 muestras sintéticas patológicas** mediante 7 perturbaciones clínicamente motivadas aplicadas **sobre el vídeo crudo** (antes de extraer las características, para preservar las correlaciones naturales):

1. Rigidez muscular (reducción de amplitud y variabilidad)
2. Temblor patológico (oscilaciones de alta frecuencia, 3–6 Hz)
3. Asimetría motora (lesión neurológica unilateral)
4. Motion blur (desenfoque direccional)
5. Fluctuación temporal / jitter (pérdida de fluidez)
6. Ruido muscular (gaussiano)
7. Defectos de captura / artefactos de adquisición

Total: **1534 muestras balanceadas** (767 normal + 767 sintético).

> **Nota**: los archivos `data_100_50_50.npz` y `target_100_50_50.npz` no se incluyen en el repo por tamaño. Descárgalos del dataset público referenciado y colócalos en `data/raw/`.

## Autor y licencia

**Máximo Fernández Riera**
TFM — Máster Universitario en Ingeniería Informática
Universitat Oberta de Catalunya (UOC)

**Licencia**: MIT — ver `LICENSE` (libre uso académico y comercial con atribución).
