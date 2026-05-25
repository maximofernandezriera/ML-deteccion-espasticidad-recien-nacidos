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

| Modelo | Accuracy | Sensibilidad | Especificidad | AUC-ROC |
|--------|----------|--------------|---------------|---------|
| **SVM** | **99.35%** | 98.69% | **100%** | **0.9997** |
| Logistic Regression | 99.02% | 99.35% | 98.70% | 0.9994 |
| Random Forest | 99.02% | 99.35% | 98.70% | 0.9981 |
| XGBoost | 99.02% | 98.69% | 99.35% | 0.9985 |

## Estructura del repositorio

```
ML-deteccion-espasticidad-recien-nacidos/
├── main_pipeline.py                # Pipeline principal (entrenamiento + evaluación)
├── test_ablation.py                # Test de ablación sistemática (7 perturbaciones)
├── requirements.txt                # Dependencias Python
├── README.md                       # Este archivo
├── README_EXPERIMENTO_B.md         # Documentación del experimento sobre vídeo crudo
├── docs/
│   └── pca_loadings.md             # Tabla completa de ponderaciones PCA → features originales
├── data/                           # (ignorado por git, ver instrucciones de descarga)
│   └── raw/
│       ├── data_100_50_50.npz      # Vídeos (1534 × 100 × 50 × 50)
│       └── target_100_50_50.npz    # Etiquetas binarias
├── reports/                        # (ignorado por git, generado al ejecutar)
│   ├── figures_video/              # 16 figuras PNG del experimento principal
│   ├── results_video/              # CSVs y JSON con métricas
│   └── ablation_<timestamp>/       # Resultados de ablación
└── models/                         # (ignorado por git)
    └── video_<timestamp>/          # Modelos serializados (.pkl)
```

## Requisitos y configuración

- **Python**: 3.10 o superior
- **CPU**: ≥4 núcleos recomendado
- **RAM**: ≥8 GB
- **GPU**: no requerida
- **Tiempo total esperado**: ~2 min (experimento principal) + ~17 min (ablación)

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
pip install opencv-python

# 3. Descargar el dataset original (ver sección Dataset más abajo) y colocarlo en data/raw/

# 4. Ejecutar el experimento principal
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
| `models/video_<timestamp>/` | Modelos serializados (`.pkl`) por modelo |

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
| SVM | kernel RBF, `C=1.0`, `gamma='scale'` |
| XGBoost | `n_estimators=200`, `learning_rate=0.1`, `max_depth=6` |

## Mapa de figuras generadas

El pipeline produce 16 figuras PNG numeradas en `reports/figures_video/`:

| Nº | Archivo | Contenido |
|----|---------|-----------|
| 01 | `fig01_dataset_distribution.png` | Distribución de clases (normal/alterado) |
| 02 | `fig02_features_correlation.png` | Matriz de correlación 30×30 |
| 03 | `fig03_features_distribution.png` | Histogramas por familia de features |
| 04 | `fig04_class_separability.png` | Separabilidad univariante por feature |
| 05 | `fig05_pca_variance.png` | Varianza explicada acumulada PCA |
| 06 | `fig06_pca_analysis.png` | Análisis PCA y proyección 2D |
| 07 | `fig07_models_comparison.png` | Comparativa global de modelos |
| 08 | `fig08_confusion_matrices.png` | Matrices de confusión |
| 09 | `fig09_roc_curves.png` | Curvas ROC con AUC |
| 10 | `fig10_pr_curves.png` | Curvas Precision-Recall |
| 11 | `fig11_heatmap.png` | Heatmap comparativo de todas las métricas |
| 12 | `fig12_cv_results.png` | Resultados validación cruzada 5-fold |
| 13 | `fig13_clinical.png` | Métricas clínicas resumidas |
| 14 | `fig14_learning_curves.png` | Curvas de aprendizaje |
| 15 | `fig15_shap_importance.png` | Importancia SHAP por componente |
| 16 | `fig16_shap_beeswarm.png` | Diagrama SHAP beeswarm para RF |

## Documentación complementaria

- [`docs/pca_loadings.md`](docs/pca_loadings.md) — **Tabla completa** de ponderaciones de las 30 características originales sobre los 12 componentes principales, agrupada por importancia SHAP.
- [`README_EXPERIMENTO_B.md`](README_EXPERIMENTO_B.md) — Descripción del experimento de aumentación sobre vídeo crudo (perturbaciones aplicadas y justificación clínica).

## Dataset

El corpus original contiene **767 vídeos de movimiento espontáneo normal** procedentes del [Infant Movements Dataset (Kaggle)](https://www.kaggle.com/datasets/) (resolución 50×50, 100 frames por muestra).

Se generan **767 muestras sintéticas patológicas** mediante 7 perturbaciones clínicamente motivadas aplicadas sobre el vídeo crudo:

1. Rigidez (reducción de amplitud y variabilidad)
2. Temblor de alta frecuencia
3. Asimetría bilateral
4. Reducción de complejidad temporal
5. Pausas prolongadas
6. Movimientos estereotipados
7. Combinaciones aleatorias

Total: **1534 muestras balanceadas** (767 normal + 767 sintético).

> **Nota**: los archivos `data_100_50_50.npz` y `target_100_50_50.npz` no se incluyen en el repo por tamaño. Contactar con el autor o descargar del dataset público referenciado en la memoria del TFM.

## Autor y licencia

**Máximo Fernández Riera**
TFM — Máster Universitario en Ingeniería Informática
Universitat Oberta de Catalunya (UOC)

**Licencia**: MIT — ver `LICENSE` (libre uso académico y comercial con atribución).
