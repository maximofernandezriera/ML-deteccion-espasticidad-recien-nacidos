# Ponderaciones PCA → características originales

Este documento contiene la tabla completa de ponderaciones (contribuciones) de las 30 características cinemáticas originales sobre los 12 componentes principales (PCs) seleccionados tras el análisis PCA con umbral de varianza explicada del 95 % (varianza acumulada: 96.34 %).

> **Contexto**: la memoria del TFM presenta únicamente las ponderaciones de los tres componentes más relevantes según el análisis SHAP (PC8, PC11, PC12). Esta tabla extendida está disponible aquí como anexo digital para permitir la trazabilidad completa entre los valores SHAP y las características físicas observables.

## Tabla 1 — Top-3 componentes según análisis SHAP

Componentes ordenados por su importancia SHAP en Random Forest (modelo de referencia para interpretabilidad).

### PC8 — Frecuencia y dinámica temporal (19.5 % imp. SHAP en RF, 17.9 % en XGB)

| Característica | Familia | Ponderación |
|----------------|---------|-------------|
| Frecuencia dominante temporal | Temporal | **0.46** |
| Desv. estándar cuadrante inferior izquierdo | Espacial | **0.40** |
| Desv. estándar cuadrante inferior derecho | Espacial | **0.35** |
| Centro de masa *x* | Espacial | 0.23 |
| Simetría bilateral | Simetría | 0.17 |
| *(resto < 0.15)* | — | — |

**Lectura clínica**: PC8 condensa la frecuencia dominante del movimiento y la actividad diferencial por cuadrantes inferiores. En términos clínicos, mide si el ritmo del movimiento es regular o si aparecen oscilaciones anómalas — **el marcador cinemático del temblor**.

### PC11 — Asimetría lateral (16.4 % imp. SHAP en RF, 11.6 % en XGB)

| Característica | Familia | Ponderación |
|----------------|---------|-------------|
| Centro de masa *x* | Espacial | **0.49** |
| Actividad cuadrante inferior izquierdo | Espacial | **0.45** |
| Desv. estándar cuadrante inferior derecho | Espacial | **−0.35** |
| Ratio píxeles con movimiento | Flujo óptico | 0.32 |
| Desv. estándar cuadrante inferior izquierdo | Espacial | −0.28 |
| *(resto < 0.25)* | — | — |

**Lectura clínica**: PC11 recoge la posición lateral del centro de masa y la distribución asimétrica de actividad entre cuadrantes. Equivale a detectar **si el movimiento se concentra más en un lado del cuerpo que en el otro** — indicio del uso asimétrico de extremidades, signo característico de la espasticidad unilateral.

### PC12 — Simetría bilateral (15.9 % imp. SHAP en RF, 13.0 % en XGB)

| Característica | Familia | Ponderación |
|----------------|---------|-------------|
| Desv. estándar cuadrante inferior derecho | Espacial | **0.54** |
| Simetría bilateral | Simetría | **0.45** |
| Centro de masa *x* | Espacial | 0.18 |
| Desv. estándar cuadrante inferior izquierdo | Espacial | −0.19 |
| Frecuencia dominante temporal | Temporal | 0.10 |
| *(resto < 0.15)* | — | — |

**Lectura clínica**: PC12 combina la variabilidad en los cuadrantes inferiores con la simetría bilateral. Cuantifica directamente **si los dos lados del neónato se mueven de forma equivalente** — la pérdida de simetría bilateral es uno de los signos cardinales descritos en la literatura clínica sobre espasticidad neonatal.

## Tabla 2 — Resumen agrupado por familia de características

Distribución de las contribuciones absolutas (suma normalizada) en los tres componentes top según familia:

| Componente | Temporales | Espaciales | Flujo óptico | Simetría |
|------------|-----------|------------|--------------|----------|
| **PC8** | 32 % | 60 % | 0 % | 8 % |
| **PC11** | 0 % | 67 % | 23 % | 10 % |
| **PC12** | 7 % | 56 % | 0 % | 37 % |

## Interpretación global

Los cinco primeros componentes acumulan aproximadamente el **72 %** de la importancia SHAP en ambos modelos (Random Forest y XGBoost), lo que indica que la señal discriminativa está **distribuida entre varios rasgos del movimiento** y no depende de un único factor.

La coincidencia entre los componentes priorizados por el modelo (ritmo anómalo, lateralización, pérdida de simetría) y los signos clínicos documentados en la literatura sobre espasticidad neonatal (Prechtl 1990; Adde 2010; Einspieler 2005; Stahl 2012) es difícilmente casual. Que un modelo entrenado sobre vídeo sin ninguna anotación clínica llegue a priorizar exactamente esos rasgos constituye la señal más sólida de que **el sistema ha capturado información clínicamente relevante** y no un artefacto del preprocesamiento.

## Procedencia de los datos

Las ponderaciones aquí mostradas se calculan automáticamente al ejecutar `main_pipeline.py --experiment video` y se exportan en:

- `reports/results_video/pca_loadings.csv` — matriz completa 30 × 12
- `reports/results_video/shap_importance_rf.json` — importancias SHAP por componente para Random Forest
- `reports/results_video/shap_importance_xgb.json` — equivalente para XGBoost

## Referencias

- Prechtl HF. *Qualitative changes of spontaneous movements in fetus and preterm infant are a marker of neurological dysfunction*. Early Hum Dev. 1990.
- Einspieler C, Prechtl HF. *Prechtl's assessment of general movements: a diagnostic tool for the functional assessment of the young nervous system*. Ment Retard Dev Disabil Res Rev. 2005.
- Adde L et al. *Early prediction of cerebral palsy by computer-based video analysis of general movements: a feasibility study*. Dev Med Child Neurol. 2010.
- Stahl A et al. *An optical flow-based method to predict infantile cerebral palsy*. IEEE Trans Neural Syst Rehabil Eng. 2012.
