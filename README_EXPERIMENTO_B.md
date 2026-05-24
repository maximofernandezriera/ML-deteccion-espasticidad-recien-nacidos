# Experimento B: Estudio de Ablación Sistemática

## Descripción

El Experimento B implementa un estudio de ablación sistemática para evaluar la contribución individual de cada tipo de perturbación aplicada en el Modo B (perturbación sobre vídeo crudo). El objetivo es demostrar empíricamente que el modelo detecta manifestaciones clínicas de espasticidad y no simplemente parámetros arbitrarios de las perturbaciones.

## Condiciones de Ablación

El experimento evalúa 8 condiciones diferentes:

| ID | Condición | Perturbación excluida | Descripción |
|---|---|---|---|
| A | Completo | Ninguna | Todas las perturbaciones activas (baseline) |
| B | Sin rigidez | `amplitude_scale` | Excluye reducción de amplitud del movimiento |
| C | Sin temblor | `apply_tremor` | Excluye temblor sinusoidal 3-6 Hz |
| D | Sin asimetría | `apply_asymmetry` | Excluye enmascaramiento asimétrico |
| E | Sin motion_blur | `apply_motion_blur` | Excluye difuminado direccional |
| F | Sin jitter | `apply_temporal_jitter` | Excluye fluctuación temporal |
| G | Sin ruido | `apply_global_noise` | Excluye ruido gaussiano |
| H | Sin artifacts | `apply_common_acquisition_artifacts` | Excluye artifacts de adquisición |

## Uso

### Ejecutar el experimento completo

```bash
cd /home/maximo/Dropbox/UOC/TMF_bruto/ML-deteccion-espasticidad-recien-nacidos
python3 main_pipeline_29032026_1516.py --experiment=ablation
```

### Ejecutar tests de verificación

```bash
python3 test_ablation.py
```

### Ejecutar el pipeline dual normal (sin ablación)

```bash
python3 main_pipeline_29032026_1516.py --experiment=dual
# o simplemente:
python3 main_pipeline_29032026_1516.py
```

## Outputs Generados

El experimento genera los siguientes outputs en `reports/ablation_YYYYMMDD_HHMMSS/`:

### Archivos principales

- **`summary_table.csv`**: Tabla con accuracy, AUC-ROC, F1-Score y tiempo por condición y modelo
- **`impact_analysis.csv`**: Análisis de impacto de cada perturbación excluida
- **`ablation_analysis.png`**: Visualización comparativa (2 gráficos)
- **`ablation_analysis.json`**: Resultados completos en formato JSON
- **`tabla_latex.tex`**: Tabla LaTeX lista para copiar a la tesis

### Estructura de directorios

```
reports/ablation_YYYYMMDD_HHMMSS/
├── summary_table.csv
├── impact_analysis.csv
├── ablation_analysis.png
├── ablation_analysis.json
├── tabla_latex.tex
├── A_completo/
│   ├── models/
│   ├── figures/
│   └── results/
├── B_sin_rigidez/
│   ├── models/
│   ├── figures/
│   └── results/
├── C_sin_temblor/
│   └── ...
└── ... (hasta H_sin_artifacts)
```

## Métricas Reportadas

Para cada condición se reportan:

- **Accuracy**: Precisión global del modelo
- **AUC-ROC**: Área bajo la curva ROC
- **F1-Score**: Media armónica de precisión y recall
- **Δ (pp)**: Diferencia en puntos porcentuales respecto al baseline completo
- **Impacto relativo**: Porcentaje de degradación respecto al baseline

## Hipótesis Esperadas

Si el modelo detecta manifestaciones clínicas reales (y no solo artefactos):

1. **Rigidez y temblor** deben mostrar el mayor impacto al ser excluidos (Δ > -10 pp)
2. **Asimetría** debe mostrar impacto moderado (Δ ~ -6 a -10 pp)
3. **Ruido y artifacts** deben mostrar impacto mínimo (Δ ~ 0 a -3 pp)

El orden de importancia debe ser consistente con la literatura semiológica sobre espasticidad neonatal.

## Tiempo de Ejecución

- **Estimado**: 15-20 minutos en CPU estándar
- **8 condiciones** × 4 modelos × validación cruzada
- **~32 ejecuciones completas** del pipeline

## Interpretación de Resultados

### Criterios de éxito

✅ **El modelo es específico a manifestaciones clínicas si:**
- Las perturbaciones clínicamente fundamentadas (rigidez, temblor) tienen mayor impacto
- Las perturbaciones no-clínicas (ruido, artifacts) tienen impacto mínimo
- El ranking de importancia es coherente con la semiología documentada

❌ **El modelo detecta artefactos si:**
- Todas las perturbaciones tienen impacto similar
- El ruido gaussiano tiene más impacto que la rigidez
- No hay diferenciación clara entre perturbaciones clínicas y no-clínicas

## Integración con la Tesis

El archivo `tabla_latex.tex` contiene código LaTeX listo para copiar al Capítulo 6 de la tesis. Ejemplo:

```latex
\begin{table}[H]
\centering
\caption{Resultados del estudio de ablación sistemática (SVM).}
\label{tab:ablation_results}
\begin{tabular}{lcccc}
\toprule
\textbf{Condición} & \textbf{Perturbación excluida} & \textbf{Accuracy} & \textbf{$\Delta$ (pp)} & \textbf{Impacto rel.} \\
\midrule
Completo & Ninguna & 0.8274 & - & 100\% \\
...
\bottomrule
\end{tabular}
\end{table}
```

## Notas Técnicas

- El experimento utiliza `RANDOM_STATE=42` para reproducibilidad
- Solo se ejecuta sobre el modo `raw_video` (Modo B)
- Cada condición genera su propio dataset sintético
- Los modelos se entrenan independientemente para cada condición
- Se usa SVM como modelo de referencia para el análisis de impacto

## Troubleshooting

### Error: "No se encontró el corpus de vídeo crudo"

Verifica que existe el archivo en una de estas rutas:
- `/home/maximo/Dropbox/UOC/TMF_bruto/ML-deteccion-espasticidad-recien-nacidos/data/raw/data_100_50_50.npz`
- `/home/maximo/Dropbox/UOC/TMF_bruto/data/raw/data_100_50_50.npz`
- `/home/maximo/Dropbox/UOC/TMF_bruto/FUENTE/kaggle_data/data_100_50_50.npz`

### Error de memoria

Si el sistema se queda sin memoria, reduce el número de condiciones editando la lista `ablation_conditions` en la función `run_ablation_experiment()`.

### Ejecución muy lenta

El experimento completo tarda ~15-20 minutos. Si necesitas resultados más rápidos, ejecuta solo las condiciones críticas (A, B, C, D).
