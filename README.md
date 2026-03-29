# ML Detección de Espasticidad en Recién Nacidos

Pipeline de Machine Learning tradicional para el apoyo al diagnóstico temprano de espasticidad en neonatos a partir de análisis de movimiento en vídeo.

## Descripción

Este proyecto aborda la detección temprana de espasticidad mediante el análisis de patrones de movimiento infantil. Utiliza algoritmos de ML tradicional (Regresión Logística, Random Forest, SVM, XGBoost) sobre características cinemáticas extraídas de vídeos.

**Características principales:**
- Clasificación binaria: Normal vs. Patológico (sintético)
- 30 características cinemáticas (flujo óptico, temporales, espaciales)
- Reducción dimensional mediante PCA (30 → 14 componentes, 95.5% varianza)
- Explicabilidad mediante análisis SHAP
- Pipeline reproducible con semilla fija (random_state=42)

## Resultados

| Modelo | Accuracy | Sensibilidad | Especificidad | AUC-ROC |
|--------|----------|--------------|---------------|---------|
| **SVM** | **99.35%** | 98.69% | **100%** | **0.9997** |
| Logistic Regression | 99.02% | 99.35% | 98.70% | 0.9994 |
| Random Forest | 99.02% | 99.35% | 98.70% | 0.9981 |
| XGBoost | 99.02% | 98.69% | 99.35% | 0.9985 |

## Estructura del Proyecto

```
├── main_pipeline.py          # Pipeline principal
├── data/
│   └── features.npz          # Características extraídas
├── models/
│   └── synthetic/            # Modelos entrenados (.pkl)
├── reports/
│   ├── figures/              # 16 figuras de análisis
│   └── results/              # Resultados en CSV y JSON
└── requirements.txt          # Dependencias Python
```

## Requisitos

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
python main_pipeline.py
```

El pipeline ejecuta automáticamente:
1. Carga de datos y generación sintética
2. Preprocesamiento (escalado, PCA)
3. Entrenamiento de 4 modelos
4. Evaluación y validación cruzada
5. Generación de 16 figuras
6. Análisis SHAP
7. Exportación de resultados

## Dataset

El corpus original contiene 767 vídeos de movimiento normal (Kaggle). Se generan 767 muestras sintéticas patológicas mediante perturbaciones clínicamente motivadas:
- Rigidez (reducción de amplitud y variabilidad)
- Temblor (oscilaciones de alta frecuencia)
- Asimetría bilateral

## Autor

**Máximo Fernández Riera**  
TFM - Máster en Ciencia de Datos  
Universitat Oberta de Catalunya (UOC)  
Fecha: 07/02/2026

## Licencia

MIT License
