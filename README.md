# Proyecto 3: Pricing de Listings Airbnb (Buenos Aires)

Sistema de prediccion de precio para listings de Airbnb en Buenos Aires. Cuatro niveles que cubren desde EDA descriptivo hasta pipeline MLOps con pricing dinamico y monitoreo de drift.

## Contexto

El dataset **Inside Airbnb (Buenos Aires)** contiene **~30,000 listings activos** en la ciudad, con sus caracteristicas estructurales (tipo, capacidad, ubicacion), atributos del host (superhost, antiguedad), precio en ARS, disponibilidad anual y reviews historicas.

El dataset se compone de 3 archivos:
- `listings.csv.gz`: cada listing con 75 columnas de atributos.
- `calendar.csv.gz`: precio y disponibilidad por dia (~10 millones de filas, usado en experto).
- `reviews.csv.gz`: texto de cada review (no usado en este proyecto).

Particularidad: el precio viene como string con formato `$1,234.00`. Requiere limpieza antes de cualquier analisis numerico.

## Estructura del proyecto

```
Proyecto 3 - Airbnb Pricing/
|-- 01-data/                          # listings.csv.gz, calendar.csv.gz, reviews.csv.gz
|-- 02-script/                        # Notebooks Jupyter (.ipynb)
|   |-- airbnb_basico.ipynb
|   |-- airbnb_intermedio.ipynb
|   |-- airbnb_avanzado.ipynb
|   |-- airbnb_experto.ipynb
|-- 03-resultados/                    # Outputs generados
|-- 04-explicacion del codigo/        # PDFs con explicacion linea por linea
|-- README.md
```

## Niveles del proyecto

| Nivel | Tecnica | Output principal | Resultado clave |
|---|---|---|---|
| Basico | EDA descriptivo (8 angulos) | 8 graficos + CSV de KPIs | Palermo 26% de listings, distribucion log-normal |
| Intermedio | 4 modelos de regresion (Linear, Ridge, Lasso, RF) sobre features tabulares + Haversine | Comparacion + feature importance | RF R^2 = 0.62, MAPE 37% |
| Avanzado | XGBoost + tuning + elasticidad precio-ocupacion + segmentacion de hosts (KMeans) + SHAP | Modelo tuned + recomendaciones de precio | R^2 = 0.70, elasticidad -0.42 |
| Experto | Pipeline MLOps + API con intervalo P10-P90 + pricing dinamico desde calendar + drift PSI/KS | Sistema productizable con pricing dinamico | Multiplicadores por dia de semana y mes |

## Hallazgos principales

1. **Palermo domina el mercado** (~26% de los listings). Recoleta y Puerto Madero son los mas caros (mediana R$ 25k-32k).
2. **Distribucion de precio es log-normal**: usar log(precio) como target estabiliza la varianza.
3. **Geografia es la feature mas predictiva**: distancia al obelisco + barrio dominan el modelo.
4. **Elasticidad precio-ocupacion = -0.42** (demanda inelastica): a mayor precio menor ocupacion, pero el ingreso adicional supera la perdida.
5. **Superhost no cobra premium** (~10% mas que no-superhost), pero tiene ratings sistematicamente mas altos (4.9 vs 4.6).
6. **Pricing dinamico**: sabado vale 24% mas que un dia base, martes 12% menos.

## Instalacion

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
# Opcionales:
pip install xgboost shap mlflow joblib
```

## Como usar

1. Descargar el dataset desde [Inside Airbnb](http://insideairbnb.com/get-the-data/) (seccion Buenos Aires) y colocar los archivos `.csv.gz` en `01-data/`.
2. Abrir cualquiera de los notebooks de `02-script/` en Jupyter o VS Code.
3. Ejecutar las celdas en orden.

## Habilidades demostradas

- EDA descriptivo con foco en datos geograficos
- Limpieza de strings de moneda
- Feature engineering: distancia geografica con Haversine, conteo de amenities, ocupacion proxy
- Regresion clasica (Linear, Ridge, Lasso)
- Tree-based models (Random Forest, XGBoost)
- Hyperparameter tuning con RandomizedSearchCV
- Analisis economico: elasticidad precio-demanda (log-log fit)
- Segmentacion de hosts con KMeans
- Interpretabilidad con permutation importance y SHAP
- MLOps: pipeline atomico, API con intervalos de prediccion empiricos (P10-P90 del bosque)
- Pricing dinamico con multiplicadores estacionales
- Monitoreo de drift PSI + KS

## Documentacion detallada

Cada notebook tiene un PDF asociado en `04-explicacion del codigo/` con explicacion linea por linea.
