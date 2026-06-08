# Proyecto 3: Pricing de Listings Airbnb - Buenos Aires (Nivel Experto)

Pipeline MLOps de pricing dinamico para listings de Airbnb en Buenos Aires. Integra modelo de prediccion de precio con intervalos de confianza empiricos (P10-P90), multiplicadores estacionales desde datos de calendario y monitoreo de drift automatico.

## Contexto

El dataset **Inside Airbnb (Buenos Aires)** contiene **~30,000 listings activos** en la ciudad, con sus caracteristicas estructurales (tipo, capacidad, ubicacion), atributos del host (superhost, antiguedad), precio en ARS, disponibilidad anual y reviews historicas.

El dataset se compone de 2 archivos principales:
- `listings.csv.gz`: cada listing con 75 columnas de atributos.
- `calendar.csv.gz`: precio y disponibilidad por dia (~10 millones de filas, usado para pricing dinamico).

Particularidad: el precio viene como string con formato `$1,234.00`. Requiere limpieza antes de cualquier analisis numerico.

## Contenido del repositorio

```
Proyecto-3-Airbnb-Pricing/
|-- airbnb_experto.ipynb    # Pipeline MLOps de pricing dinamico
|-- README.md
```

## Pipeline experto

| Componente | Descripcion | Resultado |
|------------|-------------|-----------|
| Pipeline atomico | Preprocesamiento + Random Forest sin data leakage | R^2 = 0.70 |
| API con intervalos | Prediccion de precio con rango empirico P10-P90 del bosque | Incertidumbre cuantificada |
| Pricing dinamico | Multiplicadores por dia de semana y mes desde calendar.csv.gz | Sabado +24%, martes -12% |
| Drift monitoring | PSI + Kolmogorov-Smirnov sobre features de entrada | Deteccion de cambios de mercado |

## Hallazgos principales

1. **Palermo domina el mercado** (~26% de los listings). Recoleta y Puerto Madero son los mas caros (mediana R$ 25k-32k).
2. **Distribucion de precio es log-normal**: usar log(precio) como target estabiliza la varianza.
3. **Geografia es la feature mas predictiva**: distancia al obelisco + barrio dominan el modelo.
4. **Elasticidad precio-ocupacion = -0.42** (demanda inelastica): a mayor precio menor ocupacion, pero el ingreso adicional supera la perdida.
5. **Superhost no cobra premium** (~10% mas que no-superhost), pero tiene ratings sistematicamente mas altos (4.9 vs 4.6).
6. **Pricing dinamico**: sabado vale 24% mas que un dia base, martes 12% menos.

## Instalacion

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy xgboost shap mlflow joblib
```

## Como usar

1. Descargar el dataset desde [Inside Airbnb](http://insideairbnb.com/get-the-data/) (seccion Buenos Aires) y colocar `listings.csv.gz` y `calendar.csv.gz` en la misma carpeta del notebook.
2. Abrir `airbnb_experto.ipynb` en Jupyter o VS Code.
3. Ejecutar las celdas en orden.

## Habilidades demostradas

- Pipeline atomico sin data leakage
- API con intervalos de prediccion empiricos (P10-P90 del bosque de arboles)
- Pricing dinamico con multiplicadores estacionales desde datos reales de calendario
- Feature engineering geografico (distancia Haversine, barrio)
- Monitoreo de drift PSI + KS
