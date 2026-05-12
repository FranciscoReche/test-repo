# Polymarket Earnings Signal Research Framework

Sistema cuantitativo, reproducible y modular para evaluar si mercados de Polymarket relacionados con earnings contienen señal predictiva incremental frente a consenso sell-side y precio tradicional.

## Principio central

El framework **no asume que Polymarket sea fiable**. Separa explícitamente:

1. **Market Signal**: probabilidad bruta Yes/No y dinámica temporal.
2. **Market Quality / Reliability**: liquidez, actividad, reglas, frescura, suficiencia de datos y anomalías espurias.
3. **Adjusted Predictive Signal**: señal ajustada por reliability, divergencia y contexto.

El sistema puede concluir honestamente que un mercado es `inutilizable` y conservarlo para comparar rendimiento con y sin filtros.

## Arquitectura

```text
/project
  /data/{raw,processed,external}
  /notebooks
  /src
    /config                YAML + loader de settings
    /ingestion             loaders CSV validados
    /cleaning              sanity checks pre-modelado
    /mapping               ticker-event mapping
    /feature_engineering   snapshots y features
    /labels                targets de earnings/precio
    /quality               flags, penalties, reliability score
    /models                baselines, modelos temporales, adjusted signal
    /evaluation            reportes y cohort analysis
    /backtest              backtests simples event-driven
    /anomaly_detection     possible informed-flow classifiers
    /visualization         reservado para plots/calibration
    /utils                 logging y fechas
  /tests
  /reports
  /artifacts
```

## Datos esperados

Los CSV en `project/data/raw` deben respetar los schemas definidos en `src/ingestion/loaders.py`:

- `polymarket_markets.csv`: metadatos, reglas, source, cierre, probabilidad actual, volumen, umbral.
- `polymarket_timeseries.csv`: historial timestamp/probabilidad/volumen.
- `earnings.csv`: consenso, reportado, timing, sector, tamaño.
- `equity_prices.csv`: drift pre-evento, retorno post-evento, volumen, volatilidad, expected move.

Se incluye un dataset mínimo de ejemplo para tests y smoke runs.

## Pipeline

```bash
pip install -r project/requirements.txt
python project/scripts/run_pipeline.py --config project/src/config/default.yaml
```

El pipeline ejecuta:

1. carga y normalización de datos;
2. mapping ticker-evento;
3. sanity checks obligatorios y `project/reports/data_quality_report.md`;
4. snapshots temporales T-7d, T-3d, T-1d, T-6h, T-1h, T-30m;
5. features directas, temporales, liquidez/calidad, divergencia, posible flujo informado y contexto;
6. targets beat/miss EPS, reacción de precio y magnitudes;
7. reliability score 0-100 y clasificación;
8. anomaly score y clasificación de flujo sin afirmar insider trading;
9. modelos con validación temporal estricta;
10. backtests simples y reportes.


## Salvaguardas anti-leakage

- La probabilidad raw usada para modelar se toma del último snapshot observado **antes o en** el timestamp del earnings, no de snapshots exportados tras la resolución.
- Los datos de equity se unen por ticker y fecha de evento usando la última fila disponible con `date <= event_ts`, evitando que quarters futuros contaminen eventos anteriores.
- Los labels binarios son nullable: resultados faltantes no se convierten silenciosamente en misses.
- Los baselines calibrados usan la tasa base de la ventana de entrenamiento en cada split temporal, no la tasa de todo el dataset.

## Market Reliability Score

Categorías:

- `0-20`: inutilizable
- `21-40`: muy débil
- `41-60`: usable con mucha cautela
- `61-80`: señal razonable
- `81-100`: mercado de alta calidad

Componentes configurables:

- liquidez total;
- actividad reciente;
- suficiencia temporal;
- estabilidad del pricing;
- claridad de reglas y fuente;
- alineación con consenso;
- frescura cerca del evento;
- evidencia de mercado vivo;
- ausencia de anomalías espurias.

Además genera hard flags y soft penalties para volumen bajo, mercado congelado, pocas observaciones, reglas ambiguas, source unclear, ticker-event mapping problem, duplicados, late creation, jumps espurios, profundidad insuficiente y diseño inutilizable.

## Modelado y validación

`src/models/train.py` compara explícitamente:

1. `consensus_only`
2. `price_only`
3. `polymarket_raw`
4. `polymarket_liquidity_temporal`
5. `full` con reliability, divergencia y anomalías

Modelos incluidos:

- Logistic Regression
- Random Forest
- Gradient Boosting
- baselines naive: always beat, always follow consensus, coin flip calibrado, follow recent price drift
- Ridge/RandomForest/GradientBoosting para regresión

La validación usa splits temporales walk-forward; no hay random shuffle.

## Anomaly detection

El submódulo clasifica:

- `ruido normal`
- `mercado ilíquido poco fiable`
- `flujo direccional normal`
- `posible flujo informado`
- `anomalía fuerte`

Nunca afirma insider trading. Solo marca patrones compatibles con movimientos tardíos, concentración, divergencia y jumps ajustados por liquidez.

## Recalibración

Tras añadir nuevos eventos a `project/data/raw`:

```bash
python project/scripts/recalibrate.py --config project/src/config/default.yaml
```

Esto regenera dataset maestro, quality report, model report y backtest report.

## Tests

```bash
pytest project/tests
```

## Outputs principales por evento

- `raw_polymarket_probability`
- `market_reliability_score`
- `adjusted_beat_probability`
- `adjusted_price_reaction_signal`
- `divergence_score`
- `late_move_score`
- `anomaly_score`
- `market_classification`
- `flow_classification`
