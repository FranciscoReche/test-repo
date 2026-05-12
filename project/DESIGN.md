# Diseño técnico

## Objetivo de inferencia

Evaluar si Polymarket aporta señal predictiva incremental en earnings sin asumir eficiencia ni fiabilidad. La unidad primaria es un `market_id` mapeado a un `ticker` y `event_ts` de earnings.

## Capas

### 1. Market Signal

- Probabilidad bruta Yes/No.
- Log-odds, distancia a 50%, snapshots T-7d/T-3d/T-1d/T-6h/T-1h/T-30m.
- Velocidad, persistencia, reversals y movimiento tardío.

### 2. Market Quality / Reliability

- Hard flags para ruido estructural y problemas de datos.
- Soft scores normalizados para liquidez, actividad reciente, suficiencia temporal, estabilidad, claridad, alineación, frescura, vida de mercado y limpieza de anomalías.
- Score final 0-100 con penalización por flags.

### 3. Adjusted Predictive Signal

- Combina probabilidad bruta, consenso neutral, reliability y divergencia.
- Produce `adjusted_beat_probability` y `adjusted_price_reaction_signal`.
- Penaliza mercados de baja calidad para evitar falsas señales.

## Validación

- Splits temporales walk-forward por `earnings_event_ts`.
- Comparación incremental: consenso, precio, Polymarket raw, Polymarket+liquidez+temporalidad, modelo completo.
- Cohorts por reliability, liquidez, sector y horario.

## Sanity checks obligatorios

Antes del modelado se reportan problemas de:

- umbral obsoleto vs consenso;
- volumen bajo con grandes saltos;
- falta de actividad cerca del evento;
- gaps temporales;
- duplicados ticker-evento;
- reglas/fuente ambiguas;
- timing cierre-evento desalineado.

## Postura sobre anomalías

El módulo de anomalías no etiqueta insider trading. Solo clasifica patrones como ruido, flujo direccional, posible flujo informado o anomalía fuerte según intensidad tardía, concentración, divergencia y jump ajustado por liquidez.
