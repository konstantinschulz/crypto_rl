# Roadmap

## Microstructure and Order Flow

Raw volume is a blunt instrument. Deconstructing the V in your OHLCV data exposes the actual aggression behind price movements.

Taker Buy Volume Ratio: Most exchange APIs provide the subset of volume executed by aggressive market buyers. Dividing Taker Buy Volume by Total Volume calculates a normalized Cumulative Volume Delta (CVD). This allows the agent to distinguish between a price pump driven by aggressive market buying versus a fake-out caused by the withdrawal of passive limit sell orders.

Relative Volume (RVOL): Divide the current 1-minute volume by the rolling 24-hour average. This immediately isolates anomalous institutional momentum from standard baseline trading noise, giving the agent a cleaner trigger for breakouts.
