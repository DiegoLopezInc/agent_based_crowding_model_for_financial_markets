# Agent Based Crowding Model for Financial Markets

This repository implements an agent-based model to simulate and analyze crowding behavior in financial markets. The model focuses on understanding how different trading strategies and market microstructure impact price dynamics and market stability.

## Features

- Multiple agent types with different trading strategies:
  - Momentum traders
  - Mean reversion traders
  - Random traders
- Realistic order book mechanics
- Market depth and liquidity analysis
- Volatility tracking
- Bid-ask spread analysis
- Visualization of market dynamics

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the simulation:
```python
python main.py
```

The simulation will generate plots showing:
- Asset price evolution
- Market volatility
- Bid-ask spread
- Market depth

## Project Structure

- `main.py`: Entry point for running the simulation
- `src/`
  - `market.py`: Implementation of the order book and market mechanics
  - `agents.py`: Different types of trading agents
  - `simulation.py`: Core simulation logic
  - `visualization.py`: Plotting and analysis tools

## License

MIT
