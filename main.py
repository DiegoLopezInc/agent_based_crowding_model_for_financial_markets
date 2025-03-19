import numpy as np
import pandas as pd
from src.market import Market
from src.agents import TraderAgent
from src.simulation import MarketSimulation
from src.visualization import plot_market_dynamics

def main():
    # Simulation parameters
    n_agents = 100
    n_steps = 1000
    initial_price = 100.0
    
    # Initialize market simulation
    simulation = MarketSimulation(
        n_agents=n_agents,
        initial_price=initial_price
    )
    
    # Run simulation
    results = simulation.run(n_steps)
    
    # Plot results
    plot_market_dynamics(results)

if __name__ == "__main__":
    main()
