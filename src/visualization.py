import matplotlib.pyplot as plt
from typing import Dict

def plot_market_dynamics(results: Dict):
    """Plot key market dynamics from simulation results."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot price history
    ax1.plot(results['price_history'])
    ax1.set_title('Asset Price Over Time')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Price')
    
    # Plot volatility
    ax2.plot(results['volatility_history'])
    ax2.set_title('Market Volatility')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Volatility')
    
    # Plot bid-ask spread
    ax3.plot(results['spread_history'])
    ax3.set_title('Bid-Ask Spread')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Spread')
    
    # Plot market depth
    bids = [d['bids'] for d in results['depth_history']]
    asks = [d['asks'] for d in results['depth_history']]
    ax4.plot(bids, label='Bids')
    ax4.plot(asks, label='Asks')
    ax4.set_title('Market Depth')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Volume')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()
