from typing import List, Dict
import numpy as np
from tqdm import tqdm
from .market import Market
from .agents import TraderAgent

class MarketSimulation:
    def __init__(self, n_agents: int = 100, initial_price: float = 100.0):
        self.market = Market(initial_price)
        self.agents = [TraderAgent(i) for i in range(n_agents)]
        self.history = []
        
    def run(self, n_steps: int) -> Dict:
        """Run the market simulation for n_steps."""
        for step in tqdm(range(n_steps)):
            # Get current market state
            market_state = self.market.get_market_state()
            
            # Update agents and collect actions
            for agent in self.agents:
                agent.observe_market(market_state)
                action = agent.decide_action(market_state)
                
                if action['action'] != 'hold':
                    self.market.submit_order(
                        agent_id=agent.id,
                        order_type='bid' if action['action'] == 'buy' else 'asks',
                        price=action['price'],
                        quantity=action['quantity']
                    )
            
            # Record market state
            self.history.append({
                'step': step,
                'price': market_state['price'],
                'volatility': market_state['volatility'],
                'bid_ask_spread': market_state['bid_ask_spread'],
                'order_book_depth': market_state['order_book_depth']
            })
            
        return {
            'price_history': [h['price'] for h in self.history],
            'volatility_history': [h['volatility'] for h in self.history],
            'spread_history': [h['bid_ask_spread'] for h in self.history],
            'depth_history': [h['order_book_depth'] for h in self.history]
        }
