import numpy as np
from typing import Dict

class TraderAgent:
    def __init__(self, agent_id: int, initial_cash: float = 10000.0, initial_assets: float = 100.0):
        self.id = agent_id
        self.cash = initial_cash
        self.assets = initial_assets
        self.strategy = np.random.choice(['momentum', 'mean_reversion', 'random'])
        self.memory_length = np.random.randint(5, 20)
        self.price_history = []
        
    def observe_market(self, market_state: Dict):
        """Update agent's market view."""
        self.price_history.append(market_state['price'])
        if len(self.price_history) > self.memory_length:
            self.price_history.pop(0)
            
    def decide_action(self, market_state: Dict) -> Dict:
        """Decide trading action based on strategy and market state."""
        if len(self.price_history) < 2:
            return {'action': 'hold', 'price': 0, 'quantity': 0}
            
        current_price = market_state['price']
        
        if self.strategy == 'momentum':
            return self._momentum_strategy(current_price)
        elif self.strategy == 'mean_reversion':
            return self._mean_reversion_strategy(current_price)
        else:
            return self._random_strategy(current_price)
            
    def _momentum_strategy(self, current_price: float) -> Dict:
        """Follow price trends."""
        price_change = current_price - self.price_history[-2]
        
        if price_change > 0 and self.cash > 0:
            return {
                'action': 'buy',
                'price': current_price * 1.001,  # Slightly above market price
                'quantity': min(1, self.cash / current_price)
            }
        elif price_change < 0 and self.assets > 0:
            return {
                'action': 'sell',
                'price': current_price * 0.999,  # Slightly below market price
                'quantity': min(1, self.assets)
            }
        return {'action': 'hold', 'price': 0, 'quantity': 0}
        
    def _mean_reversion_strategy(self, current_price: float) -> Dict:
        """Trade based on price deviation from mean."""
        mean_price = np.mean(self.price_history)
        
        if current_price > mean_price * 1.02 and self.assets > 0:
            return {
                'action': 'sell',
                'price': current_price * 0.999,
                'quantity': min(1, self.assets)
            }
        elif current_price < mean_price * 0.98 and self.cash > 0:
            return {
                'action': 'buy',
                'price': current_price * 1.001,
                'quantity': min(1, self.cash / current_price)
            }
        return {'action': 'hold', 'price': 0, 'quantity': 0}
        
    def _random_strategy(self, current_price: float) -> Dict:
        """Random trading strategy."""
        if np.random.random() < 0.3:  # 30% chance of trading
            if np.random.random() < 0.5 and self.cash > 0:
                return {
                    'action': 'buy',
                    'price': current_price * (1 + np.random.uniform(0, 0.02)),
                    'quantity': min(1, self.cash / current_price)
                }
            elif self.assets > 0:
                return {
                    'action': 'sell',
                    'price': current_price * (1 - np.random.uniform(0, 0.02)),
                    'quantity': min(1, self.assets)
                }
        return {'action': 'hold', 'price': 0, 'quantity': 0}
        
    def update_portfolio(self, trade_result: Dict):
        """Update agent's portfolio after trade execution."""
        if trade_result['action'] == 'buy':
            self.cash -= trade_result['price'] * trade_result['quantity']
            self.assets += trade_result['quantity']
        elif trade_result['action'] == 'sell':
            self.cash += trade_result['price'] * trade_result['quantity']
            self.assets -= trade_result['quantity']
