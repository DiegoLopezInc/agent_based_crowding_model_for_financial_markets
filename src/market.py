import numpy as np
from typing import List, Dict

class Market:
    def __init__(self, initial_price: float = 100.0):
        self.price = initial_price
        self.order_book = {'bids': [], 'asks': []}
        self.transaction_history = []
        self.volatility = 0.0
        
    def submit_order(self, agent_id: int, order_type: str, price: float, quantity: float):
        """Submit a new order to the market."""
        order = {
            'agent_id': agent_id,
            'type': order_type,
            'price': price,
            'quantity': quantity,
            'timestamp': len(self.transaction_history)
        }
        
        if order_type == 'bid':
            self.order_book['bids'].append(order)
            self.order_book['bids'].sort(key=lambda x: x['price'], reverse=True)
        else:
            self.order_book['asks'].append(order)
            self.order_book['asks'].sort(key=lambda x: x['price'])
            
        self._match_orders()
        
    def _match_orders(self):
        """Match compatible orders and execute trades."""
        while (len(self.order_book['bids']) > 0 and 
               len(self.order_book['asks']) > 0 and 
               self.order_book['bids'][0]['price'] >= self.order_book['asks'][0]['price']):
            
            bid = self.order_book['bids'][0]
            ask = self.order_book['asks'][0]
            
            # Execute trade
            trade_price = (bid['price'] + ask['price']) / 2
            trade_quantity = min(bid['quantity'], ask['quantity'])
            
            self.price = trade_price
            self.transaction_history.append({
                'price': trade_price,
                'quantity': trade_quantity,
                'buyer_id': bid['agent_id'],
                'seller_id': ask['agent_id'],
                'timestamp': len(self.transaction_history)
            })
            
            # Update orders
            bid['quantity'] -= trade_quantity
            ask['quantity'] -= trade_quantity
            
            # Remove fulfilled orders
            if bid['quantity'] == 0:
                self.order_book['bids'].pop(0)
            if ask['quantity'] == 0:
                self.order_book['asks'].pop(0)
                
        # Update market volatility
        if len(self.transaction_history) > 1:
            prices = [t['price'] for t in self.transaction_history[-10:]]
            self.volatility = np.std(prices) if len(prices) > 1 else 0.0
            
    def get_market_state(self) -> Dict:
        """Return current market state."""
        return {
            'price': self.price,
            'volatility': self.volatility,
            'bid_ask_spread': self._calculate_spread(),
            'order_book_depth': self._calculate_depth()
        }
        
    def _calculate_spread(self) -> float:
        """Calculate the current bid-ask spread."""
        if len(self.order_book['bids']) > 0 and len(self.order_book['asks']) > 0:
            return self.order_book['asks'][0]['price'] - self.order_book['bids'][0]['price']
        return float('inf')
        
    def _calculate_depth(self) -> Dict:
        """Calculate market depth at various price levels."""
        return {
            'bids': sum(order['quantity'] for order in self.order_book['bids']),
            'asks': sum(order['quantity'] for order in self.order_book['asks'])
        }
