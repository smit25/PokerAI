from collections import defaultdict
from gym_env import PokerEnv

action_types = PokerEnv.ActionType

class OpponentModel:
    """
    Ultra-fast Bayesian opponent modeling optimized for single-match adaptation.
    Uses aggressive learning rates to form robust models with minimal observations.
    """
    
    def __init__(self):
        # Prior beliefs about opponent types (skewed toward aggressive default)
        self.type_probabilities = {
            'tight_passive': 0.15,
            'tight_aggressive': 0.35,
            'loose_passive': 0.15,
            'loose_aggressive': 0.25,
            'unknown': 0.1
        }
        
        # Action counts by street
        self.actions = {
            0: defaultdict(int),  # Preflop
            1: defaultdict(int),  # Flop
            2: defaultdict(int),  # Turn
            3: defaultdict(int)   # River
        }
        
        # Tracking betting patterns
        self.betting_patterns = {
            'bet_sizes': [],
            'bet_frequency': 0.0,
            'check_frequency': 0.0,
            'fold_to_bet_frequency': 0.0,
            'aggression_factor': 0.5
        }
        
        # Redraw statistics
        self.redraw_used = False
        self.redraw_street = None
        self.redraw_discarded_card = None
        self.redraw_drawn_card = None
        
        # Observation counter
        self.observation_count = 0
        
        # Adaptation rate (extremely aggressive for single-match)
        self.learning_rate = 0.8  # Ultra-high learning rate for very fast adaptation
    
    def update(self, action, street, bet_size=0, pot_size=1, redraw_info=None):
        """Update model with new action observation."""
        self.observation_count += 1
        
        # Update action counts
        self.actions[street][action] += 1
        
        # Track betting patterns with extremely fast learning rates
        if action == action_types.RAISE.value:
            if pot_size > 0:
                self.betting_patterns['bet_sizes'].append(bet_size / pot_size)
            
            # Update aggression - ultra-fast learning rate
            self.betting_patterns['aggression_factor'] = (
                (1 - self.learning_rate) * self.betting_patterns['aggression_factor'] + 
                self.learning_rate * 0.9
            )
            
            # Update bet frequency
            self.betting_patterns['bet_frequency'] = (
                (1 - self.learning_rate) * self.betting_patterns['bet_frequency'] + 
                self.learning_rate * 1.0
            )
            
        elif action == action_types.FOLD.value:
            # Update fold frequency
            self.betting_patterns['fold_to_bet_frequency'] = (
                (1 - self.learning_rate) * self.betting_patterns['fold_to_bet_frequency'] + 
                self.learning_rate * 1.0
            )
            
            # Update aggression (folding indicates less aggression)
            self.betting_patterns['aggression_factor'] = (
                (1 - self.learning_rate) * self.betting_patterns['aggression_factor'] + 
                self.learning_rate * 0.1
            )
            
        elif action == action_types.CHECK.value:
            # Update check frequency
            self.betting_patterns['check_frequency'] = (
                (1 - self.learning_rate) * self.betting_patterns['check_frequency'] + 
                self.learning_rate * 1.0
            )
            
            # Update aggression (checking indicates less aggression)
            self.betting_patterns['aggression_factor'] = (
                (1 - self.learning_rate) * self.betting_patterns['aggression_factor'] + 
                self.learning_rate * 0.3
            )
            
        elif action == action_types.CALL.value:
            # Update aggression (calling indicates medium aggression)
            self.betting_patterns['aggression_factor'] = (
                (1 - self.learning_rate) * self.betting_patterns['aggression_factor'] + 
                self.learning_rate * 0.5
            )
        
        # Update redraw information - very important for opponent modeling
        if action == action_types.DISCARD.value and redraw_info:
            self.redraw_used = True
            self.redraw_street = redraw_info.get('street', street)
            self.redraw_discarded_card = redraw_info.get('discarded_card', -1)
            self.redraw_drawn_card = redraw_info.get('drawn_card', -1)
        
        # Bayesian updating of opponent type probabilities
        self._bayesian_update(action, street)
    
    def _bayesian_update(self, action, street):
        """Update type probabilities using Bayes' rule."""
        # Get likelihood of this action for each opponent type
        likelihoods = {}
        for opponent_type in self.type_probabilities:
            likelihoods[opponent_type] = self._get_action_likelihood(action, street, opponent_type)
        
        # Calculate normalization factor
        total = sum(self.type_probabilities[t] * likelihoods[t] for t in self.type_probabilities)
        
        # Update posterior probabilities
        if total > 0:
            for opponent_type in self.type_probabilities:
                self.type_probabilities[opponent_type] = (
                    self.type_probabilities[opponent_type] * likelihoods[opponent_type] / total
                )
    
    def _get_action_likelihood(self, action, street, opponent_type):
        """Get likelihood of an action given opponent type."""
        # Pre-computed likelihoods based on known player types
        if opponent_type == 'tight_passive':
            # Tight passive players fold and check more, rarely raise
            if action == action_types.FOLD.value:
                return 0.4 if street == 0 else 0.3
            elif action == action_types.CHECK.value:
                return 0.4
            elif action == action_types.CALL.value:
                return 0.3
            elif action == action_types.RAISE.value:
                return 0.1 if street <= 1 else 0.05
            elif action == action_types.DISCARD.value:
                return 0.2
                
        elif opponent_type == 'tight_aggressive':
            # Tight aggressive players fold weak hands but raise strong ones
            if action == action_types.FOLD.value:
                return 0.3 if street == 0 else 0.2
            elif action == action_types.CHECK.value:
                return 0.2
            elif action == action_types.CALL.value:
                return 0.2
            elif action == action_types.RAISE.value:
                return 0.4 if street <= 1 else 0.5
            elif action == action_types.DISCARD.value:
                return 0.3
                
        elif opponent_type == 'loose_passive':
            # Loose passive players call a lot but rarely raise
            if action == action_types.FOLD.value:
                return 0.2 if street == 0 else 0.15
            elif action == action_types.CHECK.value:
                return 0.3
            elif action == action_types.CALL.value:
                return 0.5
            elif action == action_types.RAISE.value:
                return 0.1
            elif action == action_types.DISCARD.value:
                return 0.3
                
        elif opponent_type == 'loose_aggressive':
            # Loose aggressive players raise a lot and rarely fold
            if action == action_types.FOLD.value:
                return 0.1
            elif action == action_types.CHECK.value:
                return 0.15
            elif action == action_types.CALL.value:
                return 0.25
            elif action == action_types.RAISE.value:
                return 0.6 if street <= 1 else 0.5
            elif action == action_types.DISCARD.value:
                return 0.4
        
        else:  # unknown type
            # Equal likelihood for all actions
            return 0.2
            
        # Default likelihood (small but non-zero to avoid multiplication by zero)
        return 0.01
    
    def classify_opponent(self):
        """Get most likely opponent type and confidence."""
        best_type = max(self.type_probabilities, key=self.type_probabilities.get)
        confidence = self.type_probabilities[best_type]
        return best_type, confidence
    
    def get_tendencies(self):
        """Return opponent tendencies for strategy adaptation."""
        # Calculate fold frequency
        fold_frequency = self.betting_patterns['fold_to_bet_frequency']
        
        # Get most recent bet sizes (last 3)
        recent_bet_sizes = self.betting_patterns['bet_sizes'][-3:] if self.betting_patterns['bet_sizes'] else []
        
        # Analyze redraw information
        redraw_analysis = self._analyze_redraw()
        
        # Compile all tendencies
        return {
            'aggression': self.betting_patterns['aggression_factor'],
            'fold_frequency': fold_frequency,
            'bet_sizes': recent_bet_sizes,
            'bet_frequency': self.betting_patterns['bet_frequency'],
            'check_frequency': self.betting_patterns['check_frequency'],
            'redraw_used': self.redraw_used,
            'redraw_analysis': redraw_analysis,
            'most_likely_type': self.classify_opponent()[0],
            'confidence': self.classify_opponent()[1],
            'observation_count': self.observation_count
        }
    
    def get_observation_count(self):
        """Get the number of actions observed so far."""
        return self.observation_count
    
    def _analyze_redraw(self):
        """Analyze opponent's redraw decision for strategic insights."""
        if not self.redraw_used:
            return {'has_redrawn': False}
            
        analysis = {
            'has_redrawn': True,
            'street': self.redraw_street,
            'hand_quality': 'unknown'
        }
        
        # If we don't have card information, return basic analysis
        if self.redraw_discarded_card == -1 or self.redraw_drawn_card == -1:
            return analysis
            
        # Extract rank and suit information
        discarded_rank = self.redraw_discarded_card // 3
        discarded_suit = self.redraw_discarded_card % 3
        drawn_rank = self.redraw_drawn_card // 3
        drawn_suit = self.redraw_drawn_card % 3
        
        # Analyze what this redraw likely means
        # Discarded high card (Ace, 9, 8)
        if discarded_rank >= 6:  # High card
            if drawn_rank >= 6:  # Drew high card
                # Looking for specific high card pattern (e.g., pair)
                analysis['hand_quality'] = 'medium-strong'
                analysis['likely_strategy'] = 'building_specific_hand'
            else:  # Drew low card after discarding high
                # Likely has a drawing hand or specific pattern
                analysis['hand_quality'] = 'medium'
                analysis['likely_strategy'] = 'drawing_to_straight_or_flush'
        else:  # Discarded low card
            if drawn_rank >= 6:  # Drew high card
                # Looking to improve with high card
                analysis['hand_quality'] = 'weak-medium'
                analysis['likely_strategy'] = 'improving_hand_value'
            else:  # Both low cards
                # Likely has very weak hand
                analysis['hand_quality'] = 'weak'
                analysis['likely_strategy'] = 'desperate_improvement'
        
        # Check if they're chasing a flush
        if drawn_suit == discarded_suit:
            analysis['chasing_flush'] = True
        
        return analysis