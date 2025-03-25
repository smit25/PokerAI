import random
from gym_env import PokerEnv

action_types = PokerEnv.ActionType

class DecisionEngine:
    """
    Advanced decision engine for 27-card poker with single-redraw option.
    Implements Meta-Strategy Adaptation System (MSAS) for rapid opponent exploitation.
    """
    
    def __init__(self, hand_evaluator, redraw_strategy, opponent_model, time_manager):
        self.hand_evaluator = hand_evaluator
        self.redraw_strategy = redraw_strategy
        self.opponent_model = opponent_model
        self.time_manager = time_manager
        
        # Initialize strategy portfolio
        self.strategies = {
            'aggressive': AggressiveStrategy(self.hand_evaluator),
            'loose_aggressive': LooseAggressiveStrategy(self.hand_evaluator),
            'exploitative': ExploitativeStrategy(self.hand_evaluator),
            'super_aggressive': SuperAggressiveStrategy(self.hand_evaluator)
        }
        
        # Start with aggressive strategy by default 
        # This is best for fast adaptation in a single match
        self.active_strategy = 'aggressive'
        
        # Strategy switching parameters
        self.min_confidence_for_switch = 0.25  # Very low threshold for ultra-fast adaptation
        self.strategy_performance = {}
        
        # Single-match tracking
        self.has_used_redraw = False
    
    def make_decision(self, obs):
        """
        Main decision function that processes observation and returns action.
        
        Args:
            obs: Observation dictionary from the environment
            
        Returns:
            Tuple of (action_type, raise_amount, card_to_discard)
        """
        # Track time for this decision
        remaining_time = self.time_manager.time_remaining()
        
        # First priority: consider redraw (this is a critical decision)
        if obs["valid_actions"][action_types.DISCARD.value]:
            should_redraw, card_idx = self.redraw_strategy.strategic_redraw(
                obs["my_cards"], 
                obs.get("community_cards", []), 
                obs["street"], 
                "SB" if obs["acting_agent"] == 0 else "BB",
                self.opponent_model
            )
            
            if should_redraw:
                self.has_used_redraw = True
                return (action_types.DISCARD.value, 0, card_idx)
        
        # Check if we have enough data to potentially switch strategies
        # Use ultra-low observation threshold (2) to adapt extremely quickly
        if self.opponent_model.get_observation_count() >= 2:
            self._update_strategy_selection()
        
        # Use current active strategy to make a decision
        action = self.strategies[self.active_strategy].get_action(
            obs, 
            self.opponent_model.get_tendencies(),
            time_remaining=remaining_time
        )
        
        # Track strategy performance
        self._track_performance(action, obs)
        
        return action
    
    def _update_strategy_selection(self):
        """Update active strategy based on opponent model."""
        # Get opponent classification with confidence
        opponent_type, confidence = self.opponent_model.classify_opponent()
        
        # Only switch if we have minimum confidence
        if confidence < self.min_confidence_for_switch:
            return
        
        # Select best counter-strategy based on opponent type
        new_strategy = 'aggressive'  # Default
        
        if opponent_type == 'tight_passive':
            new_strategy = 'super_aggressive'  # Extremely aggressive against tight passive
        elif opponent_type == 'tight_aggressive':
            new_strategy = 'exploitative'      # Targeted exploitation against tight aggressive
        elif opponent_type == 'loose_passive':
            new_strategy = 'aggressive'        # Standard aggressive against loose passive
        elif opponent_type == 'loose_aggressive':
            new_strategy = 'exploitative'      # Targeted exploitation against loose aggressive
        
        # Additional adjustment based on fold frequency
        fold_frequency = self.opponent_model.get_tendencies().get('fold_frequency', 0.5)
        if fold_frequency > 0.6:
            # If opponent folds a lot, prefer aggressive approach
            if new_strategy != 'super_aggressive':
                new_strategy = 'aggressive'
        
        # Switch if different or if performance indicates we should
        if (new_strategy != self.active_strategy or
            self.strategy_performance.get(new_strategy, 0) > 
            self.strategy_performance.get(self.active_strategy, 0)):
            self.active_strategy = new_strategy
    
    def _track_performance(self, action, obs):
        """Simple heuristic performance tracking for strategies."""
        # More aggressive actions get positive reinforcement
        if action[0] == action_types.RAISE.value:
            self.strategy_performance[self.active_strategy] = self.strategy_performance.get(self.active_strategy, 0) + 0.1
        elif action[0] == action_types.FOLD.value:
            self.strategy_performance[self.active_strategy] = self.strategy_performance.get(self.active_strategy, 0) - 0.05


class BaseStrategy:
    """Base strategy class with common functionality."""
    
    def __init__(self, hand_evaluator):
        self.hand_evaluator = hand_evaluator
        
        # Bet sizing templates (pot fractions)
        self.bet_sizes = {
            'small': 0.5,   # Half pot
            'medium': 0.75, # 3/4 pot
            'large': 1.0,   # Full pot
            'overbet': 1.5  # 1.5x pot
        }
    
    def _make_raise(self, obs, pot_fraction):
        """Make a raise of specified pot fraction."""
        pot_size = obs["my_bet"] + obs["opp_bet"]
        min_raise = obs.get("min_raise", 1)
        max_raise = obs.get("max_raise", 100)
        
        target_size = max(min_raise, int(pot_size * pot_fraction))
        bet_size = min(max_raise, target_size)
        
        return (action_types.RAISE.value, bet_size, -1)

class AggressiveStrategy(BaseStrategy):
    """Aggressive strategy that puts maximum pressure on opponents."""
    
    def __init__(self, hand_evaluator):
        super().__init__(hand_evaluator)
        
        # Aggression parameters
        self.bet_frequency = 0.8    # Bet/raise 80% of hands
        self.large_bet_threshold = 0.6  # Use large bets with top 60% of range
        self.bluff_frequency = 0.4  # Bluff 40% of weak hands
        
        # Bet sizings (larger than GTO)
        self.sizings = {
            'preflop': 1.0,  # Raise to 4bb preflop
            'flop': 0.9,     # 90% pot on flop
            'turn': 1.1,     # 110% pot on turn
            'river': 1.5     # 150% pot on river
        }
    
    def get_action(self, obs, opponent_tendencies, time_remaining=0.5):
        """Get action using aggressive strategy."""
        my_cards = obs["my_cards"]
        community_cards = obs.get("community_cards", [])
        street = obs["street"]
        position = "SB" if obs["acting_agent"] == 0 else "BB"
        facing_bet = obs["opp_bet"] > obs["my_bet"]
        valid_actions = obs["valid_actions"]
        
        # Get hand strength and randomization factor
        hand_strength = self.hand_evaluator.evaluate(my_cards, community_cards)
        random_factor = random.random()
        
        # Adjust aggression based on opponent fold frequency
        fold_frequency = opponent_tendencies.get('fold_frequency', 0.5)
        
        # Increase bluffing against players who fold a lot
        adjusted_bluff_freq = self.bluff_frequency
        if fold_frequency > 0.6:
            adjusted_bluff_freq = self.bluff_frequency + (fold_frequency - 0.6) * 0.5
        
        # Preflop strategy
        if street == 0:
            return self._preflop_aggressive(obs, hand_strength, random_factor, position, adjusted_bluff_freq)
        
        # Postflop strategy
        return self._postflop_aggressive(obs, hand_strength, random_factor, position, street, adjusted_bluff_freq)
    
    def _preflop_aggressive(self, obs, hand_strength, random_factor, position, bluff_freq):
        """Aggressive preflop strategy."""
        facing_bet = obs["opp_bet"] > obs["my_bet"]
        valid_actions = obs["valid_actions"]
        
        if position == "SB":
            # Small blind strategy - raise a lot
            if not facing_bet:
                # Open raising - very wide range
                if hand_strength > 0.3 or random_factor < bluff_freq:
                    if valid_actions[action_types.RAISE.value]:
                        return self._make_raise(obs, self.sizings['preflop'])
                else:
                    # Fold very weak hands
                    return (action_types.FOLD.value, 0, -1)
            else:
                # Facing a 3-bet - still aggressive
                if hand_strength > 0.5 or (hand_strength > 0.3 and random_factor < 0.3):
                    # 4-bet or call with good hands
                    if hand_strength > 0.7 or random_factor < 0.4:
                        if valid_actions[action_types.RAISE.value]:
                            return self._make_raise(obs, 2.5)
                    if valid_actions[action_types.CALL.value]:
                        return (action_types.CALL.value, 0, -1)
                return (action_types.FOLD.value, 0, -1)
        else:  # Big blind
            if not facing_bet:
                # Check or raise when not facing a bet
                if hand_strength > 0.5 and random_factor < 0.3:
                    if valid_actions[action_types.RAISE.value]:
                        return self._make_raise(obs, self.sizings['preflop'])
                if valid_actions[action_types.CHECK.value]:
                    return (action_types.CHECK.value, 0, -1)
            else:
                # Facing a raise - 3-bet a lot
                if hand_strength > 0.4 or (hand_strength > 0.2 and random_factor < bluff_freq):
                    if hand_strength > 0.6 or random_factor < 0.4:
                        if valid_actions[action_types.RAISE.value]:
                            return self._make_raise(obs, 3.0)
                    if valid_actions[action_types.CALL.value]:
                        return (action_types.CALL.value, 0, -1)
                return (action_types.FOLD.value, 0, -1)
        
        # Default
        if valid_actions[action_types.CHECK.value]:
            return (action_types.CHECK.value, 0, -1)
        else:
            return (action_types.FOLD.value, 0, -1)
    
    def _postflop_aggressive(self, obs, hand_strength, random_factor, position, street, bluff_freq):
        """Aggressive postflop strategy."""
        facing_bet = obs["opp_bet"] > obs["my_bet"]
        valid_actions = obs["valid_actions"]
        
        # Sizing based on street - aggressive sizing
        if street == 1:
            sizing = self.sizings['flop']
        elif street == 2:
            sizing = self.sizings['turn']
        else:
            sizing = self.sizings['river']
        
        # When facing a bet
        if facing_bet:
            # Raising range wider than GTO
            if hand_strength > 0.5 or (hand_strength > 0.3 and random_factor < bluff_freq):
                if hand_strength > 0.7 or random_factor < 0.3:
                    if valid_actions[action_types.RAISE.value]:
                        # Use bigger raises
                        return self._make_raise(obs, sizing * 3.0)
                if valid_actions[action_types.CALL.value]:
                    return (action_types.CALL.value, 0, -1)
            return (action_types.FOLD.value, 0, -1)
        
        # When first to act
        else:
            # Bet a very wide range
            if hand_strength > 0.3 or random_factor < bluff_freq:
                if valid_actions[action_types.RAISE.value]:
                    # Use big sizing with strong hands
                    if hand_strength > self.large_bet_threshold:
                        return self._make_raise(obs, sizing * 1.2)
                    return self._make_raise(obs, sizing)
            
            if valid_actions[action_types.CHECK.value]:
                return (action_types.CHECK.value, 0, -1)
            else:
                return (action_types.FOLD.value, 0, -1)

class LooseAggressiveStrategy(BaseStrategy):
    """Very loose aggressive strategy that plays many hands with aggression."""
    
    def __init__(self, hand_evaluator):
        super().__init__(hand_evaluator)
        
        # Aggression parameters - even more aggressive
        self.bet_frequency = 0.9    # Bet/raise 90% of hands
        self.large_bet_threshold = 0.5  # Use large bets with top 50% of range
        self.bluff_frequency = 0.6  # Bluff 60% of weak hands
        
        # Bet sizings (even larger)
        self.sizings = {
            'preflop': 1.2,  # Raise to 4.8bb preflop
            'flop': 1.0,     # 100% pot on flop
            'turn': 1.3,     # 130% pot on turn
            'river': 1.8     # 180% pot on river
        }
        
        # Hand range adjustments
        self.min_playable_strength = 0.2  # Play any hand above 20% strength
    
    def get_action(self, obs, opponent_tendencies, time_remaining=0.5):
        """Get action using loose aggressive strategy."""
        my_cards = obs["my_cards"]
        community_cards = obs.get("community_cards", [])
        street = obs["street"]
        position = "SB" if obs["acting_agent"] == 0 else "BB"
        facing_bet = obs["opp_bet"] > obs["my_bet"]
        valid_actions = obs["valid_actions"]
        
        # Get hand strength and randomization factor
        hand_strength = self.hand_evaluator.evaluate(my_cards, community_cards)
        random_factor = random.random()
        
        # Adjust bluffing based on opponent fold frequency
        fold_frequency = opponent_tendencies.get('fold_frequency', 0.5)
        adjusted_bluff_freq = self.bluff_frequency
        if fold_frequency > 0.5:
            adjusted_bluff_freq = self.bluff_frequency + (fold_frequency - 0.5) * 0.6
        
        # Preflop strategy
        if street == 0:
            return self._preflop_loose_aggressive(obs, hand_strength, random_factor, position, adjusted_bluff_freq)
        
        # Postflop strategy
        return self._postflop_loose_aggressive(obs, hand_strength, random_factor, position, street, adjusted_bluff_freq)
    
    def _preflop_loose_aggressive(self, obs, hand_strength, random_factor, position, bluff_freq):
        """Loose aggressive preflop strategy."""
        facing_bet = obs["opp_bet"] > obs["my_bet"]
        valid_actions = obs["valid_actions"]
        
        if position == "SB":
            # Small blind strategy - raise almost everything
            if not facing_bet:
                # Open raising - extremely wide range
                if hand_strength > self.min_playable_strength or random_factor < bluff_freq:
                    if valid_actions[action_types.RAISE.value]:
                        return self._make_raise(obs, self.sizings['preflop'])
                    elif valid_actions[action_types.CALL.value]:
                        return (action_types.CALL.value, 0, -1)
                return (action_types.FOLD.value, 0, -1)
            else:
                # Facing a 3-bet - still play very wide
                if hand_strength > 0.4 or (hand_strength > 0.25 and random_factor < 0.4):
                    # 4-bet or call with decent hands
                    if hand_strength > 0.6 or random_factor < 0.5:
                        if valid_actions[action_types.RAISE.value]:
                            return self._make_raise(obs, 2.5)
                    if valid_actions[action_types.CALL.value]:
                        return (action_types.CALL.value, 0, -1)
                return (action_types.FOLD.value, 0, -1)
        else:  # Big blind
            if not facing_bet:
                # Check or raise when not facing a bet
                if hand_strength > 0.4 or random_factor < 0.4:
                    if valid_actions[action_types.RAISE.value]:
                        return self._make_raise(obs, self.sizings['preflop'])
                if valid_actions[action_types.CHECK.value]:
                    return (action_types.CHECK.value, 0, -1)
            else:
                # Facing a raise - 3-bet very often
                if hand_strength > 0.3 or (hand_strength > 0.1 and random_factor < bluff_freq):
                    if hand_strength > 0.5 or random_factor < 0.5:
                        if valid_actions[action_types.RAISE.value]:
                            return self._make_raise(obs, 3.5)
                    if valid_actions[action_types.CALL.value]:
                        return (action_types.CALL.value, 0, -1)
                return (action_types.FOLD.value, 0, -1)
        
        # Default
        if valid_actions[action_types.CHECK.value]:
            return (action_types.CHECK.value, 0, -1)
        else:
            return (action_types.FOLD.value, 0, -1)
    
    def _postflop_loose_aggressive(self, obs, hand_strength, random_factor, position, street, bluff_freq):
        """Loose aggressive postflop strategy."""
        facing_bet = obs["opp_bet"] > obs["my_bet"]
        valid_actions = obs["valid_actions"]
        
        # Sizing based on street - very aggressive sizing
        if street == 1:
            sizing = self.sizings['flop']
        elif street == 2:
            sizing = self.sizings['turn']
        else:
            sizing = self.sizings['river']
        
        # When facing a bet
        if facing_bet:
            # Raise or call with wide range
            if hand_strength > 0.4 or (hand_strength > 0.2 and random_factor < bluff_freq):
                if hand_strength > 0.6 or random_factor < 0.4:
                    if valid_actions[action_types.RAISE.value]:
                        # Use very big raises
                        return self._make_raise(obs, sizing * 3.5)
                if valid_actions[action_types.CALL.value]:
                    return (action_types.CALL.value, 0, -1)
            return (action_types.FOLD.value, 0, -1)
        
        # When first to act
        else:
            # Bet almost any hand
            if hand_strength > 0.2 or random_factor < bluff_freq:
                if valid_actions[action_types.RAISE.value]:
                    # Use big sizing with decent hands, huge with strong hands
                    if hand_strength > 0.7:
                        return self._make_raise(obs, sizing * 1.5)
                    return self._make_raise(obs, sizing)
            
            if valid_actions[action_types.CHECK.value]:
                return (action_types.CHECK.value, 0, -1)
            else:
                return (action_types.FOLD.value, 0, -1)

class ExploitativeStrategy(BaseStrategy):
    """Adaptive exploitative strategy that targets opponent weaknesses."""
    
    def __init__(self, hand_evaluator):
        super().__init__(hand_evaluator)
        
        # Base parameters (will be adjusted dynamically)
        self.value_bet_threshold = 0.6
        self.bluff_frequency = 0.3
        self.defense_frequency = 0.5
        
        # Default bet sizings (will be adjusted based on opponent)
        self.sizings = {
            'preflop': 0.75,
            'flop': 0.7,
            'turn': 0.8,
            'river': 1.0
        }
        
        # Counter-strategy parameters
        self.counter_strategies = {
            'vs_tight_passive': {
                'bet_threshold': 0.4,     # Bet more hands against tight passive
                'bluff_frequency': 0.5,   # Bluff a lot
                'sizing_multiplier': 0.9  # Smaller bets get called more
            },
            'vs_tight_aggressive': {
                'bet_threshold': 0.7,     # Only bet strong hands
                'bluff_frequency': 0.2,   # Bluff less
                'sizing_multiplier': 1.3  # Bigger bets when we do bet
            },
            'vs_loose_passive': {
                'bet_threshold': 0.5,     # Value bet more hands
                'bluff_frequency': 0.1,   # Almost never bluff
                'sizing_multiplier': 1.2  # Bigger value bets
            },
            'vs_loose_aggressive': {
                'bet_threshold': 0.7,     # Only bet strong hands
                'bluff_frequency': 0.1,   # Almost never bluff
                'sizing_multiplier': 1.4  # Much bigger bets for value
            }
        }
    
    def get_action(self, obs, opponent_tendencies, time_remaining=0.5):
        """Get action using exploitative strategy."""
        my_cards = obs["my_cards"]
        community_cards = obs.get("community_cards", [])
        street = obs["street"]
        position = "SB" if obs["acting_agent"] == 0 else "BB"
        valid_actions = obs["valid_actions"]
        
        # Get opponent type and tendencies
        opponent_type = opponent_tendencies.get('most_likely_type', 'unknown')
        fold_frequency = opponent_tendencies.get('fold_frequency', 0.5)
        aggression = opponent_tendencies.get('aggression', 0.5)
        
        # Select counter-strategy based on opponent type
        counter_strat = self._select_counter_strategy(opponent_type, fold_frequency, aggression)
        
        # Get hand strength
        hand_strength = self.hand_evaluator.evaluate(my_cards, community_cards)
        
        # Get randomization factor
        random_factor = random.random()
        
        # Preflop strategy
        if street == 0:
            return self._preflop_exploitative(obs, hand_strength, random_factor, position, counter_strat)
        
        # Postflop strategy
        return self._postflop_exploitative(obs, hand_strength, random_factor, position, street, counter_strat)
    
    def _select_counter_strategy(self, opponent_type, fold_frequency, aggression):
        """Select appropriate counter-strategy based on opponent profile."""
        # Map opponent type to counter strategy
        if opponent_type == 'tight_passive':
            counter_strat = self.counter_strategies['vs_tight_passive']
        elif opponent_type == 'tight_aggressive':
            counter_strat = self.counter_strategies['vs_tight_aggressive']
        elif opponent_type == 'loose_passive':
            counter_strat = self.counter_strategies['vs_loose_passive']
        elif opponent_type == 'loose_aggressive':
            counter_strat = self.counter_strategies['vs_loose_aggressive']
        else:
            # Default counter strategy based on tendencies
            if aggression > 0.7:
                if fold_frequency > 0.6:
                    counter_strat = self.counter_strategies['vs_tight_aggressive']
                else:
                    counter_strat = self.counter_strategies['vs_loose_aggressive']
            else:
                if fold_frequency > 0.6:
                    counter_strat = self.counter_strategies['vs_tight_passive']
                else:
                    counter_strat = self.counter_strategies['vs_loose_passive']
        
        # Fine-tune counter strategy based on fold frequency
        # If they fold a lot, bluff more
        if fold_frequency > 0.7:
            counter_strat = counter_strat.copy()
            counter_strat['bluff_frequency'] = min(0.7, counter_strat['bluff_frequency'] * 1.5)
        
        # If they never fold, value bet more and never bluff
        elif fold_frequency < 0.3:
            counter_strat = counter_strat.copy()
            counter_strat['bluff_frequency'] = counter_strat['bluff_frequency'] * 0.5
            counter_strat['bet_threshold'] = min(0.8, counter_strat['bet_threshold'] + 0.1)
        
        return counter_strat
    
    def _preflop_exploitative(self, obs, hand_strength, random_factor, position, counter_strat):
        """Exploitative preflop strategy."""
        facing_bet = obs["opp_bet"] > obs["my_bet"]
        valid_actions = obs["valid_actions"]
        
        # Apply counter strategy adjustments
        bet_threshold = counter_strat['bet_threshold'] - 0.1  # Lower threshold preflop
        bluff_frequency = counter_strat['bluff_frequency']
        sizing_multiplier = counter_strat['sizing_multiplier']
        
        if position == "SB":
            # Small blind strategy
            if not facing_bet:
                # Open decision
                if hand_strength > bet_threshold or random_factor < bluff_frequency:
                    if valid_actions[action_types.RAISE.value]:
                        return self._make_raise(obs, self.sizings['preflop'] * sizing_multiplier)
                else:
                    # Fold weak hands
                    return (action_types.FOLD.value, 0, -1)
            else:
                # Facing a 3-bet
                # Adjust strategy based on opponent aggression
                if hand_strength > bet_threshold + 0.2 or (hand_strength > bet_threshold and random_factor < 0.3):
                    if random_factor < 0.5 and valid_actions[action_types.RAISE.value]:
                        return self._make_raise(obs, 3.0 * sizing_multiplier)
                    elif valid_actions[action_types.CALL.value]:
                        return (action_types.CALL.value, 0, -1)
                return (action_types.FOLD.value, 0, -1)
        else:  # Big blind
            if not facing_bet:
                # Check when no raise
                if valid_actions[action_types.CHECK.value]:
                    return (action_types.CHECK.value, 0, -1)
            else:
                # Facing a raise
                if hand_strength > bet_threshold or random_factor < bluff_frequency:
                    # 3-bet or call based on hand strength
                    if hand_strength > bet_threshold + 0.2 or random_factor < 0.3:
                        if valid_actions[action_types.RAISE.value]:
                            return self._make_raise(obs, 3.0 * sizing_multiplier)
                    if valid_actions[action_types.CALL.value]:
                        return (action_types.CALL.value, 0, -1)
                return (action_types.FOLD.value, 0, -1)
        
        # Default action
        if valid_actions[action_types.CHECK.value]:
            return (action_types.CHECK.value, 0, -1)
        else:
            return (action_types.FOLD.value, 0, -1)
    
    def _postflop_exploitative(self, obs, hand_strength, random_factor, position, street, counter_strat):
        """Exploitative postflop strategy."""
        facing_bet = obs["opp_bet"] > obs["my_bet"]
        valid_actions = obs["valid_actions"]
        
        # Apply counter strategy adjustments
        bet_threshold = counter_strat['bet_threshold']
        bluff_frequency = counter_strat['bluff_frequency']
        sizing_multiplier = counter_strat['sizing_multiplier']
        
        # Adjust bet threshold based on street (more conservative on later streets)
        if street == 2:  # Turn
            bet_threshold = min(0.9, bet_threshold + 0.05)
        elif street == 3:  # River
            bet_threshold = min(0.9, bet_threshold + 0.1)
            # Polarize river strategy
            if hand_strength > 0.8:
                bet_threshold = 0.0  # Always bet very strong hands
            elif hand_strength < 0.4:
                # Increase bluffing with very weak hands if opponent folds a lot
                bluff_frequency = min(0.8, bluff_frequency * 1.5)
        
        # Get street-specific sizing
        if street == 1:
            sizing = self.sizings['flop'] * sizing_multiplier
        elif street == 2:
            sizing = self.sizings['turn'] * sizing_multiplier
        else:
            sizing = self.sizings['river'] * sizing_multiplier
        
        # When facing a bet
        if facing_bet:
            # Calculate pot odds
            pot_size = obs["my_bet"] + obs["opp_bet"]
            call_amount = obs["opp_bet"] - obs["my_bet"]
            pot_odds = call_amount / (pot_size + call_amount)
            
            # Adjust calling/raising threshold based on pot odds
            calling_threshold = max(0.3, pot_odds - 0.1)
            raising_threshold = bet_threshold + 0.1
            
            # Decide whether to call/raise/fold
            if hand_strength > raising_threshold or (hand_strength > 0.5 and random_factor < bluff_frequency * 0.5):
                if valid_actions[action_types.RAISE.value]:
                    # Size based on hand strength
                    if hand_strength > 0.8:
                        return self._make_raise(obs, sizing * 1.5)  # Big raise with strong hands
                    else:
                        return self._make_raise(obs, sizing)
            
            if hand_strength > calling_threshold:
                if valid_actions[action_types.CALL.value]:
                    return (action_types.CALL.value, 0, -1)
            
            return (action_types.FOLD.value, 0, -1)
        
        # When we can bet/check
        else:
            # Decide whether to bet or check
            if hand_strength > bet_threshold or (hand_strength < 0.3 and random_factor < bluff_frequency):
                if valid_actions[action_types.RAISE.value]:
                    # Size based on hand strength
                    if hand_strength > 0.8:
                        return self._make_raise(obs, sizing * 1.2)  # Bigger with strong hands
                    elif hand_strength < 0.3:
                        # Slightly smaller bluff sizing
                        return self._make_raise(obs, sizing * 0.9)
                    else:
                        return self._make_raise(obs, sizing)
            
            if valid_actions[action_types.CHECK.value]:
                return (action_types.CHECK.value, 0, -1)
            else:
                return (action_types.FOLD.value, 0, -1)

class SuperAggressiveStrategy(BaseStrategy):
    """Extremely aggressive strategy targeting weak players who fold too much."""
    
    def __init__(self, hand_evaluator):
        super().__init__(hand_evaluator)
        
        # Extremely aggressive parameters
        self.bet_frequency = 0.95    # Bet/raise almost every hand
        self.large_bet_threshold = 0.4  # Use large bets with more hands
        self.bluff_frequency = 0.8   # Bluff very frequently
        
        # Extra large bet sizings
        self.sizings = {
            'preflop': 1.5,  # Raise to 6bb preflop
            'flop': 1.5,     # 150% pot on flop
            'turn': 2.0,     # 200% pot on turn
            'river': 2.5     # 250% pot on river
        }
        
        # Hand range adjustments
        self.min_playable_strength = 0.1  # Play almost any hand
    
    def get_action(self, obs, opponent_tendencies, time_remaining=0.5):
        """Get action using super aggressive strategy."""
        my_cards = obs["my_cards"]
        community_cards = obs.get("community_cards", [])
        street = obs["street"]
        position = "SB" if obs["acting_agent"] == 0 else "BB"
        facing_bet = obs["opp_bet"] > obs["my_bet"]
        valid_actions = obs["valid_actions"]
        
        # Get hand strength and randomization factor
        hand_strength = self.hand_evaluator.evaluate(my_cards, community_cards)
        random_factor = random.random()
        
        # Adjust strategy based on opponent's fold frequency
        fold_frequency = opponent_tendencies.get('fold_frequency', 0.5)
        
        # If opponent folds a lot, increase our aggression even more
        adjusted_bluff_freq = self.bluff_frequency
        if fold_frequency > 0.5:
            adjusted_bluff_freq = min(0.95, self.bluff_frequency + (fold_frequency - 0.5) * 0.8)
        
        # Preflop strategy
        if street == 0:
            return self._preflop_super_aggressive(obs, hand_strength, random_factor, position, adjusted_bluff_freq)
        
        # Postflop strategy
        return self._postflop_super_aggressive(obs, hand_strength, random_factor, position, street, adjusted_bluff_freq)
    
    def _preflop_super_aggressive(self, obs, hand_strength, random_factor, position, bluff_freq):
        """Super aggressive preflop strategy."""
        facing_bet = obs["opp_bet"] > obs["my_bet"]
        valid_actions = obs["valid_actions"]
        
        if position == "SB":
            # Small blind strategy - raise almost everything
            if not facing_bet:
                # Open raising range - raise almost everything
                if hand_strength > self.min_playable_strength or random_factor < bluff_freq:
                    if valid_actions[action_types.RAISE.value]:
                        return self._make_raise(obs, self.sizings['preflop'])
                # Fold only absolute garbage
                return (action_types.FOLD.value, 0, -1)
            else:
                # Facing a 3-bet
                if hand_strength > 0.3 or (hand_strength > self.min_playable_strength and random_factor < bluff_freq):
                    # 4-bet or call with wide range
                    if hand_strength > 0.5 or random_factor < 0.5:
                        if valid_actions[action_types.RAISE.value]:
                            return self._make_raise(obs, 3.0)
                    if valid_actions[action_types.CALL.value]:
                        return (action_types.CALL.value, 0, -1)
                return (action_types.FOLD.value, 0, -1)
        else:  # Big blind
            if not facing_bet:
                # Check or raise when no bet
                if hand_strength > 0.3 or random_factor < 0.5:
                    if valid_actions[action_types.RAISE.value]:
                        return self._make_raise(obs, self.sizings['preflop'])
                if valid_actions[action_types.CHECK.value]:
                    return (action_types.CHECK.value, 0, -1)
            else:
                # Facing a raise - 3-bet very frequently
                if hand_strength > 0.2 or random_factor < bluff_freq:
                    if hand_strength > 0.4 or random_factor < 0.6:
                        if valid_actions[action_types.RAISE.value]:
                            return self._make_raise(obs, 4.0)
                    if valid_actions[action_types.CALL.value]:
                        return (action_types.CALL.value, 0, -1)
                return (action_types.FOLD.value, 0, -1)
        
        # Default
        if valid_actions[action_types.CHECK.value]:
            return (action_types.CHECK.value, 0, -1)
        else:
            return (action_types.FOLD.value, 0, -1)
    
    def _postflop_super_aggressive(self, obs, hand_strength, random_factor, position, street, bluff_freq):
        """Super aggressive postflop strategy."""
        facing_bet = obs["opp_bet"] > obs["my_bet"]
        valid_actions = obs["valid_actions"]
        
        # Sizing based on street - extremely aggressive
        if street == 1:
            sizing = self.sizings['flop']
        elif street == 2:
            sizing = self.sizings['turn']
        else:
            sizing = self.sizings['river']
        
        # When facing a bet
        if facing_bet:
            # Call or raise with very wide range
            if hand_strength > 0.3 or (hand_strength > 0.1 and random_factor < bluff_freq):
                if hand_strength > 0.5 or random_factor < 0.5:
                    if valid_actions[action_types.RAISE.value]:
                        # Extremely large raises
                        return self._make_raise(obs, sizing * 1.5)
                if valid_actions[action_types.CALL.value]:
                    return (action_types.CALL.value, 0, -1)
            return (action_types.FOLD.value, 0, -1)
        
        # When first to act
        else:
            # Bet almost every hand
            if hand_strength > 0.1 or random_factor < bluff_freq:
                if valid_actions[action_types.RAISE.value]:
                    # Use huge sizing with strong hands
                    if hand_strength > 0.6:
                        return self._make_raise(obs, sizing * 1.5)
                    return self._make_raise(obs, sizing)
            
            if valid_actions[action_types.CHECK.value]:
                return (action_types.CHECK.value, 0, -1)
            else:
                return (action_types.FOLD.value, 0, -1)