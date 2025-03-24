import random
from enum import Enum
from collections import defaultdict
from gym_env import PokerEnv

action_types = PokerEnv.ActionType

class DecisionEngine:
    
    def __init__(self, hand_evaluator, redraw_strategy, opponent_model, time_manager):
        self.hand_evaluator = hand_evaluator
        self.redraw_strategy = redraw_strategy
        self.opponent_model = opponent_model
        self.time_manager = time_manager
        
        self.preflop_strategy = self._init_preflop_strategy()
        
        # Bet sizing strategy with game theory optimal ratios
        self.bet_sizing = {
            'small': 0.5,    # Half pot (for thin value and balancing ranges)
            'medium': 0.75,  # 3/4 pot (standard value bet)
            'large': 1.0,    # Full pot (strong value or semi-bluff)
            'overbet': 1.5   # 1.5x pot (polarized range)
        }
        
        # Efficient action abstraction
        self.action_abstraction = {
            'preflop': ['fold', 'call', 'raise_min', 'raise_med', 'raise_max'],
            'postflop': ['fold', 'check/call', 'bet_small', 'bet_med', 'bet_max']
        }
        
        # Strategy tables (populated by CFR approximation)
        self.strategy_table = {}
        
        # Tracking variables for regret minimization
        self.regret_sum = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))
        
        # Bluffing strategy based on game theory
        self.bluff_to_value_ratio = 0.7  # Approximately optimal bluff-to-value ratio
        
        # Learning rate for strategy adaptation
        self.learning_rate = 0.1
        
        # Information abstraction for efficient computation
        self.info_abstraction = {}
        
        self.cached_strategies = {}
    
    def _init_preflop_strategy(self):
        """Initialize pre-flop strategy chart."""
        strategy = {
            'SB': {  # Small Blind
                5: {'action': action_types.RAISE, 'sizing': 'large'},  # Premium hands
                4: {'action': action_types.RAISE, 'sizing': 'medium'},  # Strong hands
                3: {'action': action_types.RAISE, 'sizing': 'small'},   # Playable hands
                2: {'action': action_types.CALL, 'sizing': 'small'},    # Marginal hands
                1: {'action': action_types.FOLD, 'sizing': 'small'}     # Weak hands
            },
            'BB': {  # Big Blind
                5: {'action': action_types.RAISE, 'sizing': 'large'},   # Premium hands
                4: {'action': action_types.RAISE, 'sizing': 'medium'},  # Strong hands
                3: {'action': action_types.CALL, 'sizing': 'small'},    # Playable hands
                2: {'action': action_types.CHECK, 'sizing': 'small'},   # Marginal hands
                1: {'action': action_types.CHECK, 'sizing': 'small'}    # Weak hands
            }
        }
        return strategy
    
    def make_decision(self, obs):
        """
        Main decision-making function that processes the observation
        and returns the optimal action.
        
        Args:
            obs: Observation dictionary from the environment
            
        Returns:
            Tuple of (action_type, raise_amount, card_to_discard)
        """
        # Start timing the decision
        self.time_manager.start_decision()
        
        # Extract observation data
        my_cards = obs["my_cards"]
        community_cards = obs.get("community_cards", [])
        street = obs["street"]
        acting_agent = obs["acting_agent"]
        position = "SB" if acting_agent == 0 else "BB"
        valid_actions = obs["valid_actions"]
        
        # First, check if we can/should use the redraw option
        if valid_actions[action_types.DISCARD.value]:
            should_redraw, card_idx = self.redraw_strategy.strategic_redraw(
                my_cards, community_cards, street, position, self.opponent_model)
            
            # TODO
            if should_redraw:
                self.time_manager.end_decision()
                return (action_types.DISCARD.value, 0, card_idx)
        
        if street == 0:  # Pre-flop
            action = self._preflop_decision(obs)
        else:  # Post-flop
            action = self._postflop_decision(obs)
        
        self.time_manager.end_decision()
        return action
    
    def _preflop_decision(self, obs):
        my_cards = obs["my_cards"]
        acting_agent = obs["acting_agent"]
        position = "SB" if acting_agent == 0 else "BB"
        my_bet = obs["my_bet"]
        opp_bet = obs["opp_bet"]
        min_raise = obs.get("min_raise", 1)
        max_raise = obs.get("max_raise", 100)
        valid_actions = obs["valid_actions"]
        
        hand_strength = self.hand_evaluator.get_hand_strength(my_cards)
        
        # Convert to tier (1-5)
        tier = min(5, max(1, int(hand_strength * 5) + 1))
        
        base_strategy = self.preflop_strategy[position][tier]
        
        adjusted_strategy = self._adjust_strategy(base_strategy, obs, hand_strength)
        
        if adjusted_strategy['action'] == action_types.FOLD:
            if valid_actions[action_types.FOLD.value]:
                return (action_types.FOLD.value, 0, -1)
            # If fold isn't valid, check instead
            return (action_types.CHECK.value, 0, -1)
            
        elif adjusted_strategy['action'] == action_types.CHECK:
            if valid_actions[action_types.CHECK.value]:
                return (action_types.CHECK.value, 0, -1)
            # If check isn't valid, call instead
            return (action_types.CALL.value, 0, -1)
            
        elif adjusted_strategy['action'] == action_types.CALL:
            if valid_actions[action_types.CALL.value]:
                return (action_types.CALL.value, 0, -1)
            # If call isn't valid, check instead
            return (action_types.CHECK.value, 0, -1)
            
        elif adjusted_strategy['action'] == action_types.RAISE:
            if valid_actions[action_types.RAISE.value]:
                # Calculate bet size based on sizing strategy
                pot_size = my_bet + opp_bet
                bet_to_call = opp_bet - my_bet
                
                if adjusted_strategy['sizing'] == 'small':
                    target_size = max(min_raise, int(pot_size * self.bet_sizing['small']))
                elif adjusted_strategy['sizing'] == 'medium':
                    target_size = max(min_raise, int(pot_size * self.bet_sizing['medium']))
                elif adjusted_strategy['sizing'] == 'large':
                    target_size = max(min_raise, int(pot_size * self.bet_sizing['large']))
                else:
                    target_size = max(min_raise, int(pot_size * self.bet_sizing['overbet']))
                
                bet_size = min(max_raise, max(min_raise, target_size))
                return (action_types.RAISE.value, bet_size, -1)
            
            # If can't raise, call instead
            if valid_actions[action_types.CALL.value]:
                return (action_types.CALL.value, 0, -1)
            # If can't call either, check
            return (action_types.CHECK.value, 0, -1)
        
        # Default: check if possible, otherwise fold
        if valid_actions[action_types.CHECK.value]:
            return (action_types.CHECK.value, 0, -1)
        return (action_types.FOLD.value, 0, -1)
    

    def _postflop_decision(self, obs):
        my_cards = obs["my_cards"]
        community_cards = obs["community_cards"]
        street = obs["street"]
        acting_agent = obs["acting_agent"]
        my_bet = obs["my_bet"]
        opp_bet = obs["opp_bet"]
        min_raise = obs.get("min_raise", 1)
        max_raise = obs.get("max_raise", 100)
        valid_actions = obs["valid_actions"]
        
        # Calculate current pot size
        pot_size = my_bet + opp_bet
        
        # Calculate hand strength
        hand_strength = self.hand_evaluator.get_hand_strength(my_cards, community_cards)
        
        # Adjust hand strength based on opponent modeling
        adjusted_strength = self.opponent_model.adjust_hand_strength(hand_strength, street)
        
        # Calculate pot odds if we need to call
        call_amount = opp_bet - my_bet
        if call_amount > 0:
            pot_odds = call_amount / (pot_size + call_amount)
        else:
            pot_odds = 0
        
        # Check if this is a good bluffing spot
        # is_bluff_candidate = hand_strength > self.bluff_threshold
        # if is_bluff_candidate:
        #     bluff_equity = self.opponent_model.get_fold_equity()
        #     # Adjust strength for potential bluff
        #     bluff_adjusted_strength = self.opponent_model.adjust_hand_strength(
        #         hand_strength, street, is_bluff_candidate=True)
        # else:
        #     bluff_adjusted_strength = adjusted_strength
        bluff_adjusted_strength = adjusted_strength
        
        
        if call_amount == 0:  # We can check
            # Strong hand - value bet
            if bluff_adjusted_strength > 0.7:
                if valid_actions[action_types.RAISE.value]:
                    # Size based on street and strength
                    if street == 1:  # Flop
                        bet_size = max(min_raise, int(pot_size * self.bet_sizing['medium']))
                    else:  # Turn or River
                        bet_size = max(min_raise, int(pot_size * self.bet_sizing['large']))
                    
                    bet_size = min(max_raise, bet_size)
                    return (action_types.RAISE.value, bet_size, -1)
            
            # Medium hand - thin value bet or check
            elif bluff_adjusted_strength > 0.5:
                # On later streets, more likely to bet for value
                if street >= 2 and random.random() < bluff_adjusted_strength and valid_actions[action_types.RAISE.value]:
                    bet_size = max(min_raise, int(pot_size * self.bet_sizing['small']))
                    bet_size = min(max_raise, bet_size)
                    return (action_types.RAISE.value, bet_size, -1)
                else:
                    return (action_types.CHECK.value, 0, -1)
            
            # Weak hand - check or bluff
            else:
                # # Consider bluffing
                # if is_bluff_candidate and random.random() < self.bluff_frequency and valid_actions[action_types.RAISE.value]:
                #     # Smaller bluff on flop, larger on turn/river
                #     if street == 1:
                #         bet_size = max(min_raise, int(pot_size * self.bet_sizing['small']))
                #     else:
                #         bet_size = max(min_raise, int(pot_size * self.bet_sizing['medium']))
                    
                #     bet_size = min(max_raise, bet_size)
                #     return (action_types.RAISE.value, bet_size, -1)
                # else:
                return (action_types.CHECK.value, 0, -1)
        
        else:  # Facing a bet
            # Strong hand - raise for value
            if adjusted_strength > 0.8 and valid_actions[action_types.RAISE.value]:
                raise_size = max(min_raise, min(max_raise, int(call_amount * 3)))
                return (action_types.RAISE.value, raise_size, -1)
            
            # Good hand - call or raise
            elif adjusted_strength > 0.6:
                # Sometimes raise as a semi-bluff or for value
                if random.random() < adjusted_strength - 0.5 and valid_actions[action_types.RAISE.value]:
                    raise_size = max(min_raise, min(max_raise, int(call_amount * 2.5)))
                    return (action_types.RAISE.value, raise_size, -1)
                elif valid_actions[action_types.CALL.value]:
                    return (action_types.CALL.value, 0, -1)
            
            # Medium hand - call if odds are good
            elif adjusted_strength > pot_odds + 0.1:
                if valid_actions[action_types.CALL.value]:
                    return (action_types.CALL.value, 0, -1)
            
            # Weak hand - consider bluff-raising or folding
            elif is_bluff_candidate and random.random() < bluff_equity / 2 and valid_actions[action_types.RAISE.value]:
                raise_size = max(min_raise, min(max_raise, int(call_amount * 2.5)))
                return (action_types.RAISE.value, raise_size, -1)
        
        # Default: fold if we have to call, check if we can
        if call_amount > 0:
            return (action_types.FOLD.value, 0, -1)
        else:
            return (action_types.CHECK.value, 0, -1)
    
    def _adjust_strategy(self, base_strategy, obs, hand_strength):
        """
        Adjust pre-flop strategy based on game state and opponent model.
        """
        adjusted_strategy = base_strategy.copy()
        
        acting_agent = obs["acting_agent"]
        position = "SB" if acting_agent == 0 else "BB"
        my_bet = obs["my_bet"]
        opp_bet = obs["opp_bet"]
        
        aggression = self.opponent_model.get_aggression_factor()
        fold_equity = self.opponent_model.get_fold_equity()
        
        # If we're in BB facing a raise
        if position == "BB" and opp_bet > my_bet:
            # Against aggressive opponents, tighten up
            if aggression > 0.7 and adjusted_strategy['action'] != action_types.FOLD:
                if hand_strength < 0.6:
                    adjusted_strategy['action'] = action_types.FOLD
            
            # Against passive opponents, fight back more
            elif aggression < 0.3 and adjusted_strategy['action'] == action_types.FOLD:
                if hand_strength > 0.3:
                    adjusted_strategy['action'] = action_types.CALL
        
        # If we're in SB and have initiative
        elif position == "SB" and my_bet == opp_bet:
            # If opponent folds often, bluff more
            if fold_equity > 0.7 and hand_strength < 0.4:
                # Upgrade action one level
                if adjusted_strategy['action'] == action_types.FOLD:
                    adjusted_strategy['action'] = action_types.CALL
                elif adjusted_strategy['action'] == action_types.CALL:
                    adjusted_strategy['action'] = action_types.RAISE
                    adjusted_strategy['sizing'] = 'small'
        
        # Randomization to avoid predictability
        # Occasionally upgrade or downgrade action
        if random.random() < 0.1:
            if adjusted_strategy['action'] == action_types.CALL:
                if random.random() < 0.5:
                    adjusted_strategy['action'] = action_types.RAISE
                    adjusted_strategy['sizing'] = 'small'
                else:
                    adjusted_strategy['action'] = action_types.FOLD
            
            elif adjusted_strategy['action'] == action_types.RAISE:
                # Occasionally change sizing
                sizings = ['small', 'medium', 'large', 'overbet']
                current_idx = sizings.index(adjusted_strategy['sizing'])
                new_idx = max(0, min(3, current_idx + random.choice([-1, 1])))
                adjusted_strategy['sizing'] = sizings[new_idx]
        
        return adjusted_strategy
    