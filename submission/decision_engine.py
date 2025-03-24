
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
        
        # Enhanced pre-flop strategy with GTO influence
        self.preflop_strategy = self._init_preflop_strategy()
        
        # Optimal bet sizing strategy with GTO ratios
        self.bet_sizing = {
            'small': 0.5,    # Half pot (for thin value and balancing ranges)
            'medium': 0.7,   # 70% pot (standard value bet)
            'large': 1.0,    # Full pot (strong value or semi-bluff)
            'overbet': 1.5   # 1.5x pot (polarized range)
        }
        
        # Strategic adjustments based on position
        self.position_adjustments = {
            'SB': {
                'raise_freq': 1.2,     # More raising in position
                'call_freq': 0.9,      # Less calling in position
                'bluff_freq': 1.3,     # More bluffing in position
                'value_size': 1.1      # Larger value bets in position
            },
            'BB': {
                'raise_freq': 0.8,     # Less raising out of position
                'call_freq': 1.1,      # More calling out of position
                'bluff_freq': 0.7,     # Less bluffing out of position
                'value_size': 0.9      # Smaller value bets out of position
            }
        }
        
        # Game theory balanced ranges by street
        self.balanced_ranges = {
            0: {'value': 0.45, 'bluff': 0.25, 'check': 0.30},  # Preflop
            1: {'value': 0.40, 'bluff': 0.30, 'check': 0.30},  # Flop
            2: {'value': 0.50, 'bluff': 0.20, 'check': 0.30},  # Turn
            3: {'value': 0.60, 'bluff': 0.10, 'check': 0.30}   # River
        }
        
        # Adaptation rate - how quickly we deviate from GTO toward exploitation
        self.adaptation_rate = 0.3
        
        # Track our own actions for balanced play
        self.actions_by_street = {
            0: {'value': 0, 'bluff': 0, 'check': 0},  # Preflop
            1: {'value': 0, 'bluff': 0, 'check': 0},  # Flop
            2: {'value': 0, 'bluff': 0, 'check': 0},  # Turn
            3: {'value': 0, 'bluff': 0, 'check': 0}   # River
        }
    
    def _init_preflop_strategy(self):
        """Initialize enhanced pre-flop strategy chart with GTO concepts."""
        strategy = {
            'SB': {  # Small Blind
                5: {'action': action_types.RAISE, 'sizing': 'large'},   # Premium hands (AA, 99, A9s)
                4: {'action': action_types.RAISE, 'sizing': 'medium'},  # Strong hands (88, A8s, etc)
                3: {'action': action_types.RAISE, 'sizing': 'small'},   # Above average (77, A7s, etc)
                2: {'action': action_types.CALL, 'sizing': 'small'},    # Playable hands (A6s, 65s, etc)
                1: {'action': action_types.FOLD, 'sizing': 'small'}     # Weak hands (rest)
            },
            'BB': {  # Big Blind
                5: {'action': action_types.RAISE, 'sizing': 'large'},   # Premium hands
                4: {'action': action_types.RAISE, 'sizing': 'medium'},  # Strong hands
                3: {'action': action_types.CALL, 'sizing': 'small'},    # Defend vs raise
                2: {'action': action_types.CHECK, 'sizing': 'small'},   # Check with marginal
                1: {'action': action_types.CHECK, 'sizing': 'small'}    # Check with weak
            }
        }
        return strategy
    
    def make_decision(self, obs):
        """
        Advanced decision-making function that processes the observation
        and returns the optimal action using GTO principles and selective exploitation.
        """
        # Start timing the decision
        self.time_manager.start_decision()
        
        # Extract observation data
        my_cards = obs["my_cards"]
        community_cards = [c for c in obs["community_cards"] if c != -1]
        street = obs["street"]
        acting_agent = obs["acting_agent"]
        position = "SB" if acting_agent == 0 else "BB"
        valid_actions = obs["valid_actions"]
        
        # First, check if we can/should use the redraw option
        if valid_actions[action_types.DISCARD.value]:
            should_redraw, card_idx = self.redraw_strategy.strategic_redraw(
                my_cards, community_cards, street, position, self.opponent_model)
            
            if should_redraw:
                # Add protection - never discard an Ace unless we have two Aces
                if card_idx != -1:
                    card_to_discard = my_cards[card_idx]
                    card_rank = card_to_discard // 3
                    
                    other_card = my_cards[1 - card_idx]
                    other_rank = other_card // 3
                    
                    # Don't discard an Ace (rank 8) unless we have two Aces
                    if card_rank == 8 and other_rank != 8:
                        # Decide whether to skip discard
                        self.time_manager.end_decision()
                        return (action_types.CHECK.value, 0, -1)
                
                self.time_manager.end_decision()
                return (action_types.DISCARD.value, 0, card_idx)
        
        if street == 0:  # Pre-flop
            action = self._enhanced_preflop_decision(obs, position)
        else:  # Post-flop
            action = self._enhanced_postflop_decision(obs, position)
        
        # Track our action for balance tracking
        if action[0] == action_types.RAISE.value:
            hand_strength = self.hand_evaluator.get_hand_strength(my_cards, community_cards)
            
            # Determine if this is a value bet or bluff
            if hand_strength > 0.6:  # Strong hand
                self.actions_by_street[street]['value'] += 1
            elif hand_strength < 0.4:  # Weak hand
                self.actions_by_street[street]['bluff'] += 1
            else:  # Medium strength
                # Could be thin value or semi-bluff
                if street >= 2:  # Turn or River
                    self.actions_by_street[street]['value'] += 1
                else:
                    self.actions_by_street[street]['bluff'] += 1
        else:
            self.actions_by_street[street]['check'] += 1
        
        self.time_manager.end_decision()
        return action
    
    def _enhanced_preflop_decision(self, obs, position):
        """
        Enhanced pre-flop decision making with GTO+exploitative approach
        """
        my_cards = obs["my_cards"]
        my_bet = obs["my_bet"]
        opp_bet = obs["opp_bet"]
        min_raise = obs.get("min_raise", 1)
        max_raise = obs.get("max_raise", 100)
        valid_actions = obs["valid_actions"]
        
        # Get hand strength and rank cards
        hand_strength = self.hand_evaluator.get_hand_strength(my_cards)
        
        # Convert to tier (1-5)
        tier = min(5, max(1, int(hand_strength * 5) + 1))
        
        # Get base strategy from chart
        base_strategy = self.preflop_strategy[position][tier]
        
        # Apply advanced strategic adjustments
        adjusted_strategy = self._advanced_strategy_adjustment(base_strategy, obs, hand_strength, position)
        
        # Execute adjusted strategy
        action_type = adjusted_strategy['action']
        
        if action_type == action_types.FOLD:
            if valid_actions[action_types.FOLD.value]:
                return (action_types.FOLD.value, 0, -1)
            # If fold isn't valid, check instead
            return (action_types.CHECK.value, 0, -1)
            
        elif action_type == action_types.CHECK:
            if valid_actions[action_types.CHECK.value]:
                return (action_types.CHECK.value, 0, -1)
            # If check isn't valid, call instead
            if valid_actions[action_types.CALL.value]:
                return (action_types.CALL.value, 0, -1)
            return (action_types.FOLD.value, 0, -1)
            
        elif action_type == action_types.CALL:
            if valid_actions[action_types.CALL.value]:
                return (action_types.CALL.value, 0, -1)
            # If call isn't valid, check instead
            if valid_actions[action_types.CHECK.value]:
                return (action_types.CHECK.value, 0, -1)
            return (action_types.FOLD.value, 0, -1)
            
        elif action_type == action_types.RAISE:
            if valid_actions[action_types.RAISE.value]:
                # Calculate optimal bet size
                pot_size = my_bet + opp_bet
                bet_to_call = opp_bet - my_bet
                
                if adjusted_strategy['sizing'] == 'small':
                    target_size = max(min_raise, int(pot_size * self.bet_sizing['small']))
                elif adjusted_strategy['sizing'] == 'medium':
                    target_size = max(min_raise, int(pot_size * self.bet_sizing['medium']))
                elif adjusted_strategy['sizing'] == 'large':
                    target_size = max(min_raise, int(pot_size * self.bet_sizing['large']))
                else:  # overbet
                    target_size = max(min_raise, int(pot_size * self.bet_sizing['overbet']))
                
                # Apply position-based size adjustment
                pos_factor = self.position_adjustments[position]['value_size']
                target_size = int(target_size * pos_factor)
                
                # Ensure within limits
                bet_size = min(max_raise, max(min_raise, target_size))
                return (action_types.RAISE.value, bet_size, -1)
            
            # If can't raise, call instead
            if valid_actions[action_types.CALL.value]:
                return (action_types.CALL.value, 0, -1)
            # If can't call either, check or fold
            if valid_actions[action_types.CHECK.value]:
                return (action_types.CHECK.value, 0, -1)
            return (action_types.FOLD.value, 0, -1)
        
        # Default action - check if possible, otherwise fold
        if valid_actions[action_types.CHECK.value]:
            return (action_types.CHECK.value, 0, -1)
        return (action_types.FOLD.value, 0, -1)
    
    def _enhanced_postflop_decision(self, obs, position):
        """
        Enhanced post-flop decision making with GTO principles
        and selective exploitation based on opponent modeling
        """
        my_cards = obs["my_cards"]
        community_cards = [c for c in obs["community_cards"] if c != -1]
        street = obs["street"]
        my_bet = obs["my_bet"]
        opp_bet = obs["opp_bet"]
        min_raise = obs.get("min_raise", 1)
        max_raise = obs.get("max_raise", 100)
        valid_actions = obs["valid_actions"]
        
        # Calculate current pot size
        pot_size = my_bet + opp_bet
        
        # Calculate hand strength with opponent range consideration
        hand_strength = self.hand_evaluator.get_hand_strength(my_cards, community_cards)
        
        # Adjust hand strength based on opponent modeling
        adjusted_strength = self.opponent_model.adjust_hand_strength(hand_strength, street)
        
        # Calculate pot odds if we need to call
        call_amount = opp_bet - my_bet
        if call_amount > 0:
            pot_odds = call_amount / (pot_size + call_amount)
        else:
            pot_odds = 0
        
        # Determine whether this should be a bluff spot
        # Combine GTO bluffing frequency with exploitative adjustments
        gto_bluff_freq = self.balanced_ranges[street]['bluff']
        exploit_adj = 0
        
        # Exploitation adjustment based on opponent tendencies
        if self.opponent_model.get_fold_equity() > 0.7:
            exploit_adj = 0.15  # Opponent folds too much - bluff more
        elif self.opponent_model.get_fold_equity() < 0.3:
            exploit_adj = -0.1  # Opponent calls too much - bluff less
        
        # Apply positional adjustment
        pos_bluff_factor = self.position_adjustments[position]['bluff_freq']
        bluff_frequency = (gto_bluff_freq + exploit_adj) * pos_bluff_factor
        
        # Should we bluff?
        is_bluff_spot = (
            hand_strength < 0.4 and 
            random.random() < bluff_frequency and
            self._is_balanced_to_bluff(street)
        )
        
        # Decision tree based on scenario
        if call_amount == 0:  # We can check
            # Strong hand - value bet
            if adjusted_strength > 0.65:
                if valid_actions[action_types.RAISE.value]:
                    # Size based on street and strength
                    size_factor = self.bet_sizing['medium'] if street == 1 else self.bet_sizing['large']
                    bet_size = max(min_raise, int(pot_size * size_factor))
                    
                    # Apply position-based size adjustment
                    pos_factor = self.position_adjustments[position]['value_size']
                    bet_size = int(bet_size * pos_factor)
                    
                    # Ensure within limits
                    bet_size = min(max_raise, bet_size)
                    return (action_types.RAISE.value, bet_size, -1)
                else:
                    return (action_types.CHECK.value, 0, -1)
            
            # Medium hand - thin value bet or check
            elif adjusted_strength > 0.5:
                # On later streets, more likely to bet for value
                if street >= 2 and valid_actions[action_types.RAISE.value]:
                    value_freq = self.balanced_ranges[street]['value']
                    pos_value_factor = self.position_adjustments[position]['value_size']
                    
                    if random.random() < value_freq * pos_value_factor:
                        bet_size = max(min_raise, int(pot_size * self.bet_sizing['small']))
                        bet_size = min(max_raise, bet_size)
                        return (action_types.RAISE.value, bet_size, -1)
                
                return (action_types.CHECK.value, 0, -1)
            
            # Weak hand - check or bluff
            else:
                if is_bluff_spot and valid_actions[action_types.RAISE.value]:
                    # Bluff sizing - smaller on flop, larger on turn/river
                    bluff_size = max(min_raise, int(pot_size * self.bet_sizing['small']))
                    if street >= 2:
                        bluff_size = max(min_raise, int(pot_size * self.bet_sizing['medium']))
                    
                    bluff_size = min(max_raise, bluff_size)
                    return (action_types.RAISE.value, bluff_size, -1)
                else:
                    return (action_types.CHECK.value, 0, -1)
        
        else:  # Facing a bet
            # Strong hand - raise for value
            if adjusted_strength > 0.75 and valid_actions[action_types.RAISE.value]:
                # Size raise based on opponent aggression and pot
                aggression = self.opponent_model.get_aggression_factor()
                
                # Against aggressive opponents, raise larger
                raise_factor = 2.5
                if aggression > 0.7:
                    raise_factor = 3.0
                elif aggression < 0.3:
                    raise_factor = 2.0
                
                raise_size = max(min_raise, min(max_raise, int(call_amount * raise_factor)))
                return (action_types.RAISE.value, raise_size, -1)
            
            # Good hand - call or raise
            elif adjusted_strength > 0.6:
                # Sometimes raise as a semi-bluff or for value
                raise_threshold = self.balanced_ranges[street]['value'] * self.position_adjustments[position]['raise_freq']
                
                if random.random() < raise_threshold and valid_actions[action_types.RAISE.value]:
                    raise_size = max(min_raise, min(max_raise, int(call_amount * 2.5)))
                    return (action_types.RAISE.value, raise_size, -1)
                elif valid_actions[action_types.CALL.value]:
                    return (action_types.CALL.value, 0, -1)
                else:
                    return (action_types.FOLD.value, 0, -1)
            
            # Medium hand - call if odds are good
            elif adjusted_strength > pot_odds + 0.1:
                if valid_actions[action_types.CALL.value]:
                    return (action_types.CALL.value, 0, -1)
                else:
                    return (action_types.FOLD.value, 0, -1)
            
            # Weak hand - consider bluff-raising or folding
            elif is_bluff_spot and valid_actions[action_types.RAISE.value]:
                raise_size = max(min_raise, min(max_raise, int(call_amount * 2.5)))
                return (action_types.RAISE.value, raise_size, -1)
            else:
                return (action_types.FOLD.value, 0, -1)
        
        # Default action - check if we can, fold if we must call
        if call_amount > 0:
            return (action_types.FOLD.value, 0, -1)
        else:
            return (action_types.CHECK.value, 0, -1)
    
    def _advanced_strategy_adjustment(self, base_strategy, obs, hand_strength, position):
        """
        Apply advanced strategic adjustments based on game theory and opponent modeling
        """
        adjusted_strategy = base_strategy.copy()
        
        acting_agent = obs["acting_agent"]
        my_bet = obs["my_bet"]
        opp_bet = obs["opp_bet"]
        
        # Extract opponent tendencies with confidence weighting
        aggression = self.opponent_model.get_aggression_factor()
        fold_equity = self.opponent_model.get_fold_equity()
        bluff_frequency = self.opponent_model.get_bluff_frequency()
        
        # Adjust for opponent's aggression level
        if opp_bet > my_bet:  # Facing a raise
            # Against aggressive opponents, tighten up with marginal hands
            if aggression > 0.7 and adjusted_strategy['action'] != action_types.FOLD:
                if hand_strength < 0.5:
                    # Downgrade action one level
                    if adjusted_strategy['action'] == action_types.RAISE:
                        adjusted_strategy['action'] = action_types.CALL
                    elif adjusted_strategy['action'] == action_types.CALL:
                        adjusted_strategy['action'] = action_types.FOLD
            
            # Against passive opponents, be more aggressive
            elif aggression < 0.3:
                if hand_strength > 0.4 and adjusted_strategy['action'] == action_types.FOLD:
                    adjusted_strategy['action'] = action_types.CALL
                elif hand_strength > 0.6 and adjusted_strategy['action'] == action_types.CALL:
                    adjusted_strategy['action'] = action_types.RAISE
                    adjusted_strategy['sizing'] = 'medium'
        
        # Adjust for opponent's folding tendencies
        elif my_bet == opp_bet:  # Equal bets, we have initiative
            # If opponent folds too much, bluff more
            if fold_equity > 0.7:
                if random.random() < 0.3 and hand_strength > 0.3:
                    # Upgrade action to a raise
                    adjusted_strategy['action'] = action_types.RAISE
                    adjusted_strategy['sizing'] = 'small'
            
            # If opponent calls too much, value bet more thinly
            elif fold_equity < 0.3:
                if hand_strength > 0.5 and adjusted_strategy['action'] != action_types.RAISE:
                    adjusted_strategy['action'] = action_types.RAISE
                    adjusted_strategy['sizing'] = 'small'
        
        # Balance check - ensure we're not being too predictable
        if self._need_to_balance_strategy():
            # Occasionally deviate from standard play for balance
            if random.random() < 0.15:
                if adjusted_strategy['action'] == action_types.FOLD and hand_strength > 0.2:
                    # Occasionally call with weak hands
                    adjusted_strategy['action'] = action_types.CALL
                elif adjusted_strategy['action'] == action_types.CALL and hand_strength > 0.55:
                    # Occasionally raise with medium-strong hands
                    adjusted_strategy['action'] = action_types.RAISE
                    adjusted_strategy['sizing'] = 'medium'
                elif adjusted_strategy['action'] == action_types.RAISE and hand_strength < 0.7:
                    # Occasionally call instead of raise with good but not great hands
                    adjusted_strategy['action'] = action_types.CALL
        
        return adjusted_strategy
    
    def _need_to_balance_strategy(self):
        """
        Check if we need to balance our strategy based on recent actions
        """
        # If we haven't played enough hands, no need to balance yet
        if sum(sum(act.values()) for act in self.actions_by_street.values()) < 20:
            return False
        
        # Check if we're being too predictable with certain actions
        for street, actions in self.actions_by_street.items():
            total_actions = sum(actions.values())
            if total_actions > 5:
                # Calculate actual frequencies
                value_freq = actions['value'] / total_actions
                bluff_freq = actions['bluff'] / total_actions
                check_freq = actions['check'] / total_actions
                
                # Compare to balanced GTO frequencies
                gto_value = self.balanced_ranges[street]['value']
                gto_bluff = self.balanced_ranges[street]['bluff']
                gto_check = self.balanced_ranges[street]['check']
                
                # Check for imbalances
                if abs(value_freq - gto_value) > 0.15 or abs(bluff_freq - gto_bluff) > 0.15:
                    return True
        
        return False
    
    def _is_balanced_to_bluff(self, street):
        """
        Check if our value-to-bluff ratio is balanced enough to make another bluff
        """
        actions = self.actions_by_street[street]
        total_bets = actions['value'] + actions['bluff']
        
        if total_bets < 5:
            # Not enough data to worry about balance yet
            return True
            
        # Calculate current bluff frequency
        current_bluff_freq = actions['bluff'] / total_bets
        
        # Get GTO-optimal bluff frequency for this street
        gto_bluff_freq = self.balanced_ranges[street]['bluff'] / (
            self.balanced_ranges[street]['value'] + self.balanced_ranges[street]['bluff']
        )
        
        # If we're bluffing too much already, don't bluff again
        if current_bluff_freq > gto_bluff_freq + 0.1:
            return False
            
        # We're balanced or not bluffing enough, so bluffing is fine
        return True

# class DecisionEngine:
    
#     def __init__(self, hand_evaluator, redraw_strategy, opponent_model, time_manager):
#         self.hand_evaluator = hand_evaluator
#         self.redraw_strategy = redraw_strategy
#         self.opponent_model = opponent_model
#         self.time_manager = time_manager
        
#         self.preflop_strategy = self._init_preflop_strategy()
        
#         # Bet sizing strategy with game theory optimal ratios
#         self.bet_sizing = {
#             'small': 0.5,    # Half pot (for thin value and balancing ranges)
#             'medium': 0.75,  # 3/4 pot (standard value bet)
#             'large': 1.0,    # Full pot (strong value or semi-bluff)
#             'overbet': 1.5   # 1.5x pot (polarized range)
#         }
        
#         # Efficient action abstraction
#         self.action_abstraction = {
#             'preflop': ['fold', 'call', 'raise_min', 'raise_med', 'raise_max'],
#             'postflop': ['fold', 'check/call', 'bet_small', 'bet_med', 'bet_max']
#         }
        
#         # Strategy tables (populated by CFR approximation)
#         self.strategy_table = {}
        
#         # Tracking variables for regret minimization
#         self.regret_sum = defaultdict(lambda: defaultdict(float))
#         self.strategy_sum = defaultdict(lambda: defaultdict(float))
        
#         # Bluffing strategy based on game theory
#         self.bluff_to_value_ratio = 0.7  # Approximately optimal bluff-to-value ratio
        
#         # Learning rate for strategy adaptation
#         self.learning_rate = 0.1
        
#         # Information abstraction for efficient computation
#         self.info_abstraction = {}
        
#         self.cached_strategies = {}
    
#     def _init_preflop_strategy(self):
#         """Initialize pre-flop strategy chart."""
#         strategy = {
#             'SB': {  # Small Blind
#                 5: {'action': action_types.RAISE, 'sizing': 'large'},  # Premium hands
#                 4: {'action': action_types.RAISE, 'sizing': 'medium'},  # Strong hands
#                 3: {'action': action_types.RAISE, 'sizing': 'small'},   # Playable hands
#                 2: {'action': action_types.CALL, 'sizing': 'small'},    # Marginal hands
#                 1: {'action': action_types.FOLD, 'sizing': 'small'}     # Weak hands
#             },
#             'BB': {  # Big Blind
#                 5: {'action': action_types.RAISE, 'sizing': 'large'},   # Premium hands
#                 4: {'action': action_types.RAISE, 'sizing': 'medium'},  # Strong hands
#                 3: {'action': action_types.CALL, 'sizing': 'small'},    # Playable hands
#                 2: {'action': action_types.CHECK, 'sizing': 'small'},   # Marginal hands
#                 1: {'action': action_types.CHECK, 'sizing': 'small'}    # Weak hands
#             }
#         }
#         return strategy
    
#     def make_decision(self, obs):
#         """
#         Main decision-making function that processes the observation
#         and returns the optimal action.
        
#         Args:
#             obs: Observation dictionary from the environment
            
#         Returns:
#             Tuple of (action_type, raise_amount, card_to_discard)
#         """
#         # Start timing the decision
#         self.time_manager.start_decision()
        
#         # Extract observation data
#         my_cards = obs["my_cards"]
#         community_cards = obs.get("community_cards", [])
#         street = obs["street"]
#         acting_agent = obs["acting_agent"]
#         position = "SB" if acting_agent == 0 else "BB"
#         valid_actions = obs["valid_actions"]
        
#         # First, check if we can/should use the redraw option
#         if valid_actions[action_types.DISCARD.value]:
#             should_redraw, card_idx = self.redraw_strategy.strategic_redraw(
#                 my_cards, community_cards, street, position, self.opponent_model)
            
#             # TODO
#             if should_redraw:
#                 self.time_manager.end_decision()
#                 return (action_types.DISCARD.value, 0, card_idx)
        
#         if street == 0:  # Pre-flop
#             action = self._preflop_decision(obs)
#         else:  # Post-flop
#             action = self._postflop_decision(obs)
        
#         self.time_manager.end_decision()
#         return action
    
#     def _preflop_decision(self, obs):
#         my_cards = obs["my_cards"]
#         acting_agent = obs["acting_agent"]
#         position = "SB" if acting_agent == 0 else "BB"
#         my_bet = obs["my_bet"]
#         opp_bet = obs["opp_bet"]
#         min_raise = obs.get("min_raise", 1)
#         max_raise = obs.get("max_raise", 100)
#         valid_actions = obs["valid_actions"]
        
#         hand_strength = self.hand_evaluator.get_hand_strength(my_cards)
        
#         # Convert to tier (1-5)
#         tier = min(5, max(1, int(hand_strength * 5) + 1))
        
#         base_strategy = self.preflop_strategy[position][tier]
        
#         adjusted_strategy = self._adjust_strategy(base_strategy, obs, hand_strength)
        
#         if adjusted_strategy['action'] == action_types.FOLD:
#             if valid_actions[action_types.FOLD.value]:
#                 return (action_types.FOLD.value, 0, -1)
#             # If fold isn't valid, check instead
#             return (action_types.CHECK.value, 0, -1)
            
#         elif adjusted_strategy['action'] == action_types.CHECK:
#             if valid_actions[action_types.CHECK.value]:
#                 return (action_types.CHECK.value, 0, -1)
#             # If check isn't valid, call instead
#             return (action_types.CALL.value, 0, -1)
            
#         elif adjusted_strategy['action'] == action_types.CALL:
#             if valid_actions[action_types.CALL.value]:
#                 return (action_types.CALL.value, 0, -1)
#             # If call isn't valid, check instead
#             return (action_types.CHECK.value, 0, -1)
            
#         elif adjusted_strategy['action'] == action_types.RAISE:
#             if valid_actions[action_types.RAISE.value]:
#                 # Calculate bet size based on sizing strategy
#                 pot_size = my_bet + opp_bet
#                 bet_to_call = opp_bet - my_bet
                
#                 if adjusted_strategy['sizing'] == 'small':
#                     target_size = max(min_raise, int(pot_size * self.bet_sizing['small']))
#                 elif adjusted_strategy['sizing'] == 'medium':
#                     target_size = max(min_raise, int(pot_size * self.bet_sizing['medium']))
#                 elif adjusted_strategy['sizing'] == 'large':
#                     target_size = max(min_raise, int(pot_size * self.bet_sizing['large']))
#                 else:
#                     target_size = max(min_raise, int(pot_size * self.bet_sizing['overbet']))
                
#                 bet_size = min(max_raise, max(min_raise, target_size))
#                 return (action_types.RAISE.value, bet_size, -1)
            
#             # If can't raise, call instead
#             if valid_actions[action_types.CALL.value]:
#                 return (action_types.CALL.value, 0, -1)
#             # If can't call either, check
#             return (action_types.CHECK.value, 0, -1)
        
#         # Default: check if possible, otherwise fold
#         if valid_actions[action_types.CHECK.value]:
#             return (action_types.CHECK.value, 0, -1)
#         return (action_types.FOLD.value, 0, -1)
    

#     def _postflop_decision(self, obs):
#         my_cards = obs["my_cards"]
#         community_cards = obs["community_cards"]
#         street = obs["street"]
#         acting_agent = obs["acting_agent"]
#         my_bet = obs["my_bet"]
#         opp_bet = obs["opp_bet"]
#         min_raise = obs.get("min_raise", 1)
#         max_raise = obs.get("max_raise", 100)
#         valid_actions = obs["valid_actions"]
        
#         # Calculate current pot size
#         pot_size = my_bet + opp_bet
        
#         # Calculate hand strength
#         hand_strength = self.hand_evaluator.get_hand_strength(my_cards, community_cards)
        
#         # Adjust hand strength based on opponent modeling
#         adjusted_strength = self.opponent_model.adjust_hand_strength(hand_strength, street)
        
#         # Calculate pot odds if we need to call
#         call_amount = opp_bet - my_bet
#         if call_amount > 0:
#             pot_odds = call_amount / (pot_size + call_amount)
#         else:
#             pot_odds = 0
        
#         # Check if this is a good bluffing spot
#         # is_bluff_candidate = hand_strength > self.bluff_threshold
#         # if is_bluff_candidate:
#         #     bluff_equity = self.opponent_model.get_fold_equity()
#         #     # Adjust strength for potential bluff
#         #     bluff_adjusted_strength = self.opponent_model.adjust_hand_strength(
#         #         hand_strength, street, is_bluff_candidate=True)
#         # else:
#         #     bluff_adjusted_strength = adjusted_strength
#         bluff_adjusted_strength = adjusted_strength
        
        
#         if call_amount == 0:  # We can check
#             # Strong hand - value bet
#             if bluff_adjusted_strength > 0.7:
#                 if valid_actions[action_types.RAISE.value]:
#                     # Size based on street and strength
#                     if street == 1:  # Flop
#                         bet_size = max(min_raise, int(pot_size * self.bet_sizing['medium']))
#                     else:  # Turn or River
#                         bet_size = max(min_raise, int(pot_size * self.bet_sizing['large']))
                    
#                     bet_size = min(max_raise, bet_size)
#                     return (action_types.RAISE.value, bet_size, -1)
            
#             # Medium hand - thin value bet or check
#             elif bluff_adjusted_strength > 0.5:
#                 # On later streets, more likely to bet for value
#                 if street >= 2 and random.random() < bluff_adjusted_strength and valid_actions[action_types.RAISE.value]:
#                     bet_size = max(min_raise, int(pot_size * self.bet_sizing['small']))
#                     bet_size = min(max_raise, bet_size)
#                     return (action_types.RAISE.value, bet_size, -1)
#                 else:
#                     return (action_types.CHECK.value, 0, -1)
            
#             # Weak hand - check or bluff
#             else:
#                 # # Consider bluffing
#                 # if is_bluff_candidate and random.random() < self.bluff_frequency and valid_actions[action_types.RAISE.value]:
#                 #     # Smaller bluff on flop, larger on turn/river
#                 #     if street == 1:
#                 #         bet_size = max(min_raise, int(pot_size * self.bet_sizing['small']))
#                 #     else:
#                 #         bet_size = max(min_raise, int(pot_size * self.bet_sizing['medium']))
                    
#                 #     bet_size = min(max_raise, bet_size)
#                 #     return (action_types.RAISE.value, bet_size, -1)
#                 # else:
#                 return (action_types.CHECK.value, 0, -1)
        
#         else:  # Facing a bet
#             # Strong hand - raise for value
#             if adjusted_strength > 0.8 and valid_actions[action_types.RAISE.value]:
#                 raise_size = max(min_raise, min(max_raise, int(call_amount * 3)))
#                 return (action_types.RAISE.value, raise_size, -1)
            
#             # Good hand - call or raise
#             elif adjusted_strength > 0.6:
#                 # Sometimes raise as a semi-bluff or for value
#                 if random.random() < adjusted_strength - 0.5 and valid_actions[action_types.RAISE.value]:
#                     raise_size = max(min_raise, min(max_raise, int(call_amount * 2.5)))
#                     return (action_types.RAISE.value, raise_size, -1)
#                 elif valid_actions[action_types.CALL.value]:
#                     return (action_types.CALL.value, 0, -1)
            
#             # Medium hand - call if odds are good
#             elif adjusted_strength > pot_odds + 0.1:
#                 if valid_actions[action_types.CALL.value]:
#                     return (action_types.CALL.value, 0, -1)
            
#             # Weak hand - consider bluff-raising or folding
#             # elif is_bluff_candidate and random.random() < bluff_equity / 2 and valid_actions[action_types.RAISE.value]:
#             #     raise_size = max(min_raise, min(max_raise, int(call_amount * 2.5)))
#             #     return (action_types.RAISE.value, raise_size, -1)
        
#         # Default: fold if we have to call, check if we can
#         if call_amount > 0:
#             return (action_types.FOLD.value, 0, -1)
#         else:
#             return (action_types.CHECK.value, 0, -1)
    
#     def _adjust_strategy(self, base_strategy, obs, hand_strength):
#         """
#         Adjust pre-flop strategy based on game state and opponent model.
#         """
#         adjusted_strategy = base_strategy.copy()
        
#         acting_agent = obs["acting_agent"]
#         position = "SB" if acting_agent == 0 else "BB"
#         my_bet = obs["my_bet"]
#         opp_bet = obs["opp_bet"]
        
#         aggression = self.opponent_model.get_aggression_factor()
#         fold_equity = self.opponent_model.get_fold_equity()
        
#         # If we're in BB facing a raise
#         if position == "BB" and opp_bet > my_bet:
#             # Against aggressive opponents, tighten up
#             if aggression > 0.7 and adjusted_strategy['action'] != action_types.FOLD:
#                 if hand_strength < 0.6:
#                     adjusted_strategy['action'] = action_types.FOLD
            
#             # Against passive opponents, fight back more
#             elif aggression < 0.3 and adjusted_strategy['action'] == action_types.FOLD:
#                 if hand_strength > 0.3:
#                     adjusted_strategy['action'] = action_types.CALL
        
#         # If we're in SB and have initiative
#         elif position == "SB" and my_bet == opp_bet:
#             # If opponent folds often, bluff more
#             if fold_equity > 0.7 and hand_strength < 0.4:
#                 # Upgrade action one level
#                 if adjusted_strategy['action'] == action_types.FOLD:
#                     adjusted_strategy['action'] = action_types.CALL
#                 elif adjusted_strategy['action'] == action_types.CALL:
#                     adjusted_strategy['action'] = action_types.RAISE
#                     adjusted_strategy['sizing'] = 'small'
        
#         # Randomization to avoid predictability
#         # Occasionally upgrade or downgrade action
#         if random.random() < 0.1:
#             if adjusted_strategy['action'] == action_types.CALL:
#                 if random.random() < 0.5:
#                     adjusted_strategy['action'] = action_types.RAISE
#                     adjusted_strategy['sizing'] = 'small'
#                 else:
#                     adjusted_strategy['action'] = action_types.FOLD
            
#             elif adjusted_strategy['action'] == action_types.RAISE:
#                 # Occasionally change sizing
#                 sizings = ['small', 'medium', 'large', 'overbet']
#                 current_idx = sizings.index(adjusted_strategy['sizing'])
#                 new_idx = max(0, min(3, current_idx + random.choice([-1, 1])))
#                 adjusted_strategy['sizing'] = sizings[new_idx]
        
#         return adjusted_strategy
    