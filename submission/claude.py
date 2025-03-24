# Updated Player Agent Strategy

import random
from gym_env import PokerEnv
from agents.agent import Agent
from submission.hand_evaluator import HandEvaluator
from submission.redraw import Redraw
from submission.opponent_model import OpponentModel
from submission.decision_engine import DecisionEngine
from submission.time_manager import TimeManager

action_types = PokerEnv.ActionType

class PlayerAgent(Agent):
    def __name__(self):
        return "ElitePokerBot"

    def __init__(self, stream: bool = True):
        super().__init__(stream)
        # Initialize core components
        self.hand_number = 0
        self.won_hands = 0
        self.time_manager = TimeManager()
        
        self.hand_evaluator = HandEvaluator()
        self.opponent_model = OpponentModel()
        self.redraw_strategy = Redraw(self.hand_evaluator)
        self.decision_engine = DecisionEngine(
            self.hand_evaluator,
            self.redraw_strategy,
            self.opponent_model,
            self.time_manager
        )
        
        # Enhanced tracking
        self.prev_obs = None
        self.hands_played = 0
        self.hand_history = []
        
        # Positional strategy adjustments
        self.position_adjust_factor = {
            'SB': 1.2,   # Small blind position - more aggressive
            'BB': 0.9    # Big blind position - more selective
        }
        
        # Advanced tactics flags
        self.bluff_frequency = 0.15  # Base bluff frequency
        self.slowplay_threshold = 0.85  # Hand strength threshold for slowplay
        self.check_raise_frequency = 0.2  # Check-raise frequency
        
        # Adaptation settings
        self.adaptation_rate = 0.1  # How quickly we adapt to opponent's play
        self.exploit_factor = 0.0   # Start balanced, adjust based on opponent
        self.strategy_phases = {
            'early': {'hands': 20, 'style': 'tight-aggressive'},
            'middle': {'hands': 100, 'style': 'balanced'},
            'late': {'hands': float('inf'), 'style': 'adaptive'}
        }
        
        # Initialize hand strength thresholds
        self.preflop_raise_threshold = 0.5
        self.preflop_call_threshold = 0.35
        self.postflop_value_threshold = 0.6
        self.postflop_bluff_threshold = 0.3
        
        # Card group classifications
        self.premium_hands = self._initialize_premium_hands()
        
        # Track showdown results for learning
        self.showdowns = []
        
        # Initialize the reduced deception threshold
        self._setup_reduced_deception()
        
    def _setup_reduced_deception(self):
        """
        Adjust the redraw strategy to reduce deceptive plays
        """
        if hasattr(self.redraw_strategy, 'deception_threshold'):
            self.redraw_strategy.deception_threshold = 0.02  # Much lower than default
            
    def _initialize_premium_hands(self):
        """
        Define premium starting hands for this specific deck
        """
        premium_hands = []
        
        # Pairs are strong (only 9 ranks in this variant)
        for rank in range(9):
            for s1 in range(3):
                for s2 in range(s1 + 1, 3):
                    card1 = rank * 3 + s1
                    card2 = rank * 3 + s2
                    premium_hands.append((card1, card2))
        
        # A-x suited hands
        ace_rank = 8
        for rank in [6, 7]:  # 8, 9 (high connectivity)
            for suit in range(3):
                card1 = ace_rank * 3 + suit
                card2 = rank * 3 + suit
                premium_hands.append((card1, card2))
        
        # Connected high cards, suited
        for rank1 in [6, 7, 8]:  # 8, 9, A
            for rank2 in [6, 7]:  # 8, 9
                if rank1 != rank2:
                    for suit in range(3):
                        card1 = rank1 * 3 + suit
                        card2 = rank2 * 3 + suit
                        premium_hands.append((card1, card2))
        
        return premium_hands
    
    def act(self, obs, reward, terminated, truncated, info):
        """
        Main action method - enhanced with professional tactics
        """
        # Track new hands
        if self._is_new_hand(obs):
            self.hands_played += 1
            self._update_strategic_phase()
        
        # Update opponent model if we have a previous observation
        if self.prev_obs is not None:
            self._update_opponent_model(self.prev_obs, obs)
        
        # Time management - emergency strategy if time critical
        if self.time_manager.is_time_critical():
            action = self._emergency_strategy(obs)
        else:
            # Apply positional adjustments
            position = 'SB' if obs["acting_agent"] == 0 else 'BB'
            
            # Enhanced decision with positional factors
            action = self._enhanced_decision(obs, position)
        
        # Store observation for next update
        self.prev_obs = obs.copy()
        
        # Log the action (if significant)
        if action[0] in [action_types.RAISE.value, action_types.FOLD.value] or \
           (action[0] == action_types.DISCARD.value and action[2] != -1):
            self._log_action(obs, action)
        
        return action
    
    def _enhanced_decision(self, obs, position):
        """
        Enhanced decision making with pro-level strategy
        """
        # Check for discard opportunity first
        if obs["valid_actions"][action_types.DISCARD.value]:
            should_discard, card_idx = self._strategic_discard_decision(obs, position)
            if should_discard:
                return (action_types.DISCARD.value, 0, card_idx)
        
        # Street-specific strategies
        if obs["street"] == 0:
            return self._preflop_strategy(obs, position)
        else:
            return self._postflop_strategy(obs, position)
    
    def _strategic_discard_decision(self, obs, position):
        """
        Enhanced discard decision logic with anti-deception protection
        """
        my_cards = obs["my_cards"]
        community_cards = obs["community_cards"]
        street = obs["street"]
        
        # Use redraw strategy but with protection against discarding premium cards
        should_redraw, card_idx = self.redraw_strategy.strategic_redraw(
            my_cards, community_cards, street, position, self.opponent_model)
        
        # Add protection - never discard an Ace unless we have two Aces
        if should_redraw and card_idx != -1:
            card_to_discard = my_cards[card_idx]
            card_rank = card_to_discard // 3
            
            other_card = my_cards[1 - card_idx]
            other_rank = other_card // 3
            
            # Don't discard an Ace (rank 8) unless we have two Aces
            if card_rank == 8 and other_rank != 8:
                return False, -1
            
            # Don't discard a strong card if the improvement would be marginal
            current_strength = self.hand_evaluator.get_hand_strength(my_cards, community_cards)
            if card_rank >= 6 and current_strength > 0.6:  # High card, already strong hand
                # Only discard if we have a very good reason
                if random.random() > 0.1:  # 90% of the time, keep the strong card
                    return False, -1
        
        return should_redraw, card_idx
    
    def _preflop_strategy(self, obs, position):
        """
        Professional preflop strategy
        """
        my_cards = obs["my_cards"]
        hand_strength = self.hand_evaluator.get_hand_strength(my_cards)
        is_premium = (tuple(sorted(my_cards)) in self.premium_hands)
        valid_actions = obs["valid_actions"]
        aggression = self.opponent_model.get_aggression_factor()
        
        # Adjust thresholds based on opponent
        raise_threshold = self.preflop_raise_threshold * (1.1 if aggression > 0.7 else 0.9)
        call_threshold = self.preflop_call_threshold * (1.1 if aggression < 0.4 else 0.9)
        
        # Position-based adjustments
        if position == 'SB':
            raise_threshold *= 0.9  # More aggressive in position
            call_threshold *= 0.9
        else:
            raise_threshold *= 1.1  # More selective out of position
            call_threshold *= 1.1
        
        my_bet = obs["my_bet"]
        opp_bet = obs["opp_bet"]
        min_raise = obs.get("min_raise", 2)
        max_raise = obs.get("max_raise", 100)
        
        # Check if opponent has raised
        facing_raise = opp_bet > (1 if position == 'SB' else 2)
        
        # Strategic decision tree
        if is_premium or hand_strength > raise_threshold:
            # Strong hand - raise or re-raise
            if valid_actions[action_types.RAISE.value]:
                # Size bet based on position and hand strength
                if facing_raise:
                    # Re-raise - size depends on hand strength
                    if hand_strength > 0.8:
                        # Premium hand - larger re-raise
                        raise_size = min(max_raise, max(min_raise, int(opp_bet * 2.5)))
                    else:
                        # Good hand - standard re-raise
                        raise_size = min(max_raise, max(min_raise, int(opp_bet * 2)))
                else:
                    # Open raise
                    if position == 'SB':
                        raise_size = min(max_raise, max(min_raise, 3))  # 3x open from SB
                    else:
                        raise_size = min(max_raise, max(min_raise, 4))  # 4x open from BB
                
                return (action_types.RAISE.value, raise_size, -1)
            
            # Can't raise, so call
            if valid_actions[action_types.CALL.value]:
                return (action_types.CALL.value, 0, -1)
        
        elif hand_strength > call_threshold:
            # Medium hand
            if facing_raise:
                # Against raise - mainly call but occasionally re-raise as semi-bluff
                if valid_actions[action_types.CALL.value]:
                    # Sometimes re-raise as a semi-bluff
                    bluff_chance = 0.15 if position == 'SB' else 0.08
                    if random.random() < bluff_chance and valid_actions[action_types.RAISE.value]:
                        raise_size = min(max_raise, max(min_raise, int(opp_bet * 2.2)))
                        return (action_types.RAISE.value, raise_size, -1)
                    return (action_types.CALL.value, 0, -1)
            else:
                # No raise yet - raise first
                if valid_actions[action_types.RAISE.value]:
                    raise_size = min(max_raise, max(min_raise, 2))
                    return (action_types.RAISE.value, raise_size, -1)
                elif valid_actions[action_types.CALL.value]:
                    return (action_types.CALL.value, 0, -1)
                else:
                    return (action_types.CHECK.value, 0, -1)
        
        else:
            # Weak hand
            if facing_raise:
                # Against raise - mainly fold but occasionally call light
                if valid_actions[action_types.CALL.value] and random.random() < 0.1:
                    return (action_types.CALL.value, 0, -1)
                else:
                    return (action_types.FOLD.value, 0, -1)
            else:
                # No raise - check if possible, or minimum raise as bluff
                if valid_actions[action_types.CHECK.value]:
                    return (action_types.CHECK.value, 0, -1)
                elif valid_actions[action_types.RAISE.value] and random.random() < 0.15:
                    # Occasional bluff raise
                    raise_size = min(max_raise, min_raise)
                    return (action_types.RAISE.value, raise_size, -1)
                else:
                    return (action_types.FOLD.value, 0, -1)
        
        # Default action - check or fold
        if valid_actions[action_types.CHECK.value]:
            return (action_types.CHECK.value, 0, -1)
        return (action_types.FOLD.value, 0, -1)
    
    def _postflop_strategy(self, obs, position):
        """
        Professional postflop strategy
        """
        my_cards = obs["my_cards"]
        community_cards = [c for c in obs["community_cards"] if c != -1]
        street = obs["street"]
        valid_actions = obs["valid_actions"]
        
        # Calculate hand strength and drawing potential
        hand_strength = self.hand_evaluator.get_hand_strength(my_cards, community_cards)
        
        # Adjust thresholds based on street
        value_threshold = self.postflop_value_threshold
        bluff_threshold = self.postflop_bluff_threshold
        
        # Adjust based on street
        if street == 1:  # Flop
            value_threshold -= 0.05  # More aggressive on flop
            bluff_threshold -= 0.05
        elif street == 3:  # River
            value_threshold += 0.05  # More selective on river
            bluff_threshold += 0.1   # Less bluffing on river
        
        # Consider opponent tendencies
        fold_equity = self.opponent_model.get_fold_equity()
        aggression = self.opponent_model.get_aggression_factor()
        
        # Adjust thresholds based on opponent
        if fold_equity > 0.7:
            # Opponent folds too much - bluff more
            bluff_threshold -= 0.1
            value_threshold -= 0.05
        elif aggression > 0.7:
            # Opponent very aggressive - tighten up for value, loosen for bluffs
            value_threshold += 0.05
            bluff_threshold -= 0.05
        
        my_bet = obs["my_bet"]
        opp_bet = obs["opp_bet"]
        pot_size = my_bet + opp_bet
        min_raise = obs.get("min_raise", 1)
        max_raise = obs.get("max_raise", 100)
        bet_to_call = opp_bet - my_bet
        
        # Check for slowplay opportunity with monster hands
        slowplay_opportunity = (
            hand_strength > self.slowplay_threshold and 
            street < 3 and  # Don't slowplay on river
            bet_to_call == 0  # No bet to call
        )
        
        # Factor in position
        if position == 'SB':
            value_threshold -= 0.05  # More aggressive in position
            bluff_threshold -= 0.05
        else:
            value_threshold += 0.05  # More selective out of position
            bluff_threshold += 0.05
        
        # Strategic decision tree
        if hand_strength > value_threshold and not slowplay_opportunity:
            # Strong hand - bet or raise for value
            if bet_to_call == 0:
                # No bet to call - we can bet first
                if valid_actions[action_types.RAISE.value]:
                    # Size based on street
                    if street == 1:  # Flop
                        bet_size = max(min_raise, int(pot_size * 0.6))
                    elif street == 2:  # Turn
                        bet_size = max(min_raise, int(pot_size * 0.7))
                    else:  # River
                        bet_size = max(min_raise, int(pot_size * 0.8))
                    
                    bet_size = min(max_raise, bet_size)
                    return (action_types.RAISE.value, bet_size, -1)
                else:
                    return (action_types.CHECK.value, 0, -1)
            else:
                # Facing a bet - raise for value or call
                if valid_actions[action_types.RAISE.value] and hand_strength > value_threshold + 0.1:
                    # Strong enough to raise
                    raise_size = min(max_raise, max(min_raise, int(bet_to_call * 2.5)))
                    return (action_types.RAISE.value, raise_size, -1)
                elif valid_actions[action_types.CALL.value]:
                    # Call with good but not raising hands
                    return (action_types.CALL.value, 0, -1)
                else:
                    return (action_types.FOLD.value, 0, -1)
        
        elif hand_strength < bluff_threshold:
            # Weak hand - consider bluffing
            bluff_chance = self.bluff_frequency
            
            # Adjust bluff frequency based on fold equity and position
            bluff_chance *= fold_equity * 1.5
            if position == 'SB':
                bluff_chance *= 1.3  # Bluff more in position
            
            # On river, adjust bluff frequency based on showdown potential
            if street == 3:
                bluff_chance *= 0.5  # Much less bluffing on river
            
            if bet_to_call == 0:
                # No bet - potential for bluff bet
                if valid_actions[action_types.RAISE.value] and random.random() < bluff_chance:
                    # Bluff size - smaller on flop, larger on later streets
                    if street == 1:
                        bet_size = max(min_raise, int(pot_size * 0.5))
                    elif street == 2:
                        bet_size = max(min_raise, int(pot_size * 0.6))
                    else:
                        bet_size = max(min_raise, int(pot_size * 0.7))
                    
                    bet_size = min(max_raise, bet_size)
                    return (action_types.RAISE.value, bet_size, -1)
                else:
                    return (action_types.CHECK.value, 0, -1)
            else:
                # Facing a bet - float or fold
                pot_odds = bet_to_call / (pot_size + bet_to_call)
                
                # Check if we have proper pot odds
                if valid_actions[action_types.CALL.value] and (
                    pot_odds < hand_strength or  # If pot odds are good
                    (random.random() < bluff_chance * 0.7)  # Or semi-bluff opportunity
                ):
                    return (action_types.CALL.value, 0, -1)
                else:
                    return (action_types.FOLD.value, 0, -1)
        
        else:
            # Medium strength hand
            if bet_to_call == 0:
                # No bet to call - check or small value bet
                if valid_actions[action_types.RAISE.value] and hand_strength > 0.5:
                    # Small value bet with medium hand
                    bet_size = max(min_raise, int(pot_size * 0.4))
                    bet_size = min(max_raise, bet_size)
                    return (action_types.RAISE.value, bet_size, -1)
                else:
                    return (action_types.CHECK.value, 0, -1)
            else:
                # Facing a bet - check pot odds
                pot_odds = bet_to_call / (pot_size + bet_to_call)
                if valid_actions[action_types.CALL.value] and pot_odds < hand_strength:
                    return (action_types.CALL.value, 0, -1)
                else:
                    return (action_types.FOLD.value, 0, -1)
        
        # Slowplay logic for monster hands
        if slowplay_opportunity and valid_actions[action_types.CHECK.value]:
            return (action_types.CHECK.value, 0, -1)
        
        # Default action - check or fold
        if valid_actions[action_types.CHECK.value]:
            return (action_types.CHECK.value, 0, -1)
        return (action_types.FOLD.value, 0, -1)
    
    def _update_strategic_phase(self):
        """
        Update strategy based on tournament phase
        """
        if self.hands_played <= self.strategy_phases['early']['hands']:
            # Early phase - tight aggressive to gather information
            self.bluff_frequency = 0.10  # Low bluff frequency
            self.preflop_raise_threshold = 0.55  # Higher threshold for raising
            self.preflop_call_threshold = 0.40
        
        elif self.hands_played <= self.strategy_phases['middle']['hands']:
            # Middle phase - balanced play
            self.bluff_frequency = 0.15
            self.preflop_raise_threshold = 0.50
            self.preflop_call_threshold = 0.35
            
            # Adjust based on opponent tendencies
            if hasattr(self.opponent_model, 'fold_frequency'):
                if self.opponent_model.fold_frequency > 0.7:
                    # Opponent folds too much - bluff more
                    self.bluff_frequency += 0.1
                    self.exploit_factor = 0.3
        
        else:
            # Late phase - fully adaptive exploitative play
            # Heavy adjustments based on opponent profile
            aggression = self.opponent_model.get_aggression_factor()
            fold_freq = self.opponent_model.get_fold_equity()
            
            if fold_freq > 0.7:
                # Exploitative adjustment for folders
                self.bluff_frequency = 0.25
                self.preflop_raise_threshold = 0.45
                self.exploit_factor = 0.5
            elif aggression > 0.7:
                # Exploitative adjustment for aggressive opponents
                self.bluff_frequency = 0.10
                self.preflop_raise_threshold = 0.6
                self.postflop_value_threshold = 0.65
                self.exploit_factor = 0.4
            else:
                # Balanced approach for unknown or balanced opponents
                self.bluff_frequency = 0.15
                self.preflop_raise_threshold = 0.5
                self.exploit_factor = 0.2
    
    def _is_new_hand(self, obs):
        """
        Detect if this observation represents the start of a new hand
        """
        is_preflop = obs["street"] == 0
        has_blinds = obs["my_bet"] <= 2 and obs["opp_bet"] <= 2
        
        if is_preflop and has_blinds:
            if self.prev_obs is None:
                return True
                
            # Check if we've advanced from a previous hand
            prev_was_river = self.prev_obs.get("street", 0) == 3
            return prev_was_river
            
        return False
    
    def _update_opponent_model(self, prev_obs, curr_obs):
        """
        Update opponent model based on observed changes
        """
        opp_discarded_card = curr_obs.get("opp_discarded_card", -1)
        opp_drawn_card = curr_obs.get("opp_drawn_card", -1)
        
        if opp_discarded_card != -1 and opp_drawn_card != -1:
            self.opponent_model.update_from_redraw(
                opp_discarded_card, opp_drawn_card, curr_obs["street"])
        
        # Detect betting actions
        if prev_obs.get("acting_agent") == curr_obs.get("acting_agent"):
            street = curr_obs["street"]
            prev_opp_bet = prev_obs.get("opp_bet", 0)
            curr_opp_bet = curr_obs.get("opp_bet", 0)
            
            if curr_opp_bet > prev_opp_bet:
                # Opponent raised
                bet_size = curr_opp_bet - prev_opp_bet
                self.opponent_model.update_from_action(
                    action_types.RAISE.value, street, bet_size=bet_size)
            elif curr_opp_bet == prev_opp_bet and prev_obs["street"] == curr_obs["street"]:
                # Opponent checked
                self.opponent_model.update_from_action(action_types.CHECK.value, street)
            elif curr_opp_bet == prev_obs.get("my_bet", 0) and curr_opp_bet > prev_opp_bet:
                # Opponent called
                self.opponent_model.update_from_action(action_types.CALL.value, street)
            elif prev_opp_bet > curr_opp_bet or curr_obs["street"] > prev_obs["street"]:
                # Likely folded or moved to next street
                if prev_obs["street"] == curr_obs["street"]:
                    self.opponent_model.update_from_action(action_types.FOLD.value, street)
        
        # Detect showdown
        if prev_obs.get("street") == 3 and curr_obs.get("street") == 0:
            # End of hand - in a real implementation, we'd have access to
            # showdown information like opponent's cards
            self.opponent_model.update_from_hand({})
    
    def _emergency_strategy(self, obs):
        """
        Simplified strategy for emergency time situations
        """
        my_cards = obs["my_cards"]
        community_cards = obs.get("community_cards", [])
        street = obs["street"]
        valid_actions = obs["valid_actions"]
        my_bet = obs["my_bet"]
        opp_bet = obs["opp_bet"]
        
        # Quick hand strength approximation
        if street == 0:  # Preflop
            card1_rank = my_cards[0] // 3
            card2_rank = my_cards[1] // 3
            
            # Pair
            if card1_rank == card2_rank:
                strength = 0.9 if card1_rank >= 6 else 0.7  # High pair vs low pair
            # High cards (8, 9, A)
            elif card1_rank >= 6 and card2_rank >= 6:
                strength = 0.8  # Two high cards
            # One high card
            elif card1_rank >= 6 or card2_rank >= 6:
                strength = 0.5  # One high card
            # Connected cards
            elif abs(card1_rank - card2_rank) <= 2:
                strength = 0.4  # Connected low cards
            else:
                strength = 0.2  # Unconnected low cards
            
            # Adjust for suited cards
            card1_suit = my_cards[0] % 3
            card2_suit = my_cards[1] % 3
            if card1_suit == card2_suit:
                strength += 0.1  # Suited bonus
        else:
            # Post-flop - quick hand evaluation
            strength = self.hand_evaluator.get_hand_strength(my_cards, community_cards)
        
        # Quick discard decision
        if valid_actions[action_types.DISCARD.value]:
            if strength < 0.4:  # Only discard weak hands
                card1_rank = my_cards[0] // 3
                card2_rank = my_cards[1] // 3
                discard_idx = 0 if card1_rank < card2_rank else 1
                
                # Never discard an Ace
                if (discard_idx == 0 and card1_rank == 8) or (discard_idx == 1 and card2_rank == 8):
                    discard_idx = 1 if card1_rank == 8 else 0
                
                return (action_types.DISCARD.value, 0, discard_idx)
        
        # Betting decisions
        min_raise = obs.get("min_raise", 1)
        max_raise = obs.get("max_raise", 100)
        bet_to_call = opp_bet - my_bet
        
        # No bet to call
        if bet_to_call == 0:
            # Strong hand - bet
            if strength > 0.6 and valid_actions[action_types.RAISE.value]:
                # Size bet appropriately
                if street == 0:  # Preflop
                    bet_size = min(max_raise, max(min_raise, 3))
                else:  # Postflop
                    pot_size = my_bet + opp_bet
                    bet_size = min(max_raise, max(min_raise, int(pot_size * 0.6)))
                
                return (action_types.RAISE.value, bet_size, -1)
            else:
                return (action_types.CHECK.value, 0, -1)
        else:
            # Facing a bet
            if strength > 0.7:  # Strong hand
                if valid_actions[action_types.RAISE.value]:
                    # Raise with strong hands
                    raise_size = min(max_raise, max(min_raise, int(bet_to_call * 2.5)))
                    return (action_types.RAISE.value, raise_size, -1)
                else:
                    return (action_types.CALL.value, 0, -1)
            elif strength > 0.4:  # Medium hand
                return (action_types.CALL.value, 0, -1)
            else:  # Weak hand
                return (action_types.FOLD.value, 0, -1)
    
    def _log_action(self, obs, action):
        """
        Log significant actions for review
        """
        if action[0] == action_types.RAISE.value and action[1] > 10:
            self.logger.info(f"Large raise of {action[1]} on street {obs['street']}")
        elif action[0] == action_types.DISCARD.value:
            card_idx = action[2]
            if card_idx != -1:
                card = obs["my_cards"][card_idx]
                card_rank = card // 3
                card_suit = card % 3
                rank_name = "A" if card_rank == 8 else str(card_rank + 2)
                suit_name = ["♦", "♥", "♠"][card_suit]
                self.logger.info(f"Discarding {rank_name}{suit_name}")
        
    def observe(self, observation, reward, terminated, truncated, info):
        """
        Enhanced observation method to track opponent actions when it's their turn
        """
        if terminated:
            # Record hand result
            self.hand_number += 1
            if reward > 0:
                self.won_hands += 1
                self.logger.info(f"Won hand #{self.hand_number} with reward {reward}")
            elif reward < 0:
                self.logger.info(f"Lost hand #{self.hand_number} with reward {reward}")
            
            # Record showdown info if available
            if info.get("player_0_cards") and info.get("player_1_cards"):
                self.showdowns.append({
                    'hand_number': self.hand_number,
                    'result': 'win' if reward > 0 else 'loss',
                    'my_cards': info.get("player_0_cards" if observation["acting_agent"] == 0 else "player_1_cards", []),
                    'opp_cards': info.get("player_1_cards" if observation["acting_agent"] == 0 else "player_0_cards", []),
                    'board': info.get("community_cards", [])
                })
                
                # Use showdown information to refine strategy
                self._analyze_showdown(self.showdowns[-1])
        else:
            # Track significant opponent actions
            opp_action_type = None
            if self.prev_obs is not None:
                prev_opp_bet = self.prev_obs.get("opp_bet", 0)
                curr_opp_bet = observation.get("opp_bet", 0)
                
                if curr_opp_bet > prev_opp_bet:
                    opp_action_type = "raise"
                elif curr_opp_bet == prev_opp_bet and self.prev_obs.get("street") == observation.get("street"):
                    opp_action_type = "check"
                elif curr_opp_bet == self.prev_obs.get("my_bet", 0) and curr_opp_bet > prev_opp_bet:
                    opp_action_type = "call"
                
                if opp_action_type is not None:
                    self._process_opponent_action(opp_action_type, observation)
            
            # Update prev_obs for next comparison
            self.prev_obs = observation.copy()
    
    def _process_opponent_action(self, action_type, obs):
        """
        Process significant opponent actions to refine strategy
        """
        street = obs["street"]
        street_names = ["Preflop", "Flop", "Turn", "River"]
        
        if action_type == "raise" and obs["opp_bet"] > 15:
            # Log large bets - these are significant
            self.logger.info(f"Opponent made large {action_type} to {obs['opp_bet']} on {street_names[street]}")
            
            # Update exploitation strategy
            if street >= 2:  # Turn or River
                # Large bets on later streets indicate either strength or bluff
                # Track pattern to exploit later
                self.opponent_model.update_from_action(action_types.RAISE.value, street, bet_size=obs["opp_bet"])
                
                # Adjust our calling ranges based on observed showdown data
                if len(self.showdowns) > 5:
                    # Look at history of opponent's large bets
                    large_bet_showdowns = [s for s in self.showdowns if s.get('opp_big_bet_street') == street]
                    if large_bet_showdowns:
                        # Calculate how often they're bluffing with large bets
                        bluff_count = sum(1 for s in large_bet_showdowns if s.get('was_bluff', False))
                        if bluff_count / len(large_bet_showdowns) > 0.6:
                            # They bluff too much - call lighter
                            self.postflop_value_threshold -= 0.1
                            self.logger.info(f"Adjusted calling threshold based on opponent bluff tendency")
    
    def _analyze_showdown(self, showdown_info):
        """
        Analyze showdown results to refine strategy
        """
        # Skip if incomplete information
        if not showdown_info.get('my_cards') or not showdown_info.get('opp_cards'):
            return
        
        # Extract key information
        result = showdown_info.get('result')
        opp_cards = showdown_info.get('opp_cards')
        
        # Analyze opponent's final hand strength
        if result == 'loss' and opp_cards:
            # Convert card strings to indices for the evaluator
            opp_card_indices = self._card_strings_to_indices(opp_cards)
            board_indices = self._card_strings_to_indices(showdown_info.get('board', []))
            
            if opp_card_indices and board_indices:
                # Calculate their actual hand strength
                opp_strength = self.hand_evaluator.get_hand_strength(opp_card_indices, board_indices)
                
                # Check for bluffs or strong value bets
                if 'opp_bet_sizes' in showdown_info:
                    final_street = max(showdown_info.get('opp_bet_sizes', {}).keys())
                    final_bet = showdown_info.get('opp_bet_sizes', {}).get(final_street, 0)
                    
                    if final_bet > 15:  # Large bet
                        was_bluff = opp_strength < 0.6  # Define bluff threshold
                        showdown_info['was_bluff'] = was_bluff
                        
                        if was_bluff:
                            self.logger.info(f"Detected opponent bluff in hand {showdown_info['hand_number']}")
                            # Adjust our calling threshold
                            self.postflop_value_threshold = max(0.4, self.postflop_value_threshold - 0.05)
                        else:
                            # They had it - adjust our bluffing frequency
                            self.bluff_frequency = max(0.1, self.bluff_frequency - 0.02)
    
    def _card_strings_to_indices(self, card_strings):
        """
        Convert card strings (e.g., '8d') to card indices (0-26)
        """
        if not card_strings:
            return []
            
        card_indices = []
        for card_str in card_strings:
            if len(card_str) != 2:
                continue
                
            rank_char, suit_char = card_str[0], card_str[1]
            
            # Map rank
            if rank_char == 'A':
                rank_idx = 8  # Ace is highest
            elif '2' <= rank_char <= '9':
                rank_idx = int(rank_char) - 2
            else:
                continue  # Invalid rank
            
            # Map suit
            if suit_char == 'd':
                suit_idx = 0
            elif suit_char == 'h':
                suit_idx = 1
            elif suit_char == 's':
                suit_idx = 2
            else:
                continue  # Invalid suit
                
            card_idx = rank_idx * 3 + suit_idx
            card_indices.append(card_idx)
            
        return card_indices

# Updated Decision Engine with advanced GTO concepts

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

# Enhanced Redraw Strategy with reduced deception

class Redraw:
    """
    Advanced redraw strategy using GTO concepts for 27-card deck poker.
    Implements Expected Value (EV) maximization with efficient pruning for performance.
    """
    
    def __init__(self, hand_evaluator):
        self.hand_evaluator = hand_evaluator
        self.redraw_cache = {}
        
        # Significantly reduced deception threshold to avoid discarding good cards
        self.deception_threshold = 0.02  # Down from 0.1
        
    def strategic_redraw(self, hole_cards, board, street, position, opponent_model=None):
        """
        Advanced redraw strategy for poker variant with 27-card deck
        
        Args:
            hole_cards: List of hole card indices
            board: List of board card indices
            street: Current street
            position: "SB" or "BB"
            opponent_model: OpponentModel instance
            
        Returns:
            Tuple of (should_redraw, card_idx_to_discard)
        """
        # Core redraw decision
        should_redraw, card_idx = self.should_redraw(
            hole_cards, board, street, position,
            opponent_model.__dict__ if opponent_model else None
        )
        
        # If opponent model is available, make exploitative adjustments
        if opponent_model and hasattr(opponent_model, 'get_redraw_frequency'):
            # Extract opponent tendencies
            opp_redraw_freq = opponent_model.get_redraw_frequency()
            opp_fold_equity = opponent_model.get_fold_equity()
            opp_aggression = getattr(opponent_model, 'aggression', 0.5)
            
            # Current hand strength
            current_strength = self.hand_evaluator.get_hand_strength(hole_cards, board)
            
            # Case 1: Opponent redraws very often - we should be more selective
            if opp_redraw_freq > 0.7:
                # Only redraw very weak hands or for significant improvements
                if current_strength > 0.4 and should_redraw:
                    # Verify the improvement is truly significant
                    improvements = self._calculate_targeted_improvements(
                        hole_cards, board, street, position, opponent_model.__dict__
                    )
                    
                    # Only redraw for major improvements
                    if improvements[0][0] < 0.15:  # Need 15% improvement
                        return False, -1
            
            # Case 2: Opponent rarely redraws - we can be more aggressive with redraws
            elif opp_redraw_freq < 0.3 and not should_redraw:
                # Check for even marginal improvements
                improvements = self._calculate_targeted_improvements(
                    hole_cards, board, street, position, opponent_model.__dict__
                )
                
                # Redraw for even small improvements
                if improvements[0][0] > 0.05:  # Just 5% improvement needed
                    return True, improvements[0][1]
            
            # Case 3: Deceptive redraw with strong hands - REDUCED SIGNIFICANTLY
            if not should_redraw and current_strength > 0.85:  # Only for very strong hands
                # Calculate the balance between deception and value
                # Much lower deception threshold
                deception_threshold = self.deception_threshold  # Using class property (0.02)
                
                # Adjust based on opponent type:
                # - Against observant opponents (lower fold equity), deception is more valuable
                # - Against unobservant opponents (high fold equity), keeping value is better
                deception_threshold *= (1.5 - opp_fold_equity)
                
                # Additional restrictions to prevent discarding Aces
                if random.random() < deception_threshold:
                    # Need at least 10+ hands played before attempting deception
                    if opponent_model.hands_seen < 10:
                        return should_redraw, card_idx
                        
                    # Determine which card to discard (typically the lower one)
                    card1_rank = hole_cards[0] // 3
                    card2_rank = hole_cards[1] // 3
                    
                    # Never discard an Ace
                    if card1_rank == 8 or card2_rank == 8:
                        return should_redraw, card_idx
                        
                    # Discard lower card
                    discard_idx = 0 if card1_rank < card2_rank else 1
                    return True, discard_idx
        
        # Extra protection layer - never discard an Ace unless we have two Aces
        if should_redraw and card_idx != -1:
            card_to_discard = hole_cards[card_idx]
            card_rank = card_to_discard // 3
            
            other_card = hole_cards[1 - card_idx]
            other_rank = other_card // 3
            
            # Don't discard an Ace (rank 8) unless we have two Aces
            if card_rank == 8 and other_rank != 8:
                return False, -1
        
        return should_redraw, card_idx
    
    def should_redraw(self, hole_cards, board, street, position="SB", opponent_tendencies=None):
        """
        Core redraw decision-making using EV maximization
        
        Args:
            hole_cards: List of hole card indices
            board: List of board card indices
            street: Current street (0=preflop, 1=flop)
            position: "SB" or "BB"
            opponent_tendencies: Dict of opponent stats
            
        Returns:
            Tuple of (should_redraw, card_idx_to_discard)
        """
        if street > 1:
            return False, -1
        
        cache_key = (tuple(hole_cards), tuple(board), street, position)
        if cache_key in self.redraw_cache:
            return self.redraw_cache[cache_key]
        
        # Get current hand strength with potential range information
        opponent_range = self._construct_opponent_range(opponent_tendencies, street, position)
        current_strength = self.hand_evaluator.get_hand_strength(hole_cards, board, opponent_range)
        
        # For very strong hands, don't redraw
        if current_strength > 0.8:
            self.redraw_cache[cache_key] = (False, -1)
            return False, -1
        
        # Calculate expected value of redrawing vs not redrawing
        keep_ev = self._calculate_keep_ev(hole_cards, board, street, position, opponent_tendencies)
        
        # Calculate EV of redrawing each card
        redraw_options = []
        
        for card_idx in range(2):
            # Get the card we're keeping
            other_card = hole_cards[1-card_idx]
            
            # Cards that are already in play and unavailable
            unavailable_cards = set(hole_cards) | set(board)
            
            # Determine available cards for replacement
            available_cards = set(range(27)) - unavailable_cards
            
            # Group cards into buckets by rank for more efficient sampling
            rank_buckets = {}
            for card in available_cards:
                rank = card // 3
                if rank not in rank_buckets:
                    rank_buckets[rank] = []
                rank_buckets[rank].append(card)
            
            # Adaptive sample size based on available time
            sample_size = 15 if street == 0 else 10
            
            # Sample cards, giving preference to high ranks
            sampled_cards = []
            # Sample high cards (A, 9, 8) more frequently
            high_ranks = [8, 7, 6]  # A, 9, 8
            for rank in high_ranks:
                if rank in rank_buckets:
                    sampled_cards.extend(rank_buckets[rank])
            
            # If we need more samples, add middle ranks
            if len(sampled_cards) < sample_size:
                mid_ranks = [5, 4, 3]  # 7, 6, 5
                for rank in mid_ranks:
                    if rank in rank_buckets and len(sampled_cards) < sample_size:
                        sampled_cards.extend(rank_buckets[rank])
            
            # If still need more, add remaining ranks
            remaining_ranks = [2, 1, 0]  # 4, 3, 2
            for rank in remaining_ranks:
                if rank in rank_buckets and len(sampled_cards) < sample_size:
                    sampled_cards.extend(rank_buckets[rank])
            
            # Limit to desired sample size
            if len(sampled_cards) > sample_size:
                sampled_cards = random.sample(sampled_cards, sample_size)
            
            # Calculate EV for each replacement
            total_ev = 0
            weight_sum = 0
            
            for new_card in sampled_cards:
                # Higher weight for higher ranked cards
                rank = new_card // 3
                weight = 1.0 + (rank / 8.0)  # Weight from 1.0 to 2.0
                
                # Create new hand
                new_hand = [other_card, new_card]
                
                # Calculate new hand EV
                new_ev = self._calculate_hand_ev(new_hand, board, street, position, opponent_tendencies)
                
                # Weighted sum
                total_ev += new_ev * weight
                weight_sum += weight
            
            # Calculate weighted average EV
            if weight_sum > 0:
                avg_ev = total_ev / weight_sum
                redraw_options.append((avg_ev, card_idx))
            else:
                redraw_options.append((0, card_idx))
        
        # Find best redraw option
        redraw_options.sort(reverse=True)
        best_redraw_ev, best_card_idx = redraw_options[0]
        
        # Compare to keeping current hand - increased threshold to be more conservative
        if best_redraw_ev > keep_ev + 0.07:  # Increased from 0.05 to 0.07
            decision = (True, best_card_idx)
        else:
            decision = (False, -1)
        
        # Extra protection - never discard an Ace unless we have a pair of Aces
        if decision[0] and decision[1] != -1:
            card_to_discard = hole_cards[decision[1]]
            card_rank = card_to_discard // 3
            
            other_card = hole_cards[1 - decision[1]]
            other_rank = other_card // 3
            
            if card_rank == 8 and other_rank != 8:  # Trying to discard an Ace
                decision = (False, -1)  # Don't discard
        
        # Cache result
        self.redraw_cache[cache_key] = decision
        return decision
    
    def _construct_opponent_range(self, opponent_tendencies, street, position):
        """
        Construct opponent range model from tendencies.
        """
        # Default range model for unknown opponent
        if not opponent_tendencies:
            return {
                'polarization': 0.5,  # How polarized the range is (0=condensed, 1=polarized)
                'strength_cap': 1.0,   # Upper bound on hand strength
                'strength_floor': 0.0  # Lower bound on hand strength
            }
        
        # Extract relevant tendencies
        aggression = opponent_tendencies.get('aggression', 0.5)
        fold_frequency = opponent_tendencies.get('fold_frequency', 0.5)
        
        # Calculate range polarization
        # More aggressive opponents tend to have more polarized ranges
        polarization = min(1.0, aggression * 1.2)
        
        # Calculate strength bounds
        # Tight opponents have higher strength floors
        strength_floor = min(0.4, (1 - fold_frequency) * 0.5)
        
        # Street-specific adjustments
        if street == 0:  # Preflop
            pass  # Use base values
        elif street == 1:  # Flop
            # Ranges typically get stronger post-flop
            strength_floor += 0.1
        
        return {
            'polarization': polarization,
            'strength_cap': 1.0,
            'strength_floor': strength_floor
        }
    
    def _calculate_keep_ev(self, hole_cards, board, street, position, opponent_tendencies):
        """
        Calculate EV of keeping current hand.
        """
        return self._calculate_hand_ev(hole_cards, board, street, position, opponent_tendencies)
    
    def _calculate_hand_ev(self, hole_cards, board, street, position, opponent_tendencies):
        """
        Calculate expected value of a hand in the current game state.
        """
        # Construct opponent range model
        opponent_range = self._construct_opponent_range(opponent_tendencies, street, position)
        
        # Get hand strength against range
        hand_strength = self.hand_evaluator.get_hand_strength(hole_cards, board, opponent_range)
        
        # Position-based adjustments
        position_factor = 1.05 if position == "SB" else 0.95
        
        # Street-based adjustments
        street_factor = 1.0
        if street == 1:  # Flop
            # On the flop, position matters more
            position_factor = 1.1 if position == "SB" else 0.9
        
        # Calculate simplified EV
        return hand_strength * position_factor * street_factor
    
    def _calculate_targeted_improvements(self, hole_cards, board, street, position, opponent_tendencies):
        """
        Calculate improvements with targeted card sampling for efficiency.
        """
        current_ev = self._calculate_hand_ev(hole_cards, board, street, position, opponent_tendencies)
        improvements = []
        
        for card_idx in range(2):
            other_card = hole_cards[1-card_idx]
            
            # Get unavailable cards
            unavailable = set(hole_cards) | set(board)
            
            # Advanced importance sampling focused on highest potential improvement
            available_cards = []
            
            # First, check high cards (always worth sampling)
            high_ranks = [8, 7, 6]  # A, 9, 8
            for rank in high_ranks:
                for suit in range(3):
                    card = rank * 3 + suit
                    if card not in unavailable:
                        available_cards.append(card)
            
            # Then add potential straight/flush completers if on the flop
            if board and len(board) >= 3:
                # Cards that might complete straights or flushes
                all_ranks = [c // 3 for c in [other_card] + board]
                all_suits = [c % 3 for c in [other_card] + board]
                
                # Possible straight completers
                for rank in range(9):  # 2-A
                    # Check if this rank could complete a straight
                    straight_potential = False
                    for i in range(max(0, rank-4), min(9, rank+1)):
                        count = sum(1 for r in all_ranks if i <= r < i+5)
                        if count >= 3:  # Could complete a straight
                            straight_potential = True
                            break
                    
                    if straight_potential:
                        for suit in range(3):
                            card = rank * 3 + suit
                            if card not in unavailable and card not in available_cards:
                                available_cards.append(card)
                
                # Possible flush completers
                suit_counts = {}
                for s in all_suits:
                    suit_counts[s] = suit_counts.get(s, 0) + 1
                
                for suit, count in suit_counts.items():
                    if count >= 3:  # Potential flush
                        for rank in range(9):
                            card = rank * 3 + suit
                            if card not in unavailable and card not in available_cards:
                                available_cards.append(card)
            
            # Add a few random cards to ensure diversity
            all_remaining = [c for c in range(27) 
                            if c not in unavailable and c not in available_cards]
            
            if all_remaining:
                random_sample = random.sample(
                    all_remaining, 
                    min(5, len(all_remaining))
                )
                available_cards.extend(random_sample)
            
            # Calculate improvement for each card
            card_improvements = []
            
            for new_card in available_cards:
                new_hand = [other_card, new_card]
                new_ev = self._calculate_hand_ev(
                    new_hand, board, street, position, opponent_tendencies
                )
                improvement = new_ev - current_ev
                card_improvements.append(improvement)
            
            # Average improvement
            if card_improvements:
                avg_improvement = sum(card_improvements) / len(card_improvements)
                # Find max improvement (optimistic estimate)
                max_improvement = max(card_improvements) if card_improvements else 0
                # Use weighted combination of average and max
                combined_improvement = 0.7 * avg_improvement + 0.3 * max_improvement
                improvements.append((combined_improvement, card_idx))
            else:
                improvements.append((0, card_idx))
        
        # Sort by improvement (descending)
        return sorted(improvements, reverse=True)

# Updated Hand Evaluator optimized for the 27-card deck

class HandEvaluator:
    """
    Optimized hand evaluator for the 27-card deck variant.
    Implements fast evaluation tailored for the specific deck configuration.
    """
    
    def __init__(self):
        self.RANKS = "23456789A"
        self.SUITS = "dhs"  # diamonds, hearts, spades
        
        # Pre-computed lookup tables
        self.hand_strength_cache = {}
        self.starting_hand_values = self._precompute_starting_hands()
    
    def _card_str_to_idx(self, card_str):
        """Convert card string (e.g., '2d') to card index (0-26)."""
        rank, suit = card_str[0], card_str[1]
        rank_idx = self.RANKS.index(rank)
        suit_idx = self.SUITS.index(suit)
        return rank_idx * 3 + suit_idx
    
    def _card_idx_to_str(self, card_idx):
        """Convert card index (0-26) to card string (e.g., '2d')."""
        if card_idx == -1:
            return "None"
        rank_idx = card_idx // 3
        suit_idx = card_idx % 3
        return f"{self.RANKS[rank_idx]}{self.SUITS[suit_idx]}"
    
    def _precompute_starting_hands(self):
        """
        Pre-compute starting hand values tailored to the 27-card deck
        """
        starting_hands = {}
        
        def monte_carlo_equity(card1_idx, card2_idx):
            """
            Calculate starting hand equity using specialized heuristics for 27-card deck
            """
            rank1 = card1_idx // 3
            rank2 = card2_idx // 3
            suit1 = card1_idx % 3
            suit2 = card2_idx % 3
            
            # Paired hands are very strong in this deck
            if rank1 == rank2:
                # Rank value from 0 (2) to 8 (A)
                pair_strength = rank1 / 8
                return 0.75 + pair_strength * 0.25  # Pairs range from 0.75 to 1.0
            
            # Suited hands - smaller bonus than standard poker
            is_suited = suit1 == suit2
            suited_bonus = 0.08 if is_suited else 0
            
            # Handle Ace combinations specially
            if rank1 == 8 or rank2 == 8:  # One card is Ace
                non_ace_rank = rank2 if rank1 == 8 else rank1
                
                # A-9 is excellent
                if non_ace_rank == 7:  # A-9
                    ace_value = 0.8 + suited_bonus
                # A-5 for straights
                elif non_ace_rank == 3:  # A-5
                    ace_value = 0.65 + suited_bonus
                # A-8
                elif non_ace_rank == 6:  # A-8
                    ace_value = 0.7 + suited_bonus
                # Other Ace combos
                else:
                    # Scale with rank
                    ace_value = 0.5 + (non_ace_rank / 14) + suited_bonus
                
                return ace_value
                
            # Connected cards have more value in this deck
            rank_diff = abs(rank1 - rank2)
            if rank_diff <= 4:  # Can make a straight
                connectivity = 1 - (rank_diff / 5)
            else:
                connectivity = 0.3
            
            # Higher cards are better
            high_card = max(rank1, rank2) / 8
            
            # Calculate base equity
            equity = 0.3 + (high_card * 0.3) + (connectivity * 0.3) + suited_bonus
            
            # Apply non-linear scaling to better differentiate hand strengths
            return min(0.95, max(0.1, equity))
        
        # Calculate values for all possible starting hands
        for i in range(27):
            for j in range(i+1, 27):
                equity = monte_carlo_equity(i, j)
                
                # Store value for both orderings
                starting_hands[(i, j)] = equity
                starting_hands[(j, i)] = equity
        
        return starting_hands
    
    def get_hand_strength(self, hole_cards, board=[], opponent_range=None):
        """
        Get hand strength (0-1) specialized for 27-card deck
        """
        # Create cache key
        cache_key = (tuple(sorted(hole_cards)), tuple(sorted(board)))
        
        if cache_key in self.hand_strength_cache:
            return self.hand_strength_cache[cache_key]
        
        # Pre-flop - use pre-computed values
        if not board or all(c == -1 for c in board):
            strength = self.starting_hand_values.get(tuple(hole_cards), 0.3)
            self.hand_strength_cache[cache_key] = strength
            return strength
        
        # Filter out -1 cards
        effective_board = [c for c in board if c != -1]
        
        # Partial board - not enough for a full 5-card hand
        if len(hole_cards) + len(effective_board) < 5:
            partial_strength = self._estimate_partial_strength(hole_cards, effective_board)
            self.hand_strength_cache[cache_key] = partial_strength
            return partial_strength
        
        # Complete hand evaluation
        made_hand_rank = self._evaluate_hand_rank(hole_cards, effective_board)
        
        # Convert rank to 0-1 scale
        # In this 27-card variant, hand ranks are:
        # 8: Straight Flush, 7: Full House, 6: Flush, 5: Straight, 
        # 4: Three of a Kind, 3: Two Pair, 2: One Pair, 1: High Card
        normalized_strength = (made_hand_rank - 1) / 7
        
        # On flop, consider potential for improvement
        if len(effective_board) == 3:
            draw_equity = self._calculate_draw_potential(hole_cards, effective_board)
            # Blend made hand with draw potential
            blended_strength = 0.7 * normalized_strength + 0.3 * draw_equity
            self.hand_strength_cache[cache_key] = blended_strength
            return blended_strength
        
        # On turn, less weight to draws
        elif len(effective_board) == 4:
            draw_equity = self._calculate_draw_potential(hole_cards, effective_board)
            blended_strength = 0.85 * normalized_strength + 0.15 * draw_equity
            self.hand_strength_cache[cache_key] = blended_strength
            return blended_strength
        
        # On river - only made hand matters
        self.hand_strength_cache[cache_key] = normalized_strength
        return normalized_strength
    
    def _evaluate_hand_rank(self, hole_cards, board):
        """
        Evaluate complete poker hand and return a hand rank (1-8)
        Specialized for 27-card deck where four of a kind is impossible
        """
        all_cards = hole_cards + board
        
        # Extract ranks and suits
        ranks = [card // 3 for card in all_cards]
        suits = [card % 3 for card in all_cards]
        
        # Count frequencies
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        
        suit_counts = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        
        # Check for flush
        has_flush = False
        flush_suit = None
        flush_ranks = []
        
        for suit, count in suit_counts.items():
            if count >= 5:
                has_flush = True
                flush_suit = suit
                # Collect ranks of flush cards
                flush_ranks = [ranks[i] for i, s in enumerate(suits) if s == flush_suit]
                break
        
        # Check for straight - handle Ace as both high and low
        # Clone ranks and handle Ace specially
        straight_ranks = ranks.copy()
        if 8 in ranks:  # If we have an Ace
            straight_ranks.append(-1)  # Add low Ace (below 2)
        
        straight_ranks = sorted(set(straight_ranks))
        
        has_straight = False
        straight_high = None
        
        # Look for 5 consecutive ranks
        for i in range(len(straight_ranks) - 4):
            if (straight_ranks[i+4] - straight_ranks[i]) == 4:
                has_straight = True
                straight_high = straight_ranks[i+4]
        
        # Check for straight flush
        has_straight_flush = False
        if has_flush and has_straight:
            # Check if the straight cards are all the same suit
            flush_ranks_set = set(flush_ranks)
            straight_flush_ranks = [r for r in range(straight_high-4, straight_high+1) 
                                   if r in flush_ranks_set]
            
            # Special case for A-5 straight
            if -1 in straight_ranks and 3 in flush_ranks_set:
                low_straight = [r for r in [-1, 0, 1, 2, 3] if r in flush_ranks_set or 
                              (r == -1 and 8 in flush_ranks_set)]
                if len(low_straight) >= 5:
                    has_straight_flush = True
                    straight_high = 3  # 5-high straight
            
            has_straight_flush = len(straight_flush_ranks) >= 5
        
        # Evaluate hand type (from highest to lowest)
        if has_straight_flush:
            return 8  # Straight Flush
        
        # Check for Full House (three of a kind + pair)
        three_of_a_kind = [r for r, count in rank_counts.items() if count >= 3]
        pairs = [r for r, count in rank_counts.items() if count >= 2]
        
        if three_of_a_kind and len(pairs) >= 2:
            return 7  # Full House
        
        if has_flush:
            return 6  # Flush
        
        if has_straight:
            return 5  # Straight
        
        if three_of_a_kind:
            return 4  # Three of a Kind
        
        if len(pairs) >= 2:
            return 3  # Two Pair
        
        if pairs:
            return 2  # One Pair
        
        return 1  # High Card
    
    def _estimate_partial_strength(self, hole_cards, board):
        """
        Estimate hand strength for partial boards
        """
        # Start with preflop hand strength as base
        base_strength = self.starting_hand_values.get(tuple(hole_cards), 0.3)
        
        if not board:
            return base_strength
            
        # Find best made hand so far
        all_cards = hole_cards + board
        
        # Extract ranks and suits
        ranks = [card // 3 for card in all_cards]
        suits = [card % 3 for card in all_cards]
        
        # Count frequencies
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        
        suit_counts = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        
        # Check for made hands and draws
        adjustment = 0
        
        # Pairs or better
        max_rank_count = max(rank_counts.values())
        if max_rank_count >= 3:
            # Trips
            adjustment += 0.35
        elif max_rank_count == 2:
            # Pair - higher pairs are better
            pair_rank = [r for r, count in rank_counts.items() if count == 2][0]
            pair_value = pair_rank / 8  # 0-1 scale
            adjustment += 0.2 + pair_value * 0.15
            
            # Two pair
            if list(rank_counts.values()).count(2) >= 2:
                adjustment += 0.15
        
        # Flush potential
        max_suit_count = max(suit_counts.values())
        if max_suit_count >= 4:
            # Near flush
            adjustment += 0.25
        elif max_suit_count == 3:
            adjustment += 0.15
        
        # Straight potential - count the gaps
        sorted_ranks = sorted(set(ranks))
        if 8 in sorted_ranks:  # Add low Ace if we have an Ace
            sorted_ranks.append(-1)
            sorted_ranks.sort()
        
        max_connected = 1
        current_connected = 1
        for i in range(1, len(sorted_ranks)):
            if sorted_ranks[i] == sorted_ranks[i-1] + 1:
                current_connected += 1
                max_connected = max(max_connected, current_connected)
            else:
                current_connected = 1
        
        if max_connected >= 4:
            # One card away from straight
            adjustment += 0.25
        elif max_connected == 3:
            adjustment += 0.1
        
        # High card value
        high_rank = max(ranks)
        high_card_value = high_rank / 8  # 0-1 scale
        adjustment += high_card_value * 0.1
        
        # Combine base strength with adjustments
        adjusted_strength = base_strength + adjustment
        
        # Ensure result is in valid range
        return min(1.0, max(0.0, adjusted_strength))
    
    def _calculate_draw_potential(self, hole_cards, board):
        """
        Calculate potential improvement from draws
        Specialized for the 27-card deck
        """
        all_cards = hole_cards + board
        
        # Extract ranks and suits
        ranks = [card // 3 for card in all_cards]
        suits = [card % 3 for card in all_cards]
        
        # Calculate draw potential
        draw_value = 0.0
        
        # Check for flush draw
        suit_counts = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        
        # 4 cards of the same suit is a flush draw
        if max(suit_counts.values()) == 4:
            draw_value += 0.25
        
        # Check for straight draw
        sorted_ranks = sorted(set(ranks))
        if 8 in sorted_ranks:  # Add low Ace if we have an Ace
            sorted_ranks.append(-1)
            sorted_ranks.sort()
        
        # Open-ended straight draw (4 consecutive ranks)
        for i in range(len(sorted_ranks) - 3):
            if (sorted_ranks[i+3] - sorted_ranks[i]) == 3:
                draw_value += 0.20
                break
        
        # Gutshot straight draw (4 ranks with one gap)
        has_gutshot = False
        for i in range(len(sorted_ranks) - 3):
            if (sorted_ranks[i+3] - sorted_ranks[i]) == 4:
                has_gutshot = True
                break
        
        if has_gutshot:
            draw_value += 0.10
        
        # Two pair with potential for full house
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        
        if list(rank_counts.values()).count(2) >= 2:
            draw_value += 0.15
        
        # One pair with potential for trips or full house
        elif 2 in rank_counts.values():
            draw_value += 0.10
        
        # Scale the draw value
        return min(0.7, draw_value)  # Cap at 0.7

# Enhanced Opponent Model for 27-card deck poker

class OpponentModel:
    """
    Advanced opponent modeling for poker with 27-card deck.
    Uses Bayesian inference and pattern recognition to exploit opponent tendencies.
    """
    
    def __init__(self):
        # Core tracking statistics with optimized priors for this variant
        self.aggression = 0.5  # Scale 0-1, 0.5 is baseline
        self.fold_frequency = 0.5  # How often they fold to bets
        self.bluff_frequency = 0.3  # Estimated bluffing frequency
        
        # Redraw statistics
        self.redraw_count = 0
        self.hands_seen = 0
        self.redraw_history = []  # List of (discarded_card, drawn_card, street)
        
        # Bayesian confidence tracking
        self.aggression_confidence = 1.0
        self.fold_confidence = 1.0
        self.bluff_confidence = 1.0
        
        # Street-specific patterns with recency bias
        self.actions_by_street = {
            0: [],  # Preflop
            1: [],  # Flop
            2: [],  # Turn
            3: []   # River
        }
        
        # Advanced pattern detection
        self.continuation_bet_freq = 0.5  # How often they c-bet
        self.check_raise_freq = 0.1  # How often they check-raise
        self.bet_sizing_by_street = {
            0: [],  # Preflop bet sizes
            1: [],  # Flop bet sizes
            2: [],  # Turn bet sizes
            3: []   # River bet sizes
        }
        
        # Showdown tracking
        self.showdown_history = []
        self.bluffs_detected = 0
        self.value_bets_detected = 0
        
        # Pattern recognition
        self.action_sequences = {}
        
        # Learning rate with decay
        self.base_learning_rate = 0.2
        self.learning_rate = 0.2
        
        # Positional tendencies
        self.position_aggression = {
            'SB': 0.5,
            'BB': 0.5
        }
        
        # Redraw tendencies - specialized for this variant
        self.redraw_patterns = {
            'high_card_discard': 0,
            'low_card_discard': 0,
            'suited_preference': 0,
            'ace_behavior': 'unknown'  # 'keeps', 'discards', or 'unknown'
        }
    
    def update_from_action(self, action, street, bet_size=None, was_all_in=False):
        """
        Update model based on opponent's action using Bayesian updating
        """
        # Record action with recency bias
        self.actions_by_street[street].append(action)
        
        # Keep only recent actions
        max_actions = 50
        if len(self.actions_by_street[street]) > max_actions:
            self.actions_by_street[street] = self.actions_by_street[street][-max_actions:]
        
        # Update aggression metric
        if action == action_types.RAISE.value:
            # Bayesian update
            prior = self.aggression
            likelihood = 0.9  # High likelihood of aggressive player raising
            
            posterior = (likelihood * prior) / (likelihood * prior + (1 - likelihood) * (1 - prior))
            
            # Apply with learning rate and recency bias
            self.aggression = (1 - self.learning_rate) * self.aggression + self.learning_rate * posterior
            
            # Update position-specific aggression
            position = 'SB' if street % 2 == 0 else 'BB'  # Rough position estimation
            self.position_aggression[position] = (
                0.9 * self.position_aggression[position] + 0.1 * posterior
            )
            
            # Track bet sizing
            if bet_size is not None:
                self.bet_sizing_by_street[street].append(bet_size)
                # Keep only recent bet sizes
                if len(self.bet_sizing_by_street[street]) > max_actions:
                    self.bet_sizing_by_street[street] = self.bet_sizing_by_street[street][-max_actions:]
        
        elif action == action_types.CALL.value:
            self.aggression = (1 - self.learning_rate) * self.aggression + self.learning_rate * 0.5
        
        elif action == action_types.CHECK.value:
            self.aggression = (1 - self.learning_rate) * self.aggression + self.learning_rate * 0.3
        
        elif action == action_types.FOLD.value:
            self.aggression = (1 - self.learning_rate) * self.aggression + self.learning_rate * 0.2
            
            # Update fold frequency
            prior = self.fold_frequency
            likelihood = 0.9
            posterior = (likelihood * prior) / (likelihood * prior + (1 - likelihood) * (1 - prior))
            self.fold_frequency = (1 - self.learning_rate) * self.fold_frequency + self.learning_rate * posterior
        
        # Update confidence
        self.aggression_confidence = min(10.0, self.aggression_confidence + 0.05)
        self.fold_confidence = min(10.0, self.fold_confidence + 0.05)
        
        # Track action sequences for pattern detection
        all_actions = []
        for s in range(4):
            all_actions.extend(self.actions_by_street[s][-3:])
        
        if len(all_actions) >= 3:
            seq = tuple(all_actions[-3:])
            self.action_sequences[seq] = self.action_sequences.get(seq, 0) + 1
    
    def update_from_redraw(self, discarded_card, drawn_card, street):
        """
        Update model based on opponent's redraw action
        """
        self.redraw_count += 1
        self.redraw_history.append((discarded_card, drawn_card, street))
        
        # Advanced analysis of discarded cards
        discarded_rank = discarded_card // 3
        discarded_suit = discarded_card % 3
        
        # Track tendencies
        if discarded_rank >= 6:  # 8, 9, A
            self.redraw_patterns['high_card_discard'] += 1
            
            # Track Ace discard behavior specifically
            if discarded_rank == 8:  # Ace
                self.redraw_patterns['ace_behavior'] = 'discards'
        else:
            self.redraw_patterns['low_card_discard'] += 1
        
        # Look for suited preferences
        if len(self.redraw_history) >= 2:
            # Check if they consistently keep cards of a particular suit
            kept_suits = []
            for discard, _, _ in self.redraw_history:
                # For each discard, find the kept card's suit
                kept_suit = None
                for suit in range(3):
                    for rank in range(9):
                        card = rank * 3 + suit
                        if card != discard and card != drawn_card:
                            kept_suit = suit
                            break
                if kept_suit is not None:
                    kept_suits.append(kept_suit)
            
            # Check if there's a pattern in kept suits
            if kept_suits:
                suit_counts = {}
                for s in kept_suits:
                    suit_counts[s] = suit_counts.get(s, 0) + 1
                
                # If one suit dominates their keeps
                max_suit = max(suit_counts, key=suit_counts.get)
                if suit_counts[max_suit] / len(kept_suits) > 0.7:
                    self.redraw_patterns['suited_preference'] = max_suit
        
        # Update redraw frequency model
        if hasattr(self, 'redraw_frequency_by_street'):
            self.redraw_frequency_by_street[street] = (
                self.redraw_frequency_by_street[street][0] + 1,
                self.redraw_frequency_by_street[street][1] + 1
            )
        else:
            self.redraw_frequency_by_street = {
                0: (1, 1),  # (redraws, hands seen)
                1: (0, 0)
            }
            if street == 1:
                self.redraw_frequency_by_street[1] = (1, 1)
    
    def update_from_hand(self, hand_info):
        """
        Update overall stats at the end of a hand
        """
        self.hands_seen += 1
        
        # Adjust learning rate as we gather more data
        # Start aggressive, become more stable over time
        if self.hands_seen <= 10:
            self.learning_rate = self.base_learning_rate  # Fast learning at start
        elif self.hands_seen <= 50:
            self.learning_rate = self.base_learning_rate * 0.8  # Moderate
        else:
            self.learning_rate = self.base_learning_rate * 0.6  # Slower, more stable
    
    def get_redraw_frequency(self):
        """
        Get opponent's redraw frequency with confidence weighting
        """
        if self.hands_seen == 0:
            return 0.5  # Default assumption
        
        # Basic frequency
        basic_freq = self.redraw_count / self.hands_seen
        
        # Street-specific data if available
        if hasattr(self, 'redraw_frequency_by_street'):
            preflop_freq = self.redraw_frequency_by_street[0][0] / max(1, self.redraw_frequency_by_street[0][1])
            flop_freq = self.redraw_frequency_by_street[1][0] / max(1, self.redraw_frequency_by_street[1][1])
            
            # Weight based on confidence
            preflop_confidence = min(1.0, self.redraw_frequency_by_street[0][1] / 10.0)
            flop_confidence = min(1.0, self.redraw_frequency_by_street[1][1] / 10.0)
            
            # Combined weighted frequency
            weighted_freq = (
                preflop_freq * preflop_confidence + 
                flop_freq * flop_confidence + 
                0.5 * (2 - preflop_confidence - flop_confidence)
            ) / 2.0
            
            # Blend with basic frequency
            return 0.7 * weighted_freq + 0.3 * basic_freq
            
        return basic_freq
    
    def get_aggression_factor(self):
        """
        Get opponent's aggression factor with positional weighting
        """
        # Base aggression
        base_aggression = self.aggression
        
        # Apply positional weighting
        positional_aggression = 0.7 * self.position_aggression['SB'] + 0.3 * self.position_aggression['BB']
        
        # Calculate confidence-weighted result
        confidence_weight = min(1.0, self.aggression_confidence / 5.0)
        
        return (base_aggression * 0.7 + positional_aggression * 0.3) * confidence_weight + 0.5 * (1 - confidence_weight)
    
    def get_fold_equity(self):
        """
        Calculate fold equity against this opponent
        """
        # Higher fold frequency means more fold equity
        base_fold_equity = self.fold_frequency
        
        # Adjust based on aggression
        aggression_adjustment = -0.2 * (self.aggression - 0.5)
        
        # Adjust based on recent tendencies
        recent_actions = []
        for street in range(4):
            actions = self.actions_by_street[street][-10:] if self.actions_by_street[street] else []
            recent_actions.extend(actions)
        
        if recent_actions:
            recent_folds = recent_actions.count(action_types.FOLD.value)
            recent_fold_frequency = recent_folds / len(recent_actions)
            recency_adjustment = 0.3 * (recent_fold_frequency - self.fold_frequency)
        else:
            recency_adjustment = 0
        
        # Confidence-based blending with default
        confidence_weight = min(1.0, self.fold_confidence / 5.0)
        
        # Calculate final fold equity
        fold_equity = base_fold_equity + aggression_adjustment + recency_adjustment
        
        # Blend with default
        default_equity = 0.5  # Average fold equity
        
        return fold_equity * confidence_weight + default_equity * (1 - confidence_weight)
    
    def get_bluff_frequency(self):
        """
        Get opponent's estimated bluff frequency
        """
        confidence_weight = min(1.0, self.bluff_confidence / 5.0)
        
        # Calculate observed frequency
        total_aggressive_actions = self.bluffs_detected + self.value_bets_detected
        if total_aggressive_actions > 10:
            observed_frequency = self.bluffs_detected / total_aggressive_actions
        else:
            # Not enough data, rely more on prior
            observed_frequency = 0.3  # Average bluff frequency
            confidence_weight *= total_aggressive_actions / 10.0
        
        # Blend with prior based on confidence
        return (observed_frequency * confidence_weight) + (0.3 * (1 - confidence_weight))
    
    def adjust_hand_strength(self, base_strength, street, is_bluff_candidate=False):
        """
        Adjust effective hand strength based on opponent tendencies
        """
        # Start with base strength
        adjusted_strength = base_strength
        
        # Get opponent tendencies
        aggression = self.get_aggression_factor()
        fold_equity = self.get_fold_equity()
        bluff_frequency = self.get_bluff_frequency()
        
        # Against aggressive opponents, we need stronger hands
        if aggression > 0.7:
            adjusted_strength -= 0.05
        
        # Against passive opponents, we can play more marginal hands
        elif aggression < 0.3:
            adjusted_strength += 0.05
        
        # If opponent folds often and we're considering a bluff, increase effective strength
        if is_bluff_candidate and fold_equity > 0.6:
            adjusted_strength += 0.15
        
        # If opponent rarely folds, bluffing is less effective
        elif is_bluff_candidate and fold_equity < 0.3:
            adjusted_strength -= 0.1
        
        # If opponent bluffs often, we should call more with medium-strength hands
        if bluff_frequency > 0.4 and street >= 2 and 0.3 < base_strength < 0.7:
            adjusted_strength += 0.1
        
        return min(1.0, max(0.0, adjusted_strength))

# Enhanced Time Manager for optimal tournament performance

class TimeManager:
    """
    Advanced time management for poker tournaments.
    Optimizes time allocation across a match using dynamic programming concepts.
    """
    
    def __init__(self):
        # Time tracking
        self.decision_start_time = None
        self.total_time_used = 0
        self.decision_times = []
        self.street_decision_times = {
            0: [],  # Preflop
            1: [],  # Flop
            2: [],  # Turn
            3: []   # River
        }
        
        # Tournament settings
        self.tournament_phase = 1
        self.total_time_bank = 500  # 500 seconds default
        
        # Dynamic time allocation
        self.time_allocation = {
            'preflop': 0.2,
            'flop': 0.3,
            'turn': 0.25,
            'river': 0.25
        }
        
        # Importance multipliers
        self.importance_multiplier = {
            'low': 0.5,
            'normal': 1.0,
            'high': 2.0,
            'critical': 3.0
        }
        
        # Hand tracking
        self.hands_played = 0
        self.total_hands = 1000
        self.hands_remaining_prediction = 1000
        
        # Reserved time
        self.reserved_time_fraction = 0.15  # Reduced from 0.2
        self.reserved_time = self.total_time_bank * self.reserved_time_fraction
        
        # Average time tracking
        self.avg_time_per_hand = 0.9 * self.total_time_bank / self.total_hands
    
    def start_decision(self):
        """Mark the start time of a decision."""
        self.decision_start_time = time.time()
    
    def end_decision(self):
        """
        Mark the end of a decision and update time tracking.
        
        Returns:
            Elapsed time for this decision
        """
        if self.decision_start_time is None:
            return 0
            
        elapsed = time.time() - self.decision_start_time
        self.total_time_used += elapsed
        self.decision_times.append(elapsed)
        
        # Keep only recent decisions
        if len(self.decision_times) > 100:
            self.decision_times.pop(0)
        
        return elapsed
    
    def get_time_remaining(self):
        """
        Calculate remaining time in the bank.
        """
        return max(0, self.total_time_bank - self.total_time_used)
    
    def is_time_critical(self):
        """
        Check if we're running critically low on time.
        
        Returns:
            Boolean indicating if time usage should be restricted
        """
        time_remaining = self.get_time_remaining()
        hands_remaining = max(1, self.total_hands - self.hands_played)
        
        # If we have less than 10% of our optimal time remaining
        return time_remaining < 0.1 * self.avg_time_per_hand * hands_remaining
    
    def emergency_time_restriction(self):
        """
        Get an emergency time restriction when nearly out of time.
        
        Returns:
            Maximum allowed time for next decision
        """
        time_remaining = self.get_time_remaining()
        hands_remaining = max(1, self.total_hands - self.hands_played)
        
        # Ultra conservative time allocation
        return max(0.001, time_remaining / (2 * hands_remaining))

# Main integration - enhanced for 27-card deck poker

if __name__ == "__main__":
    # Initialize the upgraded bot
    bot = PlayerAgent()
    
    # This can be run directly for testing
    print("Elite Poker Bot initialized and ready to play")
    print("Specialized for 27-card deck with optimal strategy")