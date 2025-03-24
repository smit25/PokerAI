from agents.agent import Agent
from gym_env import PokerEnv
import random

# from submission.hand_evaluator import HandEvaluator
# from submission.redraw import Redraw
# from submission.opponent_model import OpponentModel
# from submission.decision_engine import DecisionEngine
# from submission.time_manager import TimeManager


# action_types = PokerEnv.ActionType


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


# class PlayerAgent(Agent):
#     def __name__(self):
#         return "PlayerAgent"

#     def __init__(self, stream: bool = True):
#         super().__init__(stream)
#         # Initialize any instance variables here
#         self.hand_number = 0
#         self.won_hands = 0
#         self.time_manager = TimeManager()
        
#         self.hand_evaluator = HandEvaluator()
#         self.opponent_model = OpponentModel()
#         self.redraw_strategy = Redraw(self.hand_evaluator)
#         self.decision_engine = DecisionEngine(
#             self.hand_evaluator,
#             self.redraw_strategy,
#             self.opponent_model,
#             self.time_manager
#         )
        
#         self.prev_obs = None
#         self.hands_played = 0
#         self.hand_history = []

#     def act_default(self, observation, reward, terminated, truncated, info):
#         # Example of using the logger
#         if observation["street"] == 0 and info["hand_number"] % 50 == 0:
#             self.logger.info(f"Hand number: {info['hand_number']}")

#         # First, get the list of valid actions we can take
#         valid_actions = observation["valid_actions"]
        
#         # Get indices of valid actions (where value is 1)
#         valid_action_indices = [i for i, is_valid in enumerate(valid_actions) if is_valid]
        
#         # Randomly choose one of the valid action indices
#         action_type = random.choice(valid_action_indices)
        
#         # Set up our response values
#         raise_amount = 0
#         card_to_discard = -1  # -1 means no discard
        
#         # If we chose to raise, pick a random amount between min and max
#         if action_type == action_types.RAISE.value:
#             if observation["min_raise"] == observation["max_raise"]:
#                 raise_amount = observation["min_raise"]
#             else:
#                 raise_amount = random.randint(
#                     observation["min_raise"],
#                     observation["max_raise"]
#                 )
        
#         # If we chose to discard, randomly pick one of our two cards (0 or 1)
#         if action_type == action_types.DISCARD.value:
#             card_to_discard = random.randint(0, 1)
        
#         return action_type, raise_amount, card_to_discard
    
#     def act(self, obs, reward, terminated, truncated, info):
#         if self._is_new_hand(obs):
#             self.hands_played += 1
        
#         # Update opponent model if we have a previous observation
#         if self.prev_obs is not None:
#             self._update_opponent_model(self.prev_obs, obs)
        
#         if self.time_manager.is_time_critical():
#             action = self._emergency_strategy(obs)
#         else:
#             action = self.decision_engine.make_decision(obs)
        
#         # Store observation for next update
#         self.prev_obs = obs.copy()
        
#         return action
    
#     def _is_new_hand(self, obs):
#         """
#         Detect if this observation represents the start of a new hand.
#         Returns: Boolean indicating if this is a new hand
#         """
#         # New hand indicators:
#         # - Street is 0 (preflop)
#         # - Small blind or big blind has been posted
#         # - Previous observation was None or from a different hand
        
#         is_preflop = obs["street"] == 0
#         has_blinds = obs["my_bet"] <= 2 and obs["opp_bet"] <= 2
        
#         if is_preflop and has_blinds:
#             if self.prev_obs is None:
#                 return True
                
#             # Check if we've advanced from a previous hand
#             prev_was_river = self.prev_obs.get("street", 0) == 3
#             return prev_was_river
            
#         return False
    
#     def _update_opponent_model(self, prev_obs, curr_obs):
#         """
#         Update opponent model based on observed changes.
#         """
        
#         opp_discarded_card = curr_obs.get("opp_discarded_card", -1)
#         opp_drawn_card = curr_obs.get("opp_drawn_card", -1)
        
#         if opp_discarded_card != -1 and opp_drawn_card != -1:
#             self.opponent_model.update_from_redraw(
#                 opp_discarded_card, opp_drawn_card, curr_obs["street"])
        
#         # Detect betting actions
#         # If it was our turn in prev_obs, and now it's our turn again,
#         # the opponent must have taken an action
#         if prev_obs.get("acting_agent") == curr_obs.get("acting_agent"):
#             street = curr_obs["street"]
#             prev_opp_bet = prev_obs.get("opp_bet", 0)
#             curr_opp_bet = curr_obs.get("opp_bet", 0)
            
#             if curr_opp_bet > prev_opp_bet:
#                 # Opponent raised
#                 self.opponent_model.update_from_action(
#                     action_types.RAISE.value, street, bet_size=curr_opp_bet-prev_opp_bet)
#             elif curr_opp_bet == prev_opp_bet and prev_obs["street"] == curr_obs["street"]:
#                 # Opponent checked
#                 self.opponent_model.update_from_action(action_types.CHECK.value, street)
#             elif curr_opp_bet == prev_obs.get("my_bet", 0) and curr_opp_bet > prev_opp_bet:
#                 # Opponent called
#                 self.opponent_model.update_from_action(action_types.CALL.value, street)
        
#         # Detect showdown
#         if prev_obs.get("street") == 3 and curr_obs.get("street") == 0:
#             # End of hand - in a real implementation, we'd have access to
#             # showdown information like opponent's cards
#             # TODO
#             # For now, we just update the hand count
#             self.opponent_model.update_from_hand({})
    
#     def _emergency_strategy(self, obs):
#         """
#         Simplified strategy for emergency time situations.
#         Makes quick decisions to avoid timeouts.
#         """
#         my_cards = obs["my_cards"]
#         community_cards = obs.get("community_cards", [])
#         street = obs["street"]
#         acting_agent = obs["acting_agent"]
#         my_bet = obs["my_bet"]
#         opp_bet = obs["opp_bet"]
#         min_raise = obs.get("min_raise", 1)
#         max_raise = obs.get("max_raise", 100)
#         valid_actions = obs["valid_actions"]
        
#         # For preflop, use a simple lookup based on card ranks
#         if street == 0:
#             # Convert cards to ranks for quick evaluation
#             card1_rank = my_cards[0] // 3
#             card2_rank = my_cards[1] // 3
            
#             # Pair
#             if card1_rank == card2_rank:
#                 strength = 0.8 if card1_rank >= 7 else 0.6  # High pair vs low pair
#             # High cards
#             elif card1_rank >= 7 and card2_rank >= 7:
#                 strength = 0.7  # Two high cards
#             # One high card
#             elif card1_rank >= 7 or card2_rank >= 7:
#                 strength = 0.5  # One high card
#             # Connected cards (approximately)
#             elif abs(card1_rank - card2_rank) <= 2:
#                 strength = 0.4  # Connected low cards
#             else:
#                 strength = 0.2  # Unconnected low cards
        
#         # Post-flop - very simple made hand check
#         else:
#             all_cards = my_cards + community_cards
            
#             rank_counts = {}
#             suit_counts = {}
            
#             for card in all_cards:
#                 rank = card // 3
#                 suit = card % 3
                
#                 rank_counts[rank] = rank_counts.get(rank, 0) + 1
#                 suit_counts[suit] = suit_counts.get(suit, 0) + 1
            
#             if max(rank_counts.values()) >= 3:  # Three of a kind or better
#                 strength = 0.8
#             elif list(rank_counts.values()).count(2) >= 2:  # Two pair
#                 strength = 0.7
#             elif max(rank_counts.values()) >= 2:  # One pair
#                 strength = 0.5
#             elif max(suit_counts.values()) >= 5:  # Flush draw or better
#                 strength = 0.6
#             else:
#                 # High card - rough estimation
#                 high_ranks = [r for r, c in rank_counts.items() if r >= 7]
#                 strength = 0.3 if high_ranks else 0.1
        
#         # Check if we should use our redraw
#         if valid_actions[action_types.DISCARD.value] and street <= 1:
#             # Only redraw very weak hands
#             if strength < 0.3:
#                 # Determine which card to discard (the lower one)
#                 card1_rank = my_cards[0] // 3
#                 card2_rank = my_cards[1] // 3
#                 discard_idx = 0 if card1_rank < card2_rank else 1
                
#                 return (action_types.DISCARD.value, 0, discard_idx)
        
#         bet_to_call = opp_bet - my_bet
        
#         # Nothing to call, we can check
#         if bet_to_call == 0:
#             # With strong hand, bet
#             if strength > 0.6 and valid_actions[action_types.RAISE.value]:
#                 bet_size = min(max_raise, max(min_raise, 2))  # Minimum bet
#                 return (action_types.RAISE.value, bet_size, -1)
#             # Otherwise check
#             else:
#                 return (action_types.CHECK.value, 0, -1)
        
#         else:
#             # Strong hand - call or raise
#             if strength > 0.7:
#                 if valid_actions[action_types.RAISE.value]:
#                     raise_size = min(max_raise, max(min_raise, bet_to_call * 2))
#                     return (action_types.RAISE.value, raise_size, -1)
#                 else:
#                     return (action_types.CALL.value, 0, -1)
            
#             # Medium hand - call small bets
#             elif strength > 0.4 and bet_to_call < 10:
#                 return (action_types.CALL.value, 0, -1)
            
#             # Weak hand - fold
#             else:
#                 return (action_types.FOLD.value, 0, -1)
            

#     def observe(self, observation, reward, terminated, truncated, info):
#         # Log interesting events when observing opponent's actions
#         pass
#         if terminated:
#             self.logger.info(f"Game ended with reward: {reward}")
#             self.hand_number += 1
#             if reward > 0:
#                 self.won_hands += 1
#             self.last_action = None
#         else:
#             # log observation keys
#             self.logger.info(f"Observation keys: {observation}")