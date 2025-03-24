from agents.agent import Agent
from gym_env import PokerEnv
import random

from submission.hand_evaluator import HandEvaluator
from submission.redraw import Redraw
from submission.opponent_model import OpponentModel
from submission.decision_engine import DecisionEngine
from submission.time_manager import TimeManager

action_types = PokerEnv.ActionType

class PlayerAgent(Agent):
    def __name__(self):
        return "PlayerAgent"

    def __init__(self, stream: bool = True):
        super().__init__(stream)
        # Initialize any instance variables here
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
        
        self.prev_obs = None
        self.hands_played = 0
        self.hand_history = []

    def act_default(self, observation, reward, terminated, truncated, info):
        # Example of using the logger
        if observation["street"] == 0 and info["hand_number"] % 50 == 0:
            self.logger.info(f"Hand number: {info['hand_number']}")

        # First, get the list of valid actions we can take
        valid_actions = observation["valid_actions"]
        
        # Get indices of valid actions (where value is 1)
        valid_action_indices = [i for i, is_valid in enumerate(valid_actions) if is_valid]
        
        # Randomly choose one of the valid action indices
        action_type = random.choice(valid_action_indices)
        
        # Set up our response values
        raise_amount = 0
        card_to_discard = -1  # -1 means no discard
        
        # If we chose to raise, pick a random amount between min and max
        if action_type == action_types.RAISE.value:
            if observation["min_raise"] == observation["max_raise"]:
                raise_amount = observation["min_raise"]
            else:
                raise_amount = random.randint(
                    observation["min_raise"],
                    observation["max_raise"]
                )
        
        # If we chose to discard, randomly pick one of our two cards (0 or 1)
        if action_type == action_types.DISCARD.value:
            card_to_discard = random.randint(0, 1)
        
        return action_type, raise_amount, card_to_discard
    
    def act(self, obs, reward, terminated, truncated, info):
        if self._is_new_hand(obs):
            self.hands_played += 1
        
        # Update opponent model if we have a previous observation
        if self.prev_obs is not None:
            self._update_opponent_model(self.prev_obs, obs)
        
        if self.time_manager.is_time_critical():
            action = self._emergency_strategy(obs)
        else:
            action = self.decision_engine.make_decision(obs)
        
        # Store observation for next update
        self.prev_obs = obs.copy()
        
        return action
    
    def _is_new_hand(self, obs):
        """
        Detect if this observation represents the start of a new hand.
        Returns: Boolean indicating if this is a new hand
        """
        # New hand indicators:
        # - Street is 0 (preflop)
        # - Small blind or big blind has been posted
        # - Previous observation was None or from a different hand
        
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
        Update opponent model based on observed changes.
        """
        
        opp_discarded_card = curr_obs.get("opp_discarded_card", -1)
        opp_drawn_card = curr_obs.get("opp_drawn_card", -1)
        
        if opp_discarded_card != -1 and opp_drawn_card != -1:
            self.opponent_model.update_from_redraw(
                opp_discarded_card, opp_drawn_card, curr_obs["street"])
        
        # Detect betting actions
        # If it was our turn in prev_obs, and now it's our turn again,
        # the opponent must have taken an action
        if prev_obs.get("acting_agent") == curr_obs.get("acting_agent"):
            street = curr_obs["street"]
            prev_opp_bet = prev_obs.get("opp_bet", 0)
            curr_opp_bet = curr_obs.get("opp_bet", 0)
            
            if curr_opp_bet > prev_opp_bet:
                # Opponent raised
                self.opponent_model.update_from_action(
                    action_types.RAISE.value, street, bet_size=curr_opp_bet-prev_opp_bet)
            elif curr_opp_bet == prev_opp_bet and prev_obs["street"] == curr_obs["street"]:
                # Opponent checked
                self.opponent_model.update_from_action(action_types.CHECK.value, street)
            elif curr_opp_bet == prev_obs.get("my_bet", 0) and curr_opp_bet > prev_opp_bet:
                # Opponent called
                self.opponent_model.update_from_action(action_types.CALL.value, street)
        
        # Detect showdown
        if prev_obs.get("street") == 3 and curr_obs.get("street") == 0:
            # End of hand - in a real implementation, we'd have access to
            # showdown information like opponent's cards
            # TODO
            # For now, we just update the hand count
            self.opponent_model.update_from_hand({})
    
    def _emergency_strategy(self, obs):
        """
        Simplified strategy for emergency time situations.
        Makes quick decisions to avoid timeouts.
        """
        my_cards = obs["my_cards"]
        community_cards = obs.get("community_cards", [])
        street = obs["street"]
        acting_agent = obs["acting_agent"]
        my_bet = obs["my_bet"]
        opp_bet = obs["opp_bet"]
        min_raise = obs.get("min_raise", 1)
        max_raise = obs.get("max_raise", 100)
        valid_actions = obs["valid_actions"]
        
        # For preflop, use a simple lookup based on card ranks
        if street == 0:
            # Convert cards to ranks for quick evaluation
            card1_rank = my_cards[0] // 3
            card2_rank = my_cards[1] // 3
            
            # Pair
            if card1_rank == card2_rank:
                strength = 0.8 if card1_rank >= 7 else 0.6  # High pair vs low pair
            # High cards
            elif card1_rank >= 7 and card2_rank >= 7:
                strength = 0.7  # Two high cards
            # One high card
            elif card1_rank >= 7 or card2_rank >= 7:
                strength = 0.5  # One high card
            # Connected cards (approximately)
            elif abs(card1_rank - card2_rank) <= 2:
                strength = 0.4  # Connected low cards
            else:
                strength = 0.2  # Unconnected low cards
        
        # Post-flop - very simple made hand check
        else:
            all_cards = my_cards + community_cards
            
            rank_counts = {}
            suit_counts = {}
            
            for card in all_cards:
                rank = card // 3
                suit = card % 3
                
                rank_counts[rank] = rank_counts.get(rank, 0) + 1
                suit_counts[suit] = suit_counts.get(suit, 0) + 1
            
            if max(rank_counts.values()) >= 3:  # Three of a kind or better
                strength = 0.8
            elif list(rank_counts.values()).count(2) >= 2:  # Two pair
                strength = 0.7
            elif max(rank_counts.values()) >= 2:  # One pair
                strength = 0.5
            elif max(suit_counts.values()) >= 5:  # Flush draw or better
                strength = 0.6
            else:
                # High card - rough estimation
                high_ranks = [r for r, c in rank_counts.items() if r >= 7]
                strength = 0.3 if high_ranks else 0.1
        
        # Check if we should use our redraw
        if valid_actions[action_types.DISCARD.value] and street <= 1:
            # Only redraw very weak hands
            if strength < 0.3:
                # Determine which card to discard (the lower one)
                card1_rank = my_cards[0] // 3
                card2_rank = my_cards[1] // 3
                discard_idx = 0 if card1_rank < card2_rank else 1
                
                return (action_types.DISCARD.value, 0, discard_idx)
        
        bet_to_call = opp_bet - my_bet
        
        # Nothing to call, we can check
        if bet_to_call == 0:
            # With strong hand, bet
            if strength > 0.6 and valid_actions[action_types.RAISE.value]:
                bet_size = min(max_raise, max(min_raise, 2))  # Minimum bet
                return (action_types.RAISE.value, bet_size, -1)
            # Otherwise check
            else:
                return (action_types.CHECK.value, 0, -1)
        
        else:
            # Strong hand - call or raise
            if strength > 0.7:
                if valid_actions[action_types.RAISE.value]:
                    raise_size = min(max_raise, max(min_raise, bet_to_call * 2))
                    return (action_types.RAISE.value, raise_size, -1)
                else:
                    return (action_types.CALL.value, 0, -1)
            
            # Medium hand - call small bets
            elif strength > 0.4 and bet_to_call < 10:
                return (action_types.CALL.value, 0, -1)
            
            # Weak hand - fold
            else:
                return (action_types.FOLD.value, 0, -1)
            

    def observe(self, observation, reward, terminated, truncated, info):
        # Log interesting events when observing opponent's actions
        pass
        if terminated:
            self.logger.info(f"Game ended with reward: {reward}")
            self.hand_number += 1
            if reward > 0:
                self.won_hands += 1
            self.last_action = None
        else:
            # log observation keys
            self.logger.info(f"Observation keys: {observation}")