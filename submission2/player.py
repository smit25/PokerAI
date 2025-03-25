from gym_env import PokerEnv
from agents.agent import Agent

action_types = PokerEnv.ActionType

# Import components
from submission2.hand_evaluator import HandEvaluator
from submission2.opponent_model import OpponentModel
from submission2.redraw_strategy import RedrawStrategy
from submission2.decision_engine import DecisionEngine
from submission2.time_manager import TimeManager

class PlayerAgent(Agent):
    def __name__(self):
        return "PlayerAgent"

    def __init__(self, stream: bool = True):
        super().__init__(stream)
        # Initialize core components
        self.time_manager = TimeManager(max_time=0.95)  # 0.95s limit to stay safe
        self.hand_evaluator = HandEvaluator()
        self.opponent_model = OpponentModel()
        self.redraw_strategy = RedrawStrategy(self.hand_evaluator)
        self.decision_engine = DecisionEngine(
            self.hand_evaluator,
            self.redraw_strategy,
            self.opponent_model,
            self.time_manager
        )
        
        # Tracking variables
        self.hand_number = 0
        self.won_hands = 0
        self.prev_obs = None
        self.hands_played = 0
        self.hand_history = []
        
        # Debug settings
        self.debug = False

    def act(self, observation, reward, terminated, truncated, info):
        """Main action method called by the environment."""
        # Start timing
        self.time_manager.start_timing()
        
        # Track hands for logging
        if observation["street"] == 0 and self.prev_obs is None:
            self.hand_number = info["hand_number"] 
            self.hands_played += 1
            # Log every 50 hands
            if self.debug and self.hand_number % 50 == 0:
                self.logger.info(f"Hand number: {self.hand_number}")
        
        # Update opponent model if we have previous observation
        if self.prev_obs is not None:
            self._update_opponent_model(observation, self.prev_obs)
        
        # Get decision from decision engine
        action_type, raise_amount, card_to_discard = self.decision_engine.make_decision(observation)
        
        # Store current observation for next update
        self.prev_obs = observation.copy()
        
        # Check timing and log if we're cutting it close
        elapsed = self.time_manager.time_elapsed()
        if elapsed > 0.9:
            self.logger.warning(f"Decision took {elapsed:.3f}s - close to limit!")
        
        return action_type, raise_amount, card_to_discard
    
    def _update_opponent_model(self, current_obs, prev_obs):
        """Update opponent model with information between observations."""
        # Only update if we can determine opponent's action
        if "last_action" in current_obs and current_obs["last_action"] is not None:
            action = current_obs["last_action"]
            street = prev_obs["street"]  # Street where action occurred
            
            # Get betting information
            bet_size = 0
            if action == action_types.RAISE.value:
                # Calculate bet size based on change in opponent bet
                bet_size = current_obs["opp_bet"] - prev_obs["opp_bet"]
            
            # Calculate pot size at time of decision
            pot_size = prev_obs["my_bet"] + prev_obs["opp_bet"]
            
            # Special handling for redraw
            redraw_info = None
            if action == action_types.DISCARD.value:
                redraw_info = {
                    'street': street,
                    'discarded_card': current_obs.get("discarded_card", -1),
                    'drawn_card': current_obs.get("drawn_card", -1)
                }
            
            # Update model
            self.opponent_model.update(
                action=action,
                street=street,
                bet_size=bet_size,
                pot_size=pot_size,
                redraw_info=redraw_info
            )