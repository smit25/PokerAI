import random

class RedrawStrategy:
    """
    Optimized redraw strategy for 27-card deck poker.
    Specialized for a single-redraw poker variant.
    """
    
    def __init__(self, hand_evaluator):
        self.hand_evaluator = hand_evaluator
        
        # Threshold configurations
        self.redraw_thresholds = {
            0: 0.5,  # Preflop: redraw hands below 50% strength
            1: 0.4   # Flop: redraw hands below 40% strength (more conservative)
        }
        
        # Hand type flags for redraw decisions
        self.hand_types = {
            'pair': {'threshold': 0.7},       # Don't break pairs unless very weak
            'high_cards': {'threshold': 0.5},  # Keep high cards unless weak overall
            'suited': {'threshold': 0.55},     # Value suited cards more
            'connected': {'threshold': 0.55},  # Value connected cards more
            'garbage': {'threshold': 0.3}      # Always redraw garbage hands
        }
    
    def strategic_redraw(self, hole_cards, community_cards, street, position, opponent_model=None):
        """Strategic redraw decision optimized for a 27-card deck."""
        # Sanity checks
        if len(hole_cards) != 2 or street > 1:
            return False, -1
        
        # Filter out invalid cards
        hole_cards = [c for c in hole_cards if c != -1]
        community_cards = [c for c in community_cards if c != -1]
        
        if len(hole_cards) < 2:
            return False, -1
        
        # Extract ranks and suits
        ranks = [card // 3 for card in hole_cards]
        suits = [card % 3 for card in hole_cards]
        
        # Check for paired hole cards (never break up a pair in general)
        if ranks[0] == ranks[1]:
            current_strength = self.hand_evaluator.evaluate(hole_cards, community_cards)
            # Only consider redrawing very weak pairs on the flop
            if street == 1 and current_strength < 0.4 and ranks[0] < 3:  # Very low pair (2-4)
                return True, 0  # Discard one of the low cards
            return False, -1
        
        # Determine current hand type
        hand_type = self._determine_hand_type(hole_cards)
        
        # Get current hand strength
        current_strength = self.hand_evaluator.evaluate(hole_cards, community_cards)
        
        # Get opponent tendencies if available
        opponent_tendencies = {}
        if opponent_model:
            opponent_tendencies = opponent_model.get_tendencies()
        
        # Adjust threshold based on opponent tendencies and hand type
        adjusted_threshold = self._adjust_threshold(
            self.redraw_thresholds[street], 
            opponent_tendencies, 
            hand_type,
            street,
            position
        )
        
        # Determine if we should redraw
        if current_strength < adjusted_threshold:
            # Determine which card to discard
            discard_idx = self._select_card_to_discard(hole_cards, community_cards)
            return True, discard_idx
        
        return False, -1
    
    def _determine_hand_type(self, hole_cards):
        """Determine the type of hand for redraw consideration."""
        ranks = [card // 3 for card in hole_cards]
        suits = [card % 3 for card in hole_cards]
        
        # Check for pairs
        if ranks[0] == ranks[1]:
            return 'pair'
        
        # Check for suited cards
        if suits[0] == suits[1]:
            return 'suited'
        
        # Check for connected cards (for straights)
        rank_diff = abs(ranks[0] - ranks[1])
        if rank_diff <= 2:
            return 'connected'
        
        # Check for high cards (9 or A)
        if ranks[0] >= 7 or ranks[1] >= 7:
            return 'high_cards'
        
        # Low unconnected cards
        return 'garbage'
    
    def _adjust_threshold(self, base_threshold, opponent_tendencies, hand_type, street, position):
        """Adjust redraw threshold based on game context."""
        adjusted = base_threshold
        
        # Adjust based on hand type
        if hand_type in self.hand_types:
            adjusted = min(adjusted, self.hand_types[hand_type]['threshold'])
        
        # Position adjustments
        if position == "SB":  # In position post-flop
            adjusted += 0.05  # Be more conservative with redraws in position
        
        # Opponent tendencies adjustments
        aggression = opponent_tendencies.get('aggression', 0.5)
        
        # Against aggressive opponents, value made hands more
        if aggression > 0.7:
            adjusted -= 0.05
        
        # Street-specific adjustments
        if street == 1:  # Flop
            # More conservative with redraw on flop
            adjusted -= 0.05
            
            # Check if opponent has redrawn
            if opponent_tendencies.get('redraw_used', False):
                # Opponent has already used their redraw
                redraw_analysis = opponent_tendencies.get('redraw_analysis', {})
                
                if redraw_analysis.get('hand_quality') == 'weak':
                    # If opponent likely has a weak hand, be more aggressive
                    adjusted += 0.1
                elif redraw_analysis.get('hand_quality') in ['medium-strong', 'strong']:
                    # If opponent likely has a strong hand, be more conservative
                    adjusted -= 0.1
        
        # Ensure threshold is in valid range
        return min(0.7, max(0.2, adjusted))
    
    def _select_card_to_discard(self, hole_cards, community_cards):
        """Select which card to discard based on strategic considerations."""
        # Extract ranks and suits
        ranks = [card // 3 for card in hole_cards]
        suits = [card % 3 for card in hole_cards]
        
        # Get community card info if available
        if community_cards:
            community_ranks = [card // 3 for card in community_cards]
            community_suits = [card % 3 for card in community_cards]
        else:
            community_ranks = []
            community_suits = []
        
        # Never discard an Ace
        if ranks[0] == 8:  # Ace's rank is 8
            return 1
        if ranks[1] == 8:
            return 0
        
        # Check for pairs with board
        for i, rank in enumerate(ranks):
            if rank in community_ranks:
                # Keep cards that pair with the board
                return 1 - i
        
        # Check for flush draws
        for i, suit in enumerate(suits):
            if community_suits.count(suit) >= 2:
                # Keep cards that could make a flush
                return 1 - i
        
        # Check for straight potential
        straight_potential = [0, 0]
        for i, rank in enumerate(ranks):
            # Count nearby ranks for straight potential
            for r in range(rank - 2, rank + 3):
                if r in community_ranks or r in [ranks[1-i]]:
                    straight_potential[i] += 1
        
        # Keep the card with better straight potential
        if straight_potential[0] > straight_potential[1]:
            return 1
        elif straight_potential[1] > straight_potential[0]:
            return 0
        
        # Default: discard the lower ranked card
        return 0 if ranks[0] < ranks[1] else 1