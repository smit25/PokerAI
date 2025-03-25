from collections import defaultdict
from gym_env import PokerEnv

action_types = PokerEnv.ActionType

class HandEvaluator:
    """Fast hand evaluator optimized for 27-card poker variant."""
    
    def __init__(self):
        # Define card constants
        self.RANKS = "23456789A"  # 2-9, Ace (9 ranks in 27-card deck)
        self.SUITS = "dhs"        # diamonds, hearts, spades (3 suits in 27-card deck)
        
        # Initialize cache for hand strength evaluations
        self.hand_cache = {}
    
    def evaluate(self, hole_cards, board=[]):
        """
        Main evaluation function - returns hand strength on 0-1 scale.
        Higher values represent stronger hands.
        """
        # Filter out invalid cards
        hole_cards = [c for c in hole_cards if c != -1]
        board = [c for c in board if c != -1]
        
        if len(hole_cards) < 2:
            return 0.5  # Default when we don't have enough information
        
        # Create cache key
        key = (tuple(sorted(hole_cards)), tuple(sorted(board)))
        
        # Check cache first for speed
        if key in self.hand_cache:
            return self.hand_cache[key]
        
        # Preflop evaluation (no board cards)
        if not board:
            strength = self._preflop_strength(hole_cards)
            self.hand_cache[key] = strength
            return strength
        
        # Postflop evaluation
        hand_type, relative_strength = self._evaluate_made_hand(hole_cards, board)
        
        # Convert hand type and relative strength to single value
        if hand_type == "high_card":
            strength = 0.2 + relative_strength * 0.2
        elif hand_type == "pair":
            strength = 0.4 + relative_strength * 0.15
        elif hand_type == "two_pair":
            strength = 0.55 + relative_strength * 0.1
        elif hand_type == "three_kind":
            strength = 0.65 + relative_strength * 0.1
        elif hand_type == "straight":
            strength = 0.75 + relative_strength * 0.05
        elif hand_type == "flush":
            strength = 0.8 + relative_strength * 0.05
        elif hand_type == "full_house":
            strength = 0.85 + relative_strength * 0.05
        elif hand_type == "four_kind":
            strength = 0.9 + relative_strength * 0.05
        elif hand_type == "straight_flush":
            strength = 0.95 + relative_strength * 0.05
        else:
            strength = 0.5  # Default
        
        # Cache and return
        self.hand_cache[key] = strength
        return strength
    
    def _preflop_strength(self, hole_cards):
        """Calculate preflop hand strength for 27-card deck variant."""
        rank1, rank2 = hole_cards[0] // 3, hole_cards[1] // 3
        suit1, suit2 = hole_cards[0] % 3, hole_cards[1] % 3
        
        # Pairs are stronger in 27-card variant
        if rank1 == rank2:
            # Higher pairs are stronger
            return 0.6 + (rank1 / 8.0) * 0.35
        
        # Extract high and low card
        high_rank = max(rank1, rank2)
        low_rank = min(rank1, rank2)
        
        # Suited bonus
        suited = suit1 == suit2
        suited_bonus = 0.1 if suited else 0
        
        # Connectedness bonus
        connected_bonus = 0
        gap = abs(rank1 - rank2)
        if gap <= 3:
            connected_bonus = 0.1 * (1 - gap/3)
            
        # Base strength (high cards matter more in 27-card variant)
        return min(0.95, max(0.1, 0.3 + (high_rank/8) * 0.3 + (low_rank/8) * 0.15 + suited_bonus + connected_bonus))
    
    def _evaluate_made_hand(self, hole_cards, board):
        """
        Evaluate the made hand and return (hand_type, relative_strength).
        Used for postflop evaluation.
        """
        # Combine all cards
        all_cards = hole_cards + board
        
        # Extract ranks and suits
        ranks = [card // 3 for card in all_cards]
        suits = [card % 3 for card in all_cards]
        
        # Count frequencies
        rank_counts = defaultdict(int)
        for r in ranks:
            rank_counts[r] += 1
        
        suit_counts = defaultdict(int)
        for s in suits:
            suit_counts[s] += 1
        
        # Find made hand types
        # Check for flush
        flush_suit = None
        for suit, count in suit_counts.items():
            if count >= 5:
                flush_suit = suit
                break
        
        # Check for straight
        unique_ranks = sorted(set(ranks))
        straight_high = None
        
        # Add low ace for wheel straight possibility
        if 8 in unique_ranks:  # 8 is Ace's rank
            straight_ranks = unique_ranks + [-1]  # Add virtual low ace
        else:
            straight_ranks = unique_ranks
            
        straight_ranks = sorted(straight_ranks)
        
        # Check for 5 consecutive ranks
        for i in range(len(straight_ranks) - 4):
            if straight_ranks[i+4] - straight_ranks[i] == 4:
                straight_high = straight_ranks[i+4]
        
        # Check for straight flush
        straight_flush = False
        if flush_suit is not None and straight_high is not None:
            # Check if the straight cards are all in the flush suit
            flush_ranks = [ranks[i] for i in range(len(ranks)) if suits[i] == flush_suit]
            if len(set(flush_ranks)) >= 5:  # Need at least 5 different ranks
                # Check for straight in the flush cards
                unique_flush_ranks = sorted(set(flush_ranks))
                
                # Add low ace for wheel straight flush possibility
                if 8 in unique_flush_ranks:
                    unique_flush_ranks = sorted(unique_flush_ranks + [-1])
                
                for i in range(len(unique_flush_ranks) - 4):
                    if unique_flush_ranks[i+4] - unique_flush_ranks[i] == 4:
                        straight_flush = True
                        break
        
        # Find four of a kind
        four_kind_rank = None
        for rank, count in rank_counts.items():
            if count >= 4:
                four_kind_rank = rank
                break
        
        # Find three of a kind and pairs
        three_kind_ranks = []
        pair_ranks = []
        
        for rank, count in rank_counts.items():
            if count >= 3:
                three_kind_ranks.append(rank)
            if count >= 2:
                pair_ranks.append(rank)
        
        # Sort by rank (high to low)
        three_kind_ranks.sort(reverse=True)
        pair_ranks.sort(reverse=True)
        
        # Determine hand type and relative strength
        if straight_flush:
            return "straight_flush", min(1.0, straight_high / 8.0)
            
        if four_kind_rank is not None:
            return "four_kind", min(1.0, four_kind_rank / 8.0)
            
        if three_kind_ranks and len(pair_ranks) >= 2:
            # Full house
            return "full_house", min(1.0, three_kind_ranks[0] / 8.0)
            
        if flush_suit is not None:
            # Get the highest 5 flush cards
            flush_cards = [ranks[i] for i in range(len(ranks)) if suits[i] == flush_suit]
            flush_cards.sort(reverse=True)
            flush_strength = min(1.0, flush_cards[0] / 8.0)
            return "flush", flush_strength
            
        if straight_high is not None:
            return "straight", min(1.0, straight_high / 8.0)
            
        if three_kind_ranks:
            return "three_kind", min(1.0, three_kind_ranks[0] / 8.0)
            
        if len(pair_ranks) >= 2:
            # Two pair
            two_pair_strength = (pair_ranks[0] / 8.0 * 0.8) + (pair_ranks[1] / 8.0 * 0.2)
            return "two_pair", min(1.0, two_pair_strength)
            
        if pair_ranks:
            return "pair", min(1.0, pair_ranks[0] / 8.0)
            
        # High card
        high_card = max(ranks)
        return "high_card", min(1.0, high_card / 8.0)
    
    def has_draw(self, hole_cards, board):
        """Check if the hand has a strong draw (flush or straight draw)."""
        # Need at least 3 cards to have a draw
        if len(board) < 3:
            return False
            
        # Extract all cards
        all_cards = hole_cards + board
        ranks = [card // 3 for card in all_cards]
        suits = [card % 3 for card in all_cards]
        
        # Check for flush draw (4 cards of same suit)
        suit_counts = defaultdict(int)
        for s in suits:
            suit_counts[s] += 1
            
        has_flush_draw = max(suit_counts.values()) == 4
        
        # Check for open-ended straight draw
        unique_ranks = sorted(set(ranks))
        
        # Add low ace
        if 8 in unique_ranks:
            unique_ranks = sorted(unique_ranks + [-1])
            
        has_straight_draw = False
        for i in range(len(unique_ranks) - 3):
            if unique_ranks[i+3] - unique_ranks[i] == 3:  # 4 consecutive ranks
                has_straight_draw = True
                break
                
        # Check for gutshot straight draw
        has_gutshot = False
        if not has_straight_draw:
            for i in range(len(unique_ranks) - 3):
                if unique_ranks[i+3] - unique_ranks[i] == 4:  # 4 ranks with 1 gap
                    has_gutshot = True
                    break
        
        return has_flush_draw or has_straight_draw or has_gutshot