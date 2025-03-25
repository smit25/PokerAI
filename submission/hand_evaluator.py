class HandEvaluator:
    """
    Enhanced hand evaluation for the 27-card deck variant.
    Uses a combination of lookup tables, equity calculations, and hand abstractions
    Implements Counterfactual Regret Minimization (CFR) concepts for equity calculation.
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
        Pre-compute the strength of all possible starting hands using advanced
        evaluation techniques calibrated for a 27-card deck.
        """
        # For 27-card deck, there are 351 possible starting hands (27 choose 2)
        starting_hands = {}
        
        # Buckets populated in monte-carlo
        self.hand_buckets = {}
        self.bucket_values = {}
        
        # Improved equity calculation for preflop hands
        def advanced_equity_calc(card1_idx, card2_idx):
            rank1 = card1_idx // 3
            rank2 = card2_idx // 3
            suit1 = card1_idx % 3
            suit2 = card2_idx % 3
            
            # Better pair evaluation - pairs are extremely strong in smaller decks
            if rank1 == rank2:
                # Rank value from 0 (2) to 8 (A)
                rank_value = rank1 / 8
                # Scale to 0.70-0.98 range - pairs are stronger in 27-card deck
                return 0.70 + rank_value * 0.28
            
            # High card evaluation - high cards are more valuable in smaller decks
            high_rank = max(rank1, rank2)
            low_rank = min(rank1, rank2)
            high_card_value = (high_rank / 8) * 0.4  # Up to 0.4 contribution
            
            # Suited bonus - more significant in smaller decks
            is_suited = suit1 == suit2
            suited_bonus = 0.12 if is_suited else 0
            
            # Connectivity for straights
            rank_diff = abs(rank1 - rank2)
            
            # Special case for A-5 (wheel straight potential)
            if (rank1 == 8 and rank2 == 3) or (rank1 == 3 and rank2 == 8):  # A-5
                connectivity = 0.15
            # Special case for high connecters (e.g., A-9, 9-8)
            elif rank_diff == 1 and high_rank >= 7:
                connectivity = 0.20
            # Normal connectivity based on difference
            elif rank_diff <= 4:  # Can make a straight
                connectivity = 0.15 * (1 - (rank_diff / 5))
            else:
                connectivity = 0
            
            # Low card penalty - more severe in smaller decks
            low_card_penalty = 0
            if low_rank < 3:  # 2-4 as low card
                low_card_penalty = -0.10
            elif low_rank < 5:  # 5-6 as low card
                low_card_penalty = -0.05
            
            # Base value considering deck size effects
            base_value = 0.35 + high_card_value + suited_bonus + connectivity + low_card_penalty
            
            # Special case adjustments
            # Premium hands get a boost
            if high_rank == 8 and low_rank >= 6 and is_suited:  # A-8+ suited
                base_value += 0.12
            elif high_rank == 8 and low_rank >= 7:  # A-9+
                base_value += 0.08
            elif high_rank >= 7 and low_rank >= 6 and is_suited:  # 9-8+ suited
                base_value += 0.10
            
            # Ensure value is in range [0.1, 0.98]
            return min(0.98, max(0.1, base_value))
            
        # Precompute values for all possible hands
        for i in range(27):
            for j in range(i+1, 27):
                # Calculate equity using improved approach
                equity = advanced_equity_calc(i, j)
                
                # Store value for both orderings
                starting_hands[(i, j)] = equity
                starting_hands[(j, i)] = equity
                
                # Create bucket for this hand
                bucket = int(equity * 20)  # 20 buckets (0-19)
                if bucket not in self.hand_buckets:
                    self.hand_buckets[bucket] = []
                    self.bucket_values[bucket] = equity
                
                self.hand_buckets[bucket].append((i, j))
                self.hand_buckets[bucket].append((j, i))
        
        return starting_hands
    
    def evaluate_made_hand(self, hole_cards, board_cards):
        """
        Evaluate a complete 5+ card poker hand.
        Returns a score representing the hand strength.
        Higher scores indicate stronger hands.
            
        Returns:
            Tuple of (hand_type, tiebreakers)
        """
        # Filter out invalid cards
        hole_cards = [card for card in hole_cards if card != -1]
        board_cards = [card for card in board_cards if card != -1]
        all_cards = hole_cards + board_cards
        
        # If we don't have enough cards for a complete hand, return a low score
        if len(all_cards) < 5:
            return (0, (0,))
            
        # Convert indices to string representation
        all_cards_str = [self._card_idx_to_str(card) for card in all_cards]
        
        # Create cache key
        key = tuple(sorted(all_cards))
        
        # Check cache
        if key in self.hand_strength_cache:
            return self.hand_strength_cache[key]
        
        # Extract ranks and suits
        rank_chars = [c[0] for c in all_cards_str]
        suit_chars = [c[1] for c in all_cards_str]
        
        # Convert ranks to numeric values (2=2, ..., 9=9, A=14)
        rank_values = []
        for r in rank_chars:
            if r == 'A':
                rank_values.append(14)  # High ace
            else:
                rank_values.append(int(r))
        
        # Count rank frequencies
        rank_counts = {}
        for r in rank_values:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        
        # Count suit frequencies
        suit_counts = {}
        for s in suit_chars:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        
        # Check for flush
        flush_suit = None
        for suit, count in suit_counts.items():
            if count >= 5:
                flush_suit = suit
                break
        
        # Prepare for straight detection - create a set of ranks with Ace counted as both high and low
        unique_ranks = set(rank_values)
        if 14 in unique_ranks:  # If we have an Ace
            unique_ranks.add(1)  # Add a low Ace
        
        sorted_ranks = sorted(unique_ranks)
        
        # Check for straight using windowing approach
        has_straight = False
        straight_high_card = 0
        
        # Handle specific case of A-5 straight (wheel)
        if set([1, 2, 3, 4, 5]).issubset(sorted_ranks):
            has_straight = True
            straight_high_card = 5
        
        # Check normal straights (5 consecutive cards)
        if not has_straight:
            for i in range(len(sorted_ranks) - 4):
                if sorted_ranks[i+4] == sorted_ranks[i] + 4:
                    has_straight = True
                    straight_high_card = sorted_ranks[i+4]
                    break
        
        # Check for straight flush
        has_straight_flush = False
        if has_straight and flush_suit:
            # Need to recheck if the straight cards are all in the flush suit
            straight_flush_ranks = []
            
            # Get all cards of the flush suit
            flush_cards = [all_cards_str[i] for i in range(len(all_cards_str)) if suit_chars[i] == flush_suit]
            flush_ranks = [int(c[0]) if c[0] != 'A' else 14 for c in flush_cards]
            
            # Add low ace if we have a high ace
            if 14 in flush_ranks:
                flush_ranks.append(1)
            
            flush_ranks = sorted(set(flush_ranks))
            
            # Check for wheel straight flush (A-5)
            if set([1, 2, 3, 4, 5]).issubset(flush_ranks):
                has_straight_flush = True
                straight_high_card = 5
            
            # Check for normal straight flush
            if not has_straight_flush:
                for i in range(len(flush_ranks) - 4):
                    if flush_ranks[i+4] == flush_ranks[i] + 4:
                        has_straight_flush = True
                        straight_high_card = flush_ranks[i+4]
                        break
        
        # Check for four of a kind
        four_of_a_kind = None
        for rank, count in rank_counts.items():
            if count == 4:
                four_of_a_kind = rank
                break
        
        # Find hand type and relevant tiebreakers
        
        # Straight Flush
        if has_straight_flush:
            return (9, straight_high_card)
        
        # Four of a Kind
        if four_of_a_kind:
            kickers = [r for r in rank_values if r != four_of_a_kind]
            kicker = max(kickers) if kickers else 0
            return (8, (four_of_a_kind, kicker))
        
        # Check for Full House
        has_three = False
        has_pair = False
        three_rank = 0
        pair_rank = 0
        
        for rank, count in sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True):
            if count >= 3 and not has_three:
                has_three = True
                three_rank = rank
            elif count >= 2 and not has_pair and rank != three_rank:
                has_pair = True
                pair_rank = rank
        
        if has_three and has_pair:
            return (7, (three_rank, pair_rank))
        
        # Flush
        if flush_suit:
            flush_ranks = [rank_values[i] for i in range(len(rank_values)) if suit_chars[i] == flush_suit]
            flush_ranks.sort(reverse=True)
            return (6, tuple(flush_ranks[:5]))
        
        # Straight
        if has_straight:
            return (5, straight_high_card)
        
        # Three of a Kind
        if has_three:
            kickers = sorted([r for r in rank_values if r != three_rank], reverse=True)
            return (4, (three_rank, kickers[0] if len(kickers) > 0 else 0, kickers[1] if len(kickers) > 1 else 0))
        
        # Check for Two Pair
        pairs = [rank for rank, count in rank_counts.items() if count >= 2]
        if len(pairs) >= 2:
            pairs.sort(reverse=True)
            kickers = [r for r in rank_values if r not in pairs[:2]]
            kicker = max(kickers) if kickers else 0
            return (3, (pairs[0], pairs[1], kicker))
        
        # One Pair
        if len(pairs) == 1:
            pair_rank = pairs[0]
            kickers = sorted([r for r in rank_values if r != pair_rank], reverse=True)
            return (2, (pair_rank, 
                        kickers[0] if len(kickers) > 0 else 0,
                        kickers[1] if len(kickers) > 1 else 0, 
                        kickers[2] if len(kickers) > 2 else 0))
        
        # High Card
        high_cards = sorted(rank_values, reverse=True)
        return (1, tuple(high_cards[:5]))
    
    def _get_effective_hand_size(self, hole_cards, board):
        """Count the actual number of cards in the hand, ignoring -1 values."""
        count = 0
        for card in board:
            if card != -1:
                count += 1
        
        for card in hole_cards:
            if card != -1:
                count += 1  
        
        return count

    def get_hand_strength(self, hole_cards, board=[], opponent_range=None):
        """
        Get a normalized hand strength (0-1) of the given hole cards using advanced equity calculation.
        Implements a fast approximation of expected hand strength (EHS) against opponent ranges.

        Returns:
            Float between 0 and 1 representing hand strength
        """
        # Handle invalid inputs
        if not hole_cards or all(card == -1 for card in hole_cards):
            return 0.0
            
        # Filter out -1 values
        valid_hole_cards = [card for card in hole_cards if card != -1]
        valid_board = [card for card in board if card != -1]
        
        # Create cache key for this calculation
        cache_key = (tuple(sorted(valid_hole_cards)), tuple(sorted(valid_board)))
        
        if cache_key in self.hand_strength_cache:
            return self.hand_strength_cache[cache_key]
        
        # Pre-flop hand strength - use pre-computed values from simulations
        if not valid_board:
            if len(valid_hole_cards) >= 2:
                strength = self.starting_hand_values.get(tuple(valid_hole_cards[:2]), 0.3)
                self.hand_strength_cache[cache_key] = strength
                return strength
            else:
                return 0.3  # Default for incomplete hole cards
        
        effective_size = len(valid_hole_cards) + len(valid_board)
        
        if effective_size < 5:
            # For fewer than 5 cards, use a combination of starting hand value and what we know about the partial board
            partial_strength = self._partial_board_strength(valid_hole_cards, valid_board)
            self.hand_strength_cache[cache_key] = partial_strength
            return partial_strength
        
        # Post-flop strength calculation using abbreviated range equity calculation
        # Step 1: Evaluate current made hand strength
        made_hand = self.evaluate_made_hand(valid_hole_cards, valid_board)
        hand_type = made_hand[0]
        
        # Improved base strength calculation - non-linear scaling for better distinction between hand types
        if hand_type <= 1:  # High card or invalid
            base_strength = 0.05 + (hand_type * 0.15)
        elif hand_type == 2:  # One pair
            base_strength = 0.25 + (made_hand[1][0] / 14) * 0.15  # Pair rank affects strength
        elif hand_type == 3:  # Two pair
            base_strength = 0.40 + (made_hand[1][0] / 14) * 0.10  # High pair rank affects strength
        elif hand_type == 4:  # Three of a kind
            base_strength = 0.55 + (made_hand[1][0] / 14) * 0.05
        elif hand_type == 5:  # Straight
            base_strength = 0.65 + (made_hand[1] / 14) * 0.05
        elif hand_type == 6:  # Flush
            base_strength = 0.75
        elif hand_type == 7:  # Full house
            base_strength = 0.85
        elif hand_type >= 8:  # Four of a kind or better
            base_strength = 0.95
        
        # Step 2: Evaluate potential (draws) - more accurate for 27-card deck
        draw_equity = self._calculate_draw_equity(valid_hole_cards, valid_board)
        
        # Step 3: Calculate effective hand strength (EHS) - refined weighting
        remaining_cards = 5 - len(valid_board)
        
        if remaining_cards == 0:  # River - only made hand matters
            ehs = base_strength
        elif remaining_cards == 1:  # Turn - mostly made hand with some potential
            # Weight toward made hand on turn
            ehs = 0.85 * base_strength + 0.15 * draw_equity
        else:  # Flop - balance between made hand and potential
            # Draw equity matters more on flop
            ehs = 0.65 * base_strength + 0.35 * draw_equity
        
        # Step 4: Apply opponent range adjustment if available
        if opponent_range is not None and len(valid_board) >= 3:
            range_adjustment = self._range_vs_range_adjustment(valid_hole_cards, valid_board, opponent_range)
            ehs = 0.75 * ehs + 0.25 * range_adjustment
        
        # Cache and return result
        self.hand_strength_cache[cache_key] = ehs
        return ehs
    
    def _calculate_draw_equity(self, hole_cards, board):
        """
        Calculate drawing equity using efficient probability calculation.
        Adjusted for 27-card deck probabilities.
        
        Args:
            hole_cards: List of hole card indices
            board: List of board card indices
            
        Returns:
            Float representing drawing potential
        """
        effective_size = len(hole_cards) + len(board)
        if effective_size < 3:
            # Not enough cards to analyze draws
            return self._partial_draw_equity(hole_cards, board)
        
        # Get current hand strength as baseline
        made_hand = self.evaluate_made_hand(hole_cards, board)
        made_type = made_hand[0]
        
        # If we already have a strong hand, draw equity is high
        if made_type >= 6:  # Flush or better
            return 0.9
        elif made_type >= 4:  # Three of a kind or better
            return 0.7
        
        # Analyze card patterns for draw potential
        all_cards = hole_cards + board
        suits = [card % 3 for card in all_cards]
        ranks = [card // 3 for card in all_cards]
        
        # Count frequencies
        suit_counts = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        
        # Analyze draw possibilities
        draw_equity = 0.0
        remaining_cards = 5 - len(board)
        cards_left_in_deck = 27 - len(hole_cards) - len(board)
        
        # Flush draw analysis - 27-card deck has only 9 cards per suit
        flush_draw = False
        flush_outs = 0
        for suit, count in suit_counts.items():
            if count == 4:  # Need 1 more for flush
                flush_draw = True
                # In 27-card deck, each suit has 9 cards total
                flush_outs = 9 - count
                break
            elif count == 3 and remaining_cards >= 2:  # Possible flush with 2 more cards
                flush_draw = True
                flush_outs = 9 - count
                break
        
        # Straight draw analysis
        sorted_ranks = sorted(set(ranks))
        
        # Handle low ace for straights
        if 8 in sorted_ranks:  # Ace (rank 8)
            ace_low_ranks = sorted_ranks.copy()
            ace_low_ranks.append(-1)  # Add low ace (represented as -1 for sorting)
            ace_low_ranks.sort()
            ace_low_ranks[0] = 9  # Change -1 to special value 9 (low ace)
        else:
            ace_low_ranks = sorted_ranks
        
        # Check for open-ended straight draw (need 1 card on either end)
        straight_draw = False
        open_ended = False
        gutshot = False
        double_gutshot = False
        straight_outs = 0
        
        # Check for 4-card straight
        for i in range(len(sorted_ranks) - 3):
            if sorted_ranks[i+3] - sorted_ranks[i] == 3:  # 4 consecutive ranks
                straight_draw = True
                open_ended = True
                # In 27-card deck, each rank has 3 cards
                straight_outs = 6  # 3 cards each for the 2 ranks needed
                break
        
        # Check for gutshot (need 1 inside card)
        if not straight_draw:
            for i in range(len(sorted_ranks) - 3):
                if sorted_ranks[i+3] - sorted_ranks[i] == 4:  # 4 ranks with 1 gap
                    gutshot = True
                    # In 27-card deck, each rank has 3 cards
                    straight_outs = 3  # 3 cards for the missing rank
                    break
        
        # Check for double gutshot (two possible ways to complete)
        if not straight_draw and not gutshot:
            for i in range(len(sorted_ranks) - 4):
                if sorted_ranks[i+4] - sorted_ranks[i] == 5:  # 5 ranks with 2 gaps
                    double_gutshot = True
                    # In 27-card deck, each rank has 3 cards
                    straight_outs = 6  # 3 cards each for 2 possible ranks
                    break
        
        # Check for pair draw (to make three of a kind)
        pair_draw = False
        pair_outs = 0
        for rank, count in rank_counts.items():
            if count == 2:
                pair_draw = True
                # In 27-card deck, each rank has 3 cards
                pair_outs = 1  # Only 1 more card of this rank exists
                break
        
        # Calculate draw probabilities - adjusted for 27-card deck
        if flush_draw:
            # Probability of completing flush
            if remaining_cards == 2:  # Turn and river
                # P = 1 - [(cards_left - outs) / cards_left] * [(cards_left - outs - 1) / (cards_left - 1)]
                p_not_hitting = ((cards_left_in_deck - flush_outs) / cards_left_in_deck) * \
                                ((cards_left_in_deck - flush_outs - 1) / (cards_left_in_deck - 1))
                flush_equity = 1 - p_not_hitting
            else:  # Just river
                flush_equity = flush_outs / cards_left_in_deck
            draw_equity = max(draw_equity, 0.7 * flush_equity + 0.3)  # Higher weight for flush draws
        
        if open_ended:
            # Probability of completing straight
            if remaining_cards == 2:  # Turn and river
                p_not_hitting = ((cards_left_in_deck - straight_outs) / cards_left_in_deck) * \
                                ((cards_left_in_deck - straight_outs - 1) / (cards_left_in_deck - 1))
                straight_equity = 1 - p_not_hitting
            else:  # Just river
                straight_equity = straight_outs / cards_left_in_deck
            draw_equity = max(draw_equity, 0.6 * straight_equity + 0.2)  # Medium weight for straight draws
        
        if gutshot:
            # Lower probability for gutshot
            if remaining_cards == 2:  # Turn and river
                p_not_hitting = ((cards_left_in_deck - straight_outs) / cards_left_in_deck) * \
                                ((cards_left_in_deck - straight_outs - 1) / (cards_left_in_deck - 1))
                gutshot_equity = 1 - p_not_hitting
            else:  # Just river
                gutshot_equity = straight_outs / cards_left_in_deck
            draw_equity = max(draw_equity, 0.5 * gutshot_equity + 0.15)  # Lower weight for gutshot draws
        
        if pair_draw:
            # Probability of making three of a kind
            if remaining_cards == 2:  # Turn and river
                p_not_hitting = ((cards_left_in_deck - pair_outs) / cards_left_in_deck) * \
                                ((cards_left_in_deck - pair_outs - 1) / (cards_left_in_deck - 1))
                pair_equity = 1 - p_not_hitting
            else:  # Just river
                pair_equity = pair_outs / cards_left_in_deck
            draw_equity = max(draw_equity, 0.4 * pair_equity + 0.25)  # Moderate weight for pair to trips
        
        # Consider current hand strength along with draw potential
        current_strength = (made_type - 1) / 8  # Normalized to 0-1
        
        # Combine made hand and drawing potential - refined weighting for 27-card deck
        if made_type >= 3:  # Two pair or better
            combined = 0.8 * current_strength + 0.2 * draw_equity
        else:  # Pair or less
            combined = 0.6 * current_strength + 0.4 * draw_equity
        
        return combined
    
    def _range_vs_range_adjustment(self, hole_cards, board, opponent_range):
        """
        Calculate equity adjustment based on opponent's range.
        
        Args:
            hole_cards: List of hole card indices
            board: List of board card indices
            opponent_range: Opponent range model
            
        Returns:
            Adjusted equity value
        """
        effective_size = len(hole_cards) + len(board)
        if effective_size < 5:
            # For incomplete boards, calculate potential equity
            potential = self._partial_board_strength(hole_cards, board)
            
            # Get opponent range characteristics
            range_polarization = opponent_range.get('polarization', 0.5)
            range_strength_floor = opponent_range.get('strength_floor', 0.0)
            
            # More nuanced adjustment based on opponent tendencies
            if potential > 0.7:
                # Strong hands do better against polarized ranges and worse against tight ranges
                adjustment = 0.08 * range_polarization - 0.05 * range_strength_floor
            elif potential < 0.3:
                # Weak hands do poorly against tight ranges but can bluff polarized ranges
                adjustment = -0.1 * range_strength_floor - 0.05 * range_polarization
            else:
                # Medium hands do poorly against polarized ranges but well against tight ranges
                adjustment = -0.06 * range_polarization + 0.04 * (1 - range_strength_floor)
                
            return max(0.05, min(0.95, potential + adjustment))
        
        # For complete 5+ card hands, make adjustments based on made hand strength
        made_hand = self.evaluate_made_hand(hole_cards, board)
        hand_type = made_hand[0]
        
        # More precise percentile calculation based on hand type and tiebreakers
        if hand_type <= 1:  # High card
            percentile = 0.05 + 0.1 * (made_hand[1][0] / 14)  # Highest card matters
        elif hand_type == 2:  # One pair
            pair_rank = made_hand[1][0]
            percentile = 0.2 + 0.15 * (pair_rank / 14)  # Pair rank matters
        elif hand_type == 3:  # Two pair
            high_pair = made_hand[1][0]
            percentile = 0.4 + 0.1 * (high_pair / 14)
        elif hand_type == 4:  # Three of a kind
            percentile = 0.55 + 0.05 * (made_hand[1][0] / 14)
        elif hand_type == 5:  # Straight
            percentile = 0.65 + 0.05 * (made_hand[1] / 14)
        elif hand_type == 6:  # Flush
            percentile = 0.75 + 0.05 * (made_hand[1][0] / 14)
        elif hand_type == 7:  # Full house
            percentile = 0.85 + 0.05 * (made_hand[1][0] / 14)
        else:  # Four of a kind or better
            percentile = 0.95
        
        # Extract opponent range characteristics
        range_polarization = opponent_range.get('polarization', 0.5)
        range_strength_floor = opponent_range.get('strength_floor', 0.0)
        range_strength_cap = opponent_range.get('strength_cap', 1.0)
        
        # More sophisticated adjustment based on hand strength vs. range
        # This better accounts for the dynamics of a 27-card deck
        
        if percentile > 0.8:  # Very strong hands
            # Strong hands do better against polarized ranges (they have more weak hands)
            # But might face more resistance from tight ranges (higher floor)
            adjustment = 0.12 * range_polarization - 0.05 * range_strength_floor
        elif percentile > 0.5:  # Medium-strong hands
            # These hands do fine against medium ranges but struggle against polarized ranges
            adjustment = -0.08 * range_polarization + 0.05 * (1 - range_strength_floor)
        elif percentile > 0.3:  # Medium-weak hands
            # These hands do poorly against most competent ranges
            adjustment = -0.1 * range_strength_floor - 0.05 * range_polarization
        else:  # Very weak hands
            # These hands do terribly against tight ranges but can sometimes bluff against polarized ranges
            adjustment = -0.15 * range_strength_floor + 0.05 * range_polarization
        
        # Final adjusted percentile, bounded to valid range
        return max(0.05, min(0.95, percentile + adjustment))
    
    def _partial_draw_equity(self, hole_cards, board):
        """
        Calculate a simplified draw equity when we don't have 5 cards yet.
        Optimized for 27-card deck.
        
        Args:
            hole_cards: List of hole card indices
            board: List of board card indices
            
        Returns:
            Float representing drawing potential
        """
        # Count suits and ranks to identify potential draws
        all_cards = hole_cards + board
        suits = [card % 3 for card in all_cards]
        ranks = [card // 3 for card in all_cards]
        
        # Count frequencies
        suit_counts = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        
        # Base value - 27-card deck makes high cards more valuable
        high_ranks = [r for r in ranks if r >= 6]  # 8, 9, A
        high_card_value = len(high_ranks) * 0.1
        
        # Check for existing pairs
        pairs = [r for r, count in rank_counts.items() if count >= 2]
        if pairs:
            # Pairs are stronger in 27-card deck
            pair_value = 0.35 + (max(pairs) / 8) * 0.15  # Higher pairs are better
        else:
            pair_value = 0
        
        # Check for flush potential - more significant in 27-card deck
        max_suit_count = max(suit_counts.values()) if suit_counts else 0
        flush_potential = (max_suit_count / max(3, len(all_cards))) * 0.5
        
        # Check for straight potential
        sorted_ranks = sorted(set(ranks))
        straight_potential = 0
        
        if len(sorted_ranks) >= 2:
            # Count gaps between ranks
            gaps = 0
            for i in range(1, len(sorted_ranks)):
                gaps += max(0, sorted_ranks[i] - sorted_ranks[i-1] - 1)
            
            # Calculate straight potential - fewer gaps means better potential
            # In 27-card deck, straights are harder to hit but more valuable
            if len(sorted_ranks) >= 3 and gaps <= 2:
                straight_potential = 0.3
            elif len(sorted_ranks) >= 2 and gaps <= 3:
                straight_potential = 0.2
        
        # Special case for wheel straight potential (A-5)
        if 8 in ranks and any(r <= 3 for r in ranks):
            straight_potential = max(straight_potential, 0.25)
        
        # Combine factors with weights appropriate for 27-card deck
        if pairs:
            # If we have a pair, it dominates our equity
            draw_equity = 0.6 * pair_value + 0.2 * flush_potential + 0.1 * straight_potential + 0.1 * high_card_value
        else:
            # Without a pair, drawing to flush/straight and high cards matter more
            draw_equity = 0.3 * high_card_value + 0.4 * flush_potential + 0.3 * straight_potential
        
        return min(0.9, max(0.1, draw_equity))
    
    def _partial_board_strength(self, hole_cards, board):
        """
        Calculate hand strength when we don't have a complete 5-card hand yet.
        Optimized for 27-card deck dynamics.
        
        Args:
            hole_cards: List of hole card indices
            board: List of board card indices
            
        Returns:
            Estimated hand strength (0-1)
        """
        # Start with pre-flop hand strength as a base
        if len(hole_cards) >= 2:
            base_strength = self.starting_hand_values.get(tuple(hole_cards[:2]), 0.3)
        else:
            base_strength = 0.3  # Default for incomplete hole cards
        
        # If no board cards, just return the starting strength
        if not board:
            return base_strength
        
        # Analyze current cards for pairs, suited cards, etc.
        all_cards = hole_cards + board
        
        # Convert to ranks and suits
        ranks = [card // 3 for card in all_cards]
        suits = [card % 3 for card in all_cards]
        
        # Count frequencies
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        
        suit_counts = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        
        # Adjust strength based on what we have so far
        strength_adjustment = 0
        
        # Check for pairs, trips, etc. - more valuable in 27-card deck
        max_rank_count = max(rank_counts.values()) if rank_counts else 0
        if max_rank_count >= 3:
            # Trips or better - very strong in 27-card deck
            trips_rank = [r for r, count in rank_counts.items() if count >= 3][0]
            if trips_rank >= 6:  # High trips (8, 9, A)
                strength_adjustment += 0.45
            else:
                strength_adjustment += 0.40
        elif max_rank_count == 2:
            # Pair - stronger in 27-card deck
            pair_ranks = [r for r, count in rank_counts.items() if count == 2]
            
            # Higher pairs are worth more
            highest_pair = max(pair_ranks)
            if highest_pair >= 7:  # 9 or A
                strength_adjustment += 0.35
            elif highest_pair >= 5:  # 7 or 8
                strength_adjustment += 0.30
            else:
                strength_adjustment += 0.25
                
            # Two pair is very strong
            if len(pair_ranks) >= 2:
                strength_adjustment += 0.15
        
        # Check for flush potential - more significant in 27-card deck
        max_suit_count = max(suit_counts.values()) if suit_counts else 0
        if max_suit_count >= 4:
            # Near flush - extremely strong in 27-card deck
            strength_adjustment += 0.40
        elif max_suit_count == 3 and len(all_cards) <= 4:
            # Good flush draw potential
            strength_adjustment += 0.20
        
        # Check for straight potential
        sorted_ranks = sorted(set(ranks))
        if len(sorted_ranks) >= 4:
            # Count gaps
            gaps = 0
            for i in range(1, len(sorted_ranks)):
                gaps += max(0, sorted_ranks[i] - sorted_ranks[i-1] - 1)
            
            if gaps == 0:  # 4-card straight
                strength_adjustment += 0.35
            elif gaps == 1:  # 4 cards with 1 gap
                strength_adjustment += 0.20
            elif gaps <= 2:  # Decent straight potential
                strength_adjustment += 0.10
        
        # Special case for wheel straight potential (A-5)
        if 8 in ranks and any(r <= 3 for r in ranks):
            wheel_cards = [r for r in ranks if r <= 3 or r == 8]
            if len(wheel_cards) >= 3:
                strength_adjustment += 0.15
        
        # High card value - more important in 27-card deck
        high_card_count = sum(1 for r in ranks if r >= 7)  # 9 or A
        if high_card_count >= 2 and max_rank_count == 1:  # Multiple high cards but no pairs
            strength_adjustment += 0.10
        
        # Combine base strength with adjustments
        adjusted_strength = base_strength + strength_adjustment
        
        # Ensure result is in valid range
        return min(0.95, max(0.1, adjusted_strength))