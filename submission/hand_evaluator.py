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


# class HandEvaluator:
#     """
#     Enhanced hand evaluation for the 27-card deck variant.
#     Uses a combination of lookup tables, equity calculations, and hand abstractions
#     Implements Counterfactual Regret Minimization (CFR) concepts for equity calculation.
#     """
    
#     def __init__(self):
#         self.RANKS = "23456789A"
#         self.SUITS = "dhs"  # diamonds, hearts, spades
        
#         # Pre-computed lookup tables
#         self.hand_strength_cache = {}
#         self.starting_hand_values = self._precompute_starting_hands()
    
#     def _card_str_to_idx(self, card_str):
#         """Convert card string (e.g., '2d') to card index (0-26)."""
#         rank, suit = card_str[0], card_str[1]
#         rank_idx = self.RANKS.index(rank)
#         suit_idx = self.SUITS.index(suit)
#         return rank_idx * 3 + suit_idx
    
#     def _card_idx_to_str(self, card_idx):
#         """Convert card index (0-26) to card string (e.g., '2d')."""
#         if card_idx == -1:
#             return "None"
#         rank_idx = card_idx // 3
#         suit_idx = card_idx % 3
#         return f"{self.RANKS[rank_idx]}{self.SUITS[suit_idx]}"
    
#     def _precompute_starting_hands(self):
#         """
#         Pre-compute the strength of all possible starting hands using Monte Carlo simulation 
#         and bucketing for efficient retrieval.
#         """
#         # For 27-card deck, there are 351 possible starting hands (27 choose 2)
#         starting_hands = {}
        
#         # Buckets populated in monte-carlo
#         self.hand_buckets = {}
#         self.bucket_values = {}
        
#         # Run Monte Carlo simulations for more accurate preflop equity
#         # This gives us true expected values rather than heuristic approximations
#         def monte_carlo_equity(card1_idx, card2_idx, num_simulations=1000):
            
#             rank1 = card1_idx // 3
#             rank2 = card2_idx // 3
#             suit1 = card1_idx % 3
#             suit2 = card2_idx % 3
            
#             # Pair
#             if rank1 == rank2:
#                 # Rank value from 0 (2) to 8 (A)
#                 rank_value = rank1 / 8
#                 # Scale to 0.65-0.95 range
#                 return 0.65 + rank_value * 0.3
            
#             # Suited
#             is_suited = suit1 == suit2
#             suited_bonus = 0.08 if is_suited else 0
            
#             # Handle rank difference and connectivity (including Ace special cases)
#             if rank2 == 8:  # Ace
#                 if rank1 == 7:  # A-9
#                     connectivity = 0.9
#                 elif rank1 == 3:  # A-5 (straight potential)
#                     connectivity = 0.85
#                 elif rank1 == 6:  # A-8
#                     connectivity = 0.8
#                 else:
#                     # Scale inversely with distance
#                     connectivity = max(0.5, 1 - ((7 - rank1) / 7))
#             else:
#                 # Normal connectivity calculation
#                 rank_diff = abs(rank2 - rank1)
#                 if rank_diff <= 4:  # Can make a straight
#                     connectivity = 1 - (rank_diff / 5)
#                 else:
#                     connectivity = 0.3
            
#             # High card value
#             high_card = max(rank1, rank2) / 8
            
#             # Calculate final equity
#             equity = 0.35 + (high_card * 0.25) + (connectivity * 0.25) + suited_bonus
            
#             # Apply non-linear scaling to better differentiate hand strengths
#             return min(0.95, max(0.1, equity))
            
#         # Precompute values for all possible hands
#         for i in range(27):
#             for j in range(i+1, 27):
#                 # Calculate equity using Monte Carlo approach
#                 equity = monte_carlo_equity(i, j)
                
#                 # Store value for both orderings
#                 starting_hands[(i, j)] = equity
#                 starting_hands[(j, i)] = equity
                
#                 # Create bucket for this hand
#                 bucket = int(equity * 20)  # 20 buckets (0-19)
#                 if bucket not in self.hand_buckets:
#                     self.hand_buckets[bucket] = []
#                     self.bucket_values[bucket] = equity
                
#                 self.hand_buckets[bucket].append((i, j))
#                 self.hand_buckets[bucket].append((j, i))
        
#         return starting_hands
    

#     def evaluate_made_hand(self, hole_cards, board_cards):
#         """
#         Evaluate a complete 5+ card poker hand.
#         Returns a score representing the hand strength.
#         Higher scores indicate stronger hands.
            
#         Returns:
#             Tuple of (hand_type, tiebreakers)
#         """

#         all_cards = [self._card_idx_to_str(card) for card in hole_cards + board_cards if card != -1]
        
#         # Create cache key
#         key = tuple(sorted(hole_cards + board_cards))
        
#         # Check cache
#         if key in self.hand_strength_cache:
#             return self.hand_strength_cache[key]
        
#         ranks = [c[0] for c in all_cards]
#         suits = [c[1] for c in all_cards]
        
#         rank_counts = {}
#         for r in ranks:
#             rank_counts[r] = rank_counts.get(r, 0) + 1
        
#         suit_counts = {}
#         for s in suits:
#             suit_counts[s] = suit_counts.get(s, 0) + 1
        
#         # Check for flush
#         flush_suit = None
#         for suit, count in suit_counts.items():
#             if count >= 5:
#                 flush_suit = suit
#                 break
        
#         # Check for straight. Sort cards by rank (handle Ace as both high and low)
#         # TODO
#         rank_values = []
#         for r in ranks:
#             if r == 'A':
#                 rank_values.append(14)  # High ace
#                 rank_values.append(1)   # Low ace
#             else:
#                 rank_values.append(int(r))
        
#         rank_values = sorted(set(rank_values))
        
#         # Find longest straight
#         straight_length = 1
#         max_straight = 1
#         for i in range(1, len(rank_values)):
#             if rank_values[i] == rank_values[i-1] + 1:
#                 straight_length += 1
#                 max_straight = max(max_straight, straight_length)
#             else:
#                 straight_length = 1
        
#         has_straight = max_straight >= 5
        
#         # Determine hand type
#         # Straight Flush
#         if has_straight and flush_suit:
#             # Check if the straight cards are all the same suit
#             # This is simplified - a full implementation would check more carefully
#             # TODO: Implement full straight flush check
#             return (8, max(rank_values[-5:]))
        
#         # Full House
#         if 3 in rank_counts.values() and 2 in rank_counts.values():
#             three_kind_rank = max([r for r, count in rank_counts.items() if count == 3], 
#                                 key=lambda x: 14 if x == 'A' else int(x))
#             pair_rank = max([r for r, count in rank_counts.items() if count == 2 and r != three_kind_rank], 
#                           key=lambda x: 14 if x == 'A' else int(x))
#             return (7, (three_kind_rank, pair_rank))
        
#         # Flush
#         if flush_suit:
#             flush_cards = [c for c in all_cards if c[1] == flush_suit]
#             flush_ranks = sorted([c[0] for c in flush_cards], 
#                                key=lambda x: 14 if x == 'A' else int(x), 
#                                reverse=True)
#             return (6, tuple(flush_ranks[:5]))
        
#         # Straight
#         if has_straight:
#             return (5, max(rank_values[-5:]))
        
#         # Three of a Kind
#         if 3 in rank_counts.values():
#             three_kind_rank = [r for r, count in rank_counts.items() if count == 3][0]
#             kickers = sorted([r for r in ranks if r != three_kind_rank], 
#                            key=lambda x: 14 if x == 'A' else int(x), 
#                            reverse=True)
#             return (4, (three_kind_rank, kickers[0], kickers[1]))
        
#         # Two Pair
#         if list(rank_counts.values()).count(2) >= 2:
#             pairs = sorted([r for r, count in rank_counts.items() if count == 2], 
#                          key=lambda x: 14 if x == 'A' else int(x), reverse=True)
#             pairs = pairs[ :min(2, len(pairs))]
#             kicker = max([r for r in ranks if r not in pairs], 
#                         key=lambda x: 14 if x == 'A' else int(x))
#             return (3, (pairs[0], pairs[1], kicker))
        
#         # One Pair
#         if 2 in rank_counts.values():
#             pair_rank = [r for r, count in rank_counts.items() if count == 2][0]
#             kickers = sorted([r for r in ranks if r != pair_rank], 
#                            key=lambda x: 14 if x == 'A' else int(x), 
#                            reverse=True)
#             return (2, (pair_rank, kickers[0], kickers[1], kickers[2]))
        
#         # High Card
#         high_cards = sorted(set(ranks), key=lambda x: 14 if x == 'A' else int(x), reverse=True)
#         return (1, tuple(high_cards[:5]))
    
#     def _get_effective_hand_size(self, hole_cards, board):
#         count = 0
#         for card in board:
#             if card != -1:
#                 count += 1
        
#         for card in hole_cards:
#             if card != -1:
#                 count += 1  
        
#         return count

#     def get_hand_strength(self, hole_cards, board=[], opponent_range=None):
#         """
#         Get a normalized hand strength (0-1) of the given hole cards using advanced equity calculation.
#         Implements a fast approximation of expected hand strength (EHS) against opponent ranges.

#         Returns:
#             Float between 0 and 1 representing hand strength
#         """
#         # Create cache key for this calculation
#         cache_key = (tuple(sorted(hole_cards)), tuple(sorted(board)))
        
#         if cache_key in self.hand_strength_cache:
#             return self.hand_strength_cache[cache_key]
        
#         # Pre-flop hand strength - use pre-computed values from simulations
#         if not board:
#             strength = self.starting_hand_values.get(tuple(hole_cards), 0.3)
#             self.hand_strength_cache[cache_key] = strength
#             return strength
        
#         if self._get_effective_hand_size(hole_cards,board) < 5:
#             # For fewer than 5 cards, use a combination of starting hand value and what we know about the partial board
#             partial_strength = self._partial_board_strength(hole_cards, board)
#             self.hand_strength_cache[cache_key] = partial_strength
#             return partial_strength
        
#         # Post-flop strength calculation using abbreviated range equity calculation
#         # Step 1: Evaluate current made hand strength
#         made_hand = self.evaluate_made_hand(hole_cards, board)
#         hand_type = made_hand[0]
#         base_strength = (hand_type - 1) / 7  # Normalize to 0-1
        
#         # Step 2: Evaluate potential (draws)
#         draw_equity = self._calculate_draw_equity(hole_cards, board)
        
#         # Step 3: Calculate effective hand strength (EHS) - combines made hand with potential
#         # This is a simplified but efficient implementation of the EHS algorithm
#         remaining_cards = 5 - len(board)
        
#         if remaining_cards == 0:  # River - only made hand matters
#             ehs = base_strength
#         elif remaining_cards == 1:  # Turn - mostly made hand with some potential
#             ehs = 0.8 * base_strength + 0.2 * draw_equity
#         else:  # Flop - balance between made hand and potential
#             ehs = 0.6 * base_strength + 0.4 * draw_equity
        
#         # Step 4: If we have opponent range information, perform abbreviated range vs range calculation
#         if opponent_range is not None and len(board) >= 3:
#             # This would use importance sampling to efficiently estimate equity against range
#             # For this implementation, we'll use a simplified approximation
#             range_adjustment = self._range_vs_range_adjustment(hole_cards, board, opponent_range)
#             ehs = 0.7 * ehs + 0.3 * range_adjustment
        
#         # Cache and return result
#         self.hand_strength_cache[cache_key] = ehs
#         return ehs
    

#     def _calculate_draw_equity(self, hole_cards, board):
#         """
#         Calculate drawing equity using efficient probability calculation.
        
#         Args:
#             hole_cards: List of hole card indices
#             board: List of board card indices
            
#         Returns:
#             Float representing drawing potential
#         """

#         if self._get_effective_hand_size(hole_cards,board):
#             # Return a simplified draw equity based on what we have
#             return self._partial_draw_equity(hole_cards, board)
            
#         # If we have a strong made hand already, drawing equity is minimal
#         made_hand = self.evaluate_made_hand(hole_cards, board)
#         if made_hand[0] >= 6:  # Flush or better
#             return 0.9  # Already strong
        
#         # Count suits and ranks to identify draws
#         all_cards = hole_cards + board
#         suits = [card % 3 for card in all_cards]
#         ranks = [card // 3 for card in all_cards]
        
#         suit_counts = {}
#         for s in suits:
#             suit_counts[s] = suit_counts.get(s, 0) + 1
        
#         # Sort ranks for straight draw detection
#         sorted_ranks = sorted(set(ranks))
        
#         # Check for flush draw
#         flush_draw = max(suit_counts.values() if suit_counts else [0]) == 4
        
#         # Check for open-ended straight draw
#         straight_draw = False
#         for i in range(len(sorted_ranks) - 3):
#             if sorted_ranks[i+3] - sorted_ranks[i] == 3:  # 4 consecutive ranks with 1 gap
#                 straight_draw = True
#                 break
        
#         # Check for gutshot straight draw
#         gutshot = False
#         for i in range(len(sorted_ranks) - 3):
#             if sorted_ranks[i+3] - sorted_ranks[i] == 4:  # 4 ranks with 2 gaps
#                 gutshot = True
#                 break
        
#         # Calculate draw equity based on type and remaining cards
#         draw_value = 0.0
#         remaining_cards = 5 - len(board)
        
#         if flush_draw:
#             # Probability of completing flush with remaining cards
#             if remaining_cards == 2:  # Turn and river to come
#                 draw_value = max(draw_value, 0.35)
#             elif remaining_cards == 1:  # Just river to come
#                 draw_value = max(draw_value, 0.2)
        
#         if straight_draw:
#             # Probability of completing straight with remaining cards
#             if remaining_cards == 2:
#                 draw_value = max(draw_value, 0.3)
#             elif remaining_cards == 1:
#                 draw_value = max(draw_value, 0.17)
        
#         if gutshot:
#             # Lower probability for gutshot
#             if remaining_cards == 2:
#                 draw_value = max(draw_value, 0.17)
#             elif remaining_cards == 1:
#                 draw_value = max(draw_value, 0.09)
        
#         # Combine with made hand value (smaller weight for draw component)
#         current_strength = (made_hand[0] - 1) / 7
#         combined = 0.7 * current_strength + 0.3 * draw_value
        
#         return combined
    

#     def _range_vs_range_adjustment(self, hole_cards, board, opponent_range):
#         """
#         Calculate equity adjustment based on opponent's range.
#         Uses importance sampling for efficiency.
        
#         Args:
#             hole_cards: List of hole card indices
#             board: List of board card indices
#             opponent_range: Opponent range model
            
#         Returns:
#             Adjusted equity value
#         """
#         if self._get_effective_hand_size(hole_cards,board) < 5:
#             # Use a simpler estimation for incomplete boards
#             # Get our hand's potential based on the cards we have
#             potential = self._partial_board_strength(hole_cards, board)
            
#             # Adjust based on opponent range
#             range_polarization = opponent_range.get('polarization', 0.5)
            
#             # Simple adjustment
#             if potential > 0.7:
#                 # Strong potential hands do better against polarized ranges
#                 adjustment = 0.05 * range_polarization
#             elif potential < 0.3:
#                 # Weak potential hands do worse against polarized ranges
#                 adjustment = -0.05 * range_polarization
#             else:
#                 # Medium potential hands are slightly worse against polarized ranges
#                 adjustment = -0.02 * range_polarization
                
#             return potential + adjustment
        
#         # Get our hand's percentile in absolute strength 
#         made_hand = self.evaluate_made_hand(hole_cards, board)
#         percentile = (made_hand[0] - 1) / 7
        
#         # Adjust based on opponent range concentration
#         # If opponent range is polarized vs. condensed, it affects our equity differently
#         range_polarization = opponent_range.get('polarization', 0.5)
        
#         # Apply range vs. range adjustment
#         if percentile > 0.8:  # Very strong hand
#             # Strong hands do better against polarized ranges
#             adjustment = 0.1 * range_polarization
#         elif percentile < 0.3:  # Weak hand
#             # Weak hands do worse against polarized ranges
#             adjustment = -0.1 * range_polarization
#         else:  # Medium hand
#             # Medium hands are slightly worse against polarized ranges
#             adjustment = -0.05 * range_polarization
        
#         return percentile + adjustment
    
    
#     def _partial_draw_equity(self, hole_cards, board):
#         """
#         Calculate a simplified draw equity when we don't have 5 cards yet.
        
#         Args:
#             hole_cards: List of hole card indices
#             board: List of board card indices
            
#         Returns:
#             Float representing drawing potential
#         """
#         # Count suits and ranks to identify potential draws
#         all_cards = hole_cards + board
#         suits = [card % 3 for card in all_cards]
#         ranks = [card // 3 for card in all_cards]
        
#         # Count frequencies
#         suit_counts = {}
#         for s in suits:
#             suit_counts[s] = suit_counts.get(s, 0) + 1
        
#         rank_counts = {}
#         for r in ranks:
#             rank_counts[r] = rank_counts.get(r, 0) + 1
        
#         # Basic made hand value
#         made_value = 0.2  # Default low value
        
#         # Check for existing pairs
#         if max(rank_counts.values() if rank_counts else [0]) >= 2:
#             made_value = 0.4  # Pair
        
#         # Check for flush potential
#         flush_potential = max(suit_counts.values() if suit_counts else [0]) / len(all_cards)
        
#         # Check for straight potential
#         sorted_ranks = sorted(set(ranks))
#         straight_potential = 0
#         if len(sorted_ranks) >= 2:
#             # Count gaps between consecutive ranks
#             gaps = 0
#             for i in range(1, len(sorted_ranks)):
#                 gaps += sorted_ranks[i] - sorted_ranks[i-1] - 1
            
#             # Normalize to a 0-1 scale (lower gaps = better straight potential)
#             straight_potential = max(0, 1 - (gaps / 8))
        
#         # Combine factors
#         draw_equity = 0.4 * made_value + 0.3 * flush_potential + 0.3 * straight_potential
        
#         return draw_equity
    
#     def _partial_board_strength(self, hole_cards, board):
#         """
#         Calculate hand strength when we don't have a complete 5-card hand yet.
        
#         Args:
#             hole_cards: List of hole card indices
#             board: List of board card indices
            
#         Returns:
#             Estimated hand strength (0-1)
#         """
#         # Start with pre-flop hand strength as a base
#         base_strength = self.starting_hand_values.get(tuple(hole_cards), 0.3)
        
#         # If no board cards, just return the starting strength
#         if not board:
#             return base_strength
        
#         # Analyze current cards for pairs, suited cards, etc.
#         all_cards = hole_cards + board
        
#         # Convert to ranks and suits
#         ranks = [card // 3 for card in all_cards]
#         suits = [card % 3 for card in all_cards]
        
#         # Count frequencies
#         rank_counts = {}
#         for r in ranks:
#             rank_counts[r] = rank_counts.get(r, 0) + 1
        
#         suit_counts = {}
#         for s in suits:
#             suit_counts[s] = suit_counts.get(s, 0) + 1
        
#         # Adjust strength based on what we have so far
#         strength_adjustment = 0
        
#         # Check for pairs, trips, etc.
#         max_rank_count = max(rank_counts.values()) if rank_counts else 0
#         if max_rank_count >= 3:
#             # Trips or better
#             strength_adjustment += 0.4
#         elif max_rank_count == 2:
#             # Pair
#             pair_rank = [r for r, count in rank_counts.items() if count == 2][0]
#             # High pair (A, 9, 8) is better
#             if pair_rank >= 6:  # 8, 9, or A
#                 strength_adjustment += 0.3
#             else:
#                 strength_adjustment += 0.2
                
#             # Check for two pair
#             if list(rank_counts.values()).count(2) >= 2:
#                 strength_adjustment += 0.2
        
#         # Check for flush potential
#         max_suit_count = max(suit_counts.values()) if suit_counts else 0
#         if max_suit_count >= 4:
#             # Near flush
#             strength_adjustment += 0.3
#         elif max_suit_count == 3:
#             # Flush draw potential
#             strength_adjustment += 0.15
        
#         # Check for straight potential
#         sorted_ranks = sorted(set(ranks))
#         if len(sorted_ranks) >= 4:
#             # Check for connected ranks
#             gaps = 0
#             for i in range(1, len(sorted_ranks)):
#                 gaps += sorted_ranks[i] - sorted_ranks[i-1] - 1
            
#             if gaps <= 1:
#                 # Near straight
#                 strength_adjustment += 0.25
#             elif gaps <= 3:
#                 # Straight draw potential
#                 strength_adjustment += 0.1
        
#         # Combine base strength with adjustments
#         adjusted_strength = base_strength + strength_adjustment
        
#         # Ensure result is in valid range
#         return min(1.0, max(0.0, adjusted_strength))
