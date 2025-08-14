from itertools import combinations

# Assign scores to each technique. Higher means more difficult.
TECHNIQUES = {
    'naked_single': 1,
    'hidden_single': 1.2,
    'naked_pair': 2,
    'hidden_pair': 2.5,
    'naked_triple': 2.6,
    'hidden_triple': 2.8,
    'locked_candidates': 3,
    'x_wing': 3.5,
}


# --- Pre-calculate units and peers for efficiency ---
def get_box_cells(box_id):
    start_row, start_col = (box_id // 3) * 3, (box_id % 3) * 3
    return [(r, c) for r in range(start_row, start_row + 3) for c in range(start_col, start_col + 3)]


ROWS = [[(r, c) for c in range(9)] for r in range(9)]
COLS = [[(r, c) for r in range(9)] for c in range(9)]
BOXES = [get_box_cells(b) for b in range(9)]
UNITS = ROWS + COLS + BOXES
CELLS = [(r, c) for r in range(9) for c in range(9)]
PEERS = {cell: (set(ROWS[cell[0]]) | set(COLS[cell[1]]) | set(BOXES[(cell[0] // 3) * 3 + cell[1] // 3])) - {cell} for
         cell in CELLS}


# --- Main Rater Logic ---
def rate_puzzle(board):
    try:
        candidates = get_initial_candidates(board)
    except ValueError:
        return "Invalid", -1

    # Store a log of techniques used
    technique_log = []

    # The techniques are tried in order of difficulty.
    solver_functions = [
        apply_singles,
        apply_locked_candidates,
        apply_naked_subsets,
        apply_hidden_subsets,
        apply_x_wing
    ]

    while True:
        # Count candidates to check for progress
        candidates_before = sum(len(candidates[r][c]) for r, c in CELLS)

        # If solved, we're done
        if candidates_before == 81:
            if not technique_log: return "Easy", 1
            max_difficulty_score = max(technique_log)
            if max_difficulty_score <= 1.2: return "Easy", max_difficulty_score
            if max_difficulty_score <= 2.8: return "Medium", max_difficulty_score
            if max_difficulty_score <= 3.4: return "Hard", max_difficulty_score
            return "Extreme", max_difficulty_score

        # Apply techniques one by one
        made_progress = False
        for solve_func in solver_functions:
            candidates, changes_made, technique_name, technique_score = solve_func(candidates)
            if changes_made:
                technique_log.append(technique_score)
                made_progress = True
                break  # Restart the loop to try simple techniques again

        # If no technique made any progress, the puzzle is unsolveable
        if not made_progress:
            return "Unsolvable / Requires Guessing", 99


# --- Candidate Management ---
def get_initial_candidates(board):
    candidates = [[set(range(1, 10)) for _ in range(9)] for _ in range(9)]
    for r in range(9):
        for c in range(9):
            if board[r][c] != 0:
                if not assign(candidates, r, c, board[r][c]):
                    raise ValueError("Invalid puzzle state during initialization")
    return candidates


def assign(candidates, r, c, val):
    """Assigns a value to a cell and propagates the constraint via elimination."""
    for v in list(candidates[r][c] - {val}):
        if not eliminate(candidates, r, c, v):
            return False
    return True


def eliminate(candidates, r, c, val):
    """Eliminates a candidate and propagates constraints."""
    if val not in candidates[r][c]:
        return True  # Already eliminated

    candidates[r][c].remove(val)

    # If a cell is reduced to one candidate, it's a Naked Single. Assign it.
    if len(candidates[r][c]) == 1:
        single_val = list(candidates[r][c])[0]
        for peer_r, peer_c in PEERS[(r, c)]:
            if not eliminate(candidates, peer_r, peer_c, single_val):
                return False
    # Check for Hidden Singles in the units this cell belongs to
    for unit in [u for u in UNITS if (r, c) in u]:
        d_places = [cell for cell in unit if val in candidates[cell[0]][cell[1]]]
        if not d_places: return False  # Contradiction
        if len(d_places) == 1:
            if not assign(candidates, d_places[0][0], d_places[0][1], val):
                return False
    return True


# --- Logical Solving Techniques ---
def apply_singles(candidates):
    return candidates, False, 'single', 0


def apply_locked_candidates(candidates):
    """Finds Locked Candidates (Pointing and Claiming)."""
    for unit in BOXES:
        for val in range(1, 10):
            places = [cell for cell in unit if val in candidates[cell[0]][cell[1]]]
            if len(places) > 1:
                # Check if all places for 'val' in this box lie on the same row or column
                rows = {r for r, c in places}
                cols = {c for r, c in places}
                if len(rows) == 1:  # Pointing (Row)
                    row = rows.pop()
                    for c_peer in range(9):
                        if (row, c_peer) not in unit and val in candidates[row][c_peer]:
                            if eliminate(candidates, row, c_peer, val):
                                return candidates, True, 'locked_candidates', TECHNIQUES['locked_candidates']
                if len(cols) == 1:  # Pointing (Col)
                    col = cols.pop()
                    for r_peer in range(9):
                        if (r_peer, col) not in unit and val in candidates[r_peer][col]:
                            if eliminate(candidates, r_peer, col, val):
                                return candidates, True, 'locked_candidates', TECHNIQUES['locked_candidates']
    return candidates, False, 'locked_candidates', 0


def apply_naked_subsets(candidates):
    """Finds Naked Pairs/Triples, applies first, and returns."""
    for unit in UNITS:
        for size in [2, 3]:
            # Find cells with 2 or 'size' candidates
            candidate_cells = [cell for cell in unit if 2 <= len(candidates[cell[0]][cell[1]]) <= size]
            if len(candidate_cells) < size: continue

            for group in combinations(candidate_cells, size):
                all_cands_in_group = set.union(*(candidates[r][c] for r, c in group))
                if len(all_cands_in_group) == size:
                    # Naked subset found. Try to eliminate from peers in the unit.
                    peers_to_check = set(unit) - set(group)
                    for r_peer, c_peer in peers_to_check:
                        common_cands = candidates[r_peer][c_peer].intersection(all_cands_in_group)
                        if common_cands:
                            for val in common_cands:
                                if eliminate(candidates, r_peer, c_peer, val):
                                    name = 'naked_pair' if size == 2 else 'naked_triple'
                                    return candidates, True, name, TECHNIQUES[name]
    return candidates, False, 'naked_subset', 0


def apply_hidden_subsets(candidates):
    """Finds Hidden Pairs/Triples, applies first, and returns."""
    for unit in UNITS:
        for size in [2, 3]:
            unassigned_cells = [cell for cell in unit if len(candidates[cell[0]][cell[1]]) > 1]
            if len(unassigned_cells) <= size: continue

            for cand_group in combinations(range(1, 10), size):
                places = {cell for cell in unassigned_cells if
                          any(c in candidates[cell[0]][cell[1]] for c in cand_group)}

                val_places = {cell for cell in unassigned_cells for val in cand_group if
                              val in candidates[cell[0]][cell[1]]}

                if len(val_places) == size:
                    # Hidden subset found. The numbers in cand_group only appear in val_places.
                    # Eliminate other candidates from these cells.
                    for r, c in val_places:
                        cands_to_remove = candidates[r][c] - set(cand_group)
                        if cands_to_remove:
                            for val in cands_to_remove:
                                if eliminate(candidates, r, c, val):
                                    name = 'hidden_pair' if size == 2 else 'hidden_triple'
                                    return candidates, True, name, TECHNIQUES[name]
    return candidates, False, 'hidden_subset', 0


def apply_x_wing(candidates):
    """Finds X-Wings across rows and columns, applies first, and returns."""
    for val in range(1, 10):
        # Row-based X-Wing
        rows_with_two = {r: [c for c in range(9) if val in candidates[r][c]] for r in range(9)}
        rows_with_two = {r: places for r, places in rows_with_two.items() if len(places) == 2}

        if len(rows_with_two) < 2: continue
        for r1, r2 in combinations(rows_with_two.keys(), 2):
            if rows_with_two[r1] == rows_with_two[r2]:
                cols = rows_with_two[r1]
                # X-Wing found. Eliminate val from other cells in these columns.
                for c in cols:
                    for r_peer in range(9):
                        if r_peer not in [r1, r2] and val in candidates[r_peer][c]:
                            if eliminate(candidates, r_peer, c, val):
                                return candidates, True, 'x_wing', TECHNIQUES['x_wing']
    return candidates, False, 'x_wing', 0