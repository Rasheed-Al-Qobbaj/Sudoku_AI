import random
import copy
from backend.csp import solve_board as solve_with_cp
from backend.backtracking import count_solutions


def generate_puzzle(difficulty='medium'):
    """
    Generates a new Sudoku puzzle with a unique solution.

    1. Creates a full, valid Sudoku board.
    2. Removes numbers one by one while ensuring the solution remains unique.
    """
    # 1. Create a full, valid solution
    # Start with an empty board and let the CP solver fill it.
    base_board = [[0] * 9 for _ in range(9)]
    solve_with_cp(base_board)

    # Set the number of cells to remove based on difficulty
    if difficulty == 'easy':
        attempts = 35
    elif difficulty == 'hard':
        attempts = 55
    else:  # medium
        attempts = 45

    puzzle = copy.deepcopy(base_board)
    cells = list(range(81))
    random.shuffle(cells)  # Get a random order to remove cells

    # 2. "Poke holes" in the grid
    removed_count = 0
    for cell_index in cells:
        if removed_count >= attempts:
            break

        row = cell_index // 9
        col = cell_index % 9

        if puzzle[row][col] == 0:
            continue

        # Temporarily remove the number
        temp = puzzle[row][col]
        puzzle[row][col] = 0

        # Create a copy to test for a unique solution
        board_to_test = copy.deepcopy(puzzle)

        # 3. Check if the solution is still unique
        if count_solutions(board_to_test) != 1:
            # If not unique, put the number back
            puzzle[row][col] = temp
        else:
            removed_count += 1

    return puzzle