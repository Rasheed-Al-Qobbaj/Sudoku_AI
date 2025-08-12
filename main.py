import time
import copy

import backtracking
from puzzles import puzzle_easy, puzzle_hard, print_board

solvers = {
    "Naive Backtracking": backtracking.solve_board,
}

puzzles = {
    "Easy Puzzle": puzzle_easy,
    "Hard Puzzle": puzzle_hard
}

if __name__ == '__main__':
    for puzzle_name, puzzle in puzzles.items():
        print(f"--- Solving: {puzzle_name} ---")
        print("Initial board:")
        print_board(puzzle)
        print("\n" + "="*30 + "\n")

        for solver_name, solver_func in solvers.items():
            # We need a deep copy of the puzzle for each solver
            # so they don't solve an already completed board.
            board_to_solve = copy.deepcopy(puzzle)

            print(f"-> Running {solver_name}...")
            start_time = time.time()
            solved = solver_func(board_to_solve)
            end_time = time.time()
            duration = (end_time - start_time) * 1000 # in milliseconds

            if solved:
                print("Final board:")
                print_board(board_to_solve)
                print(f"SUCCESS: Solved in {duration:.4f} ms\n")
            else:
                print(f"FAILURE or NOT IMPLEMENTED: Could not solve. Time: {duration:.4f} ms\n")

            print("-" * 30)

        print("\n\n")