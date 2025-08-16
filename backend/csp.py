from ortools.sat.python import cp_model

def solve_board(board):
    """Solves the sudoku board using a CP-SAT solver."""
    model = cp_model.CpModel()

    # --- Create variables ---
    # Create a 9x9 grid of integer variables, with each cell taking a value from 1 to 9.
    grid = {}
    for r in range(9):
        for c in range(9):
            grid[(r, c)] = model.NewIntVar(1, 9, f'grid_{r}_{c}')

    # --- Create constraints ---
    # All cells in a row must be different.
    for r in range(9):
        model.AddAllDifferent([grid[(r, c)] for c in range(9)])

    # All cells in a column must be different.
    for c in range(9):
        model.AddAllDifferent([grid[(r, c)] for r in range(9)])

    # All cells in a 3x3 box must be different.
    for box_r in range(3):
        for box_c in range(3):
            box = []
            for r in range(box_r * 3, box_r * 3 + 3):
                for c in range(box_c * 3, box_c * 3 + 3):
                    box.append(grid[(r, c)])
            model.AddAllDifferent(box)

    # --- Add initial puzzle values as constraints ---
    for r in range(9):
        for c in range(9):
            if board[r][c] != 0:
                model.Add(grid[(r, c)] == board[r][c])

    # --- Solve the model ---
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # --- Extract the solution ---
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        for r in range(9):
            for c in range(9):
                board[r][c] = solver.Value(grid[(r,c)])
        return True
    else:
        return False # No solution found