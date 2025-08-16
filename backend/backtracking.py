def find_empty(board):
    """Finds the next empty cell (represented by 0)"""
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return (i, j)  # row, col
    return None

def is_valid(board, num, pos):
    """Checks if a number is valid in a given position"""
    # Check row
    for i in range(9):
        if board[pos[0]][i] == num and pos[1] != i:
            return False

    # Check column
    for i in range(9):
        if board[i][pos[1]] == num and pos[0] != i:
            return False

    # Check 3x3 box
    box_x = pos[1] // 3
    box_y = pos[0] // 3
    for i in range(box_y * 3, box_y * 3 + 3):
        for j in range(box_x * 3, box_x * 3 + 3):
            if board[i][j] == num and (i, j) != pos:
                return False

    return True

def solve_board(board):
    """Solves the sudoku board using backtracking"""
    find = find_empty(board)
    if not find:
        return True  # Board is solved
    else:
        row, col = find

    for num in range(1, 10):
        if is_valid(board, num, (row, col)):
            board[row][col] = num

            if solve_board(board):
                return True

            board[row][col] = 0  # Backtrack

    return False

def count_solutions(board):
    """Recursively counts the number of solutions for a given board."""
    find = find_empty(board)
    if not find:
        # No empty cells means we found one complete solution
        return 1

    row, col = find
    count = 0

    for num in range(1, 10):
        if is_valid(board, num, (row, col)):
            board[row][col] = num
            count += count_solutions(board)
            # Crucially, don't stop after finding one solution.
            # Continue searching for more.

            # Backtrack to explore other possibilities
            board[row][col] = 0

            # If we found more than one solution, we can stop early
            if count > 1:
                return count

    return count