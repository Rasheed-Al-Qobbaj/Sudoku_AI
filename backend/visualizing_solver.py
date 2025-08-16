def find_empty(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return (i, j)
    return None


def is_valid(board, num, pos):
    # Check row
    if any(board[pos[0]][i] == num for i in range(9)):
        return False
    # Check column
    if any(board[i][pos[1]] == num for i in range(9)):
        return False
    # Check box
    box_x, box_y = pos[1] // 3, pos[0] // 3
    for i in range(box_y * 3, box_y * 3 + 3):
        for j in range(box_x * 3, box_x * 3 + 3):
            if board[i][j] == num:
                return False
    return True


def solve_and_visualize(board):
    """A generator function that yields each step of the backtracking process."""
    find = find_empty(board)
    if not find:
        yield {'status': 'solved'}
        return True

    row, col = find
    for num in range(1, 10):
        if is_valid(board, num, (row, col)):
            board[row][col] = num
            yield {'row': row, 'col': col, 'val': num, 'type': 'trying'}

            if solve_and_visualize(board):
                if next(solve_and_visualize(board), {}).get('status') == 'solved':
                    yield {'status': 'solved'}
                    return True

            board[row][col] = 0
            yield {'row': row, 'col': col, 'val': 0, 'type': 'backtrack'}

    return False


def get_solve_steps(board):
    """Runs the visualizing solver and collects all the steps in a list."""
    steps = []
    board_copy = [row[:] for row in board]
    solver_generator = solve_and_visualize(board_copy)

    for step in solver_generator:
        steps.append(step)
        if step.get('status') == 'solved':
            break

    return steps