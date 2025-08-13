# A Node represents a '1' in the sparse matrix
class Node:
    def __init__(self, col_header, row_idx):
        self.col = col_header
        self.row_idx = row_idx
        # Pointers for the toroidal doubly-linked list
        self.left = self.right = self.up = self.down = self

    def link_down(self, node):
        node.down = self.down
        node.down.up = node
        node.up = self
        self.down = node


# A Column object represents a constraint
class Column(Node):
    def __init__(self, name=""):
        super().__init__(self, -1)  # A column header's row index is -1
        self.size = 0
        self.name = name

    def cover(self):
        # "Hide" this column by removing it from the header list
        self.right.left = self.left
        self.left.right = self.right

        # Move down the column, covering all rows that have a '1' in this column
        i = self.down
        while i != self:
            j = i.right
            while j != i:
                j.down.up = j.up
                j.up.down = j.down
                j.col.size -= 1
                j = j.right
            i = i.down

    def uncover(self):
        # "Un-hide" this column by re-inserting it into the header list
        i = self.up
        while i != self:
            j = i.left
            while j != i:
                j.col.size += 1
                j.down.up = j
                j.up.down = j
                j = j.left
            i = i.up
        self.right.left = self
        self.left.right = self


class SudokuDL:
    def __init__(self, board):
        self.board = board
        self.root = Column("root")
        self.solution = []
        self.solution_board = [[0] * 9 for _ in range(9)]

        self._build_matrix()
        self._add_initial_values()

    def _build_matrix(self):
        headers = [self.root]
        # Create 324 column headers for our 324 constraints
        for i in range(324):
            new_col = Column(str(i))
            headers.append(new_col)
            # Link the new column to the previous one
            new_col.left = headers[-2]
            headers[-2].right = new_col
        self.root.left = headers[-1]
        headers[-1].right = self.root

        # Create the 729 rows representing each possible choice
        for r in range(9):
            for c in range(9):
                for num in range(1, 10):
                    row_idx = (r * 81) + (c * 9) + (num - 1)
                    # This choice satisfies 4 constraints. Find their column indices.
                    box = (r // 3) * 3 + (c // 3)

                    # 1. Cell constraint (r, c)
                    p1 = r * 9 + c
                    # 2. Row-Number constraint (r, num)
                    p2 = 81 + (r * 9) + (num - 1)
                    # 3. Col-Number constraint (c, num)
                    p3 = 162 + (c * 9) + (num - 1)
                    # 4. Box-Number constraint (b, num)
                    p4 = 243 + (box * 9) + (num - 1)

                    # Link the 4 nodes for this choice into the matrix
                    n1 = Node(headers[p1 + 1], row_idx)
                    n2 = Node(headers[p2 + 1], row_idx)
                    n3 = Node(headers[p3 + 1], row_idx)
                    n4 = Node(headers[p4 + 1], row_idx)

                    n1.right, n1.left = n2, n4
                    n2.right, n2.left = n3, n1
                    n3.right, n3.left = n4, n2
                    n4.right, n4.left = n1, n3

                    headers[p1 + 1].link_down(n1)
                    headers[p1 + 1].size += 1
                    headers[p2 + 1].link_down(n2)
                    headers[p2 + 1].size += 1
                    headers[p3 + 1].link_down(n3)
                    headers[p3 + 1].size += 1
                    headers[p4 + 1].link_down(n4)
                    headers[p4 + 1].size += 1

    def _add_initial_values(self):
        for r in range(9):
            for c in range(9):
                num = self.board[r][c]
                if num != 0:
                    # If there's an initial number, we must select the corresponding row
                    row_idx = (r * 81) + (c * 9) + (num - 1)

                    # Cover all columns satisfied by this choice
                    col_to_cover = None
                    i = self.root.right
                    while i != self.root:
                        j = i.down
                        while j != i:
                            if j.row_idx == row_idx:
                                # This is the row we need to pre-select
                                col_to_cover = i
                                break
                            j = j.down
                        if col_to_cover:
                            break
                        i = i.right

                    if col_to_cover:
                        # Cover the primary column first
                        col_to_cover.cover()
                        self.solution.append(col_to_cover.down)  # Add the node to our solution

                        # Cover all other columns satisfied by this node's row
                        k = col_to_cover.down.right
                        while k != col_to_cover.down:
                            k.col.cover()
                            k = k.right

    def _search(self):
        if self.root.right == self.root:
            # All constraints satisfied, solution found
            self._map_solution_to_board()
            return True

        # Heuristic: choose column with smallest size
        c = None
        s = float('inf')
        j = self.root.right
        while j != self.root:
            if j.size < s:
                s = j.size
                c = j
            j = j.right

        if c is None or c.size == 0:
            return False  # Dead end

        c.cover()

        r = c.down
        while r != c:
            self.solution.append(r)

            # Cover other columns satisfied by this row
            k = r.right
            while k != r:
                k.col.cover()
                k = k.right

            if self._search():
                return True

            # Backtrack
            self.solution.pop()

            k = r.left
            while k != r:
                k.col.uncover()
                k = k.left

            r = r.down

        c.uncover()
        return False

    def _map_solution_to_board(self):
        for node in self.solution:
            row_idx = node.row_idx
            r = row_idx // 81
            c = (row_idx // 9) % 9
            num = (row_idx % 9) + 1
            self.solution_board[r][c] = num

    def solve(self):
        if self._search():
            for i in range(9):
                for j in range(9):
                    self.board[i][j] = self.solution_board[i][j]
            return True
        return False


def solve_board(board):
    solver = SudokuDL(board)
    return solver.solve()