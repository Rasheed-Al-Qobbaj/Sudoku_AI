# app.py
from flask import Flask, render_template, jsonify, request
import copy

from backend import generator
from backend import csp as solver
from backend import visualizing_solver

app = Flask(__name__)


@app.route('/')
def home():
    puzzle = generator.generate_puzzle(difficulty='easy')
    solution = copy.deepcopy(puzzle)
    solver.solve_board(solution)
    return render_template('index.html', puzzle=puzzle, solution=solution)


@app.route('/generate-puzzle')
def generate_puzzle_api():
    difficulty = request.args.get('difficulty', 'easy')
    puzzle = generator.generate_puzzle(difficulty=difficulty)
    solution = copy.deepcopy(puzzle)
    solver.solve_board(solution)
    return jsonify({'puzzle': puzzle, 'solution': solution})


@app.route('/visualize-solve', methods=['POST'])
def visualize_solve_api():
    data = request.get_json()
    board = data.get('board')
    if not board:
        return jsonify({'error': 'No board data provided'}), 400

    steps = visualizing_solver.get_solve_steps(board)
    return jsonify({'steps': steps})


@app.route('/ai-solver')
def ai_solver_page():
    return render_template('ai_solver.html')


@app.route('/about')
def about_page():
    return "This is where we'll explain how the project works!"


if __name__ == '__main__':
    app.run(debug=True)