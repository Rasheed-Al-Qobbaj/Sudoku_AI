from flask import Flask, render_template


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/ai-solver')
def ai_solver_page():
    return "This is where the AI solver will go!"

@app.route('/about')
def about_page():
    return "This is where we'll explain how the project works!"


if __name__ == '__main__':
    app.run(debug=True)