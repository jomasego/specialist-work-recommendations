from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Backend for Specialist Work Recommendations AI is running!"

if __name__ == '__main__':
    app.run(debug=True)
