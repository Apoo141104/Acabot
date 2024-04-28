from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat-container')
def chat_container():
    return render_template('chat-container.html')

if __name__ == '__main__':
    app.run(debug=True)

