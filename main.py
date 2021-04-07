from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return 'ML microservice'

@app.route('/')
def show():
    #abort(405)
    return {'prova':'prova'}

if __name__ == '__main__':
    app.run()