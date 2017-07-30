from flask import Flask, render_template, request

app = Flask(__name__)
app.config.from_object('config')


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        print f.readline()
        return 'file uploaded successfully'

if __name__ == "__main__":
    app.run()

