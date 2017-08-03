from flask import Flask, render_template, request, url_for, redirect


app = Flask(__name__)
app.config.from_object('config')
filename = ""


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        f = request.files['file']

        label = 1
        diagnosis = "Positive" if label else "Negative"
        return render_template('result.html', result=diagnosis)

if __name__ == "__main__":
    app.run()

