from flask import Flask, render_template, request, url_for, redirect
import incres_predict
import os
import time

app = Flask(__name__)
app.config.from_object('config')
filename = ""


@app.route('/', methods=['GET', 'POST'])
def index():
	if request.method == 'GET':
		return render_template('index.html')
	if request.method == 'POST':
		cwd = os.getcwd()
		start = 0
		timeused = 0
		path = os.path.join(cwd, "static/image.png")
		print(path)

	
		# try:

		f = request.files['file']
		f.save(path)
		print(os.listdir(cwd))

		start = time.time()
		label = int(incres_predict.predict(input = path))
		timeused = time.time() - start
		print(timeused)
		

		diagnosis = "Malignant" if label else "Benign"
		timeline = "Time used: " + str(round(timeused, 2)) + " seconds"
		print(diagnosis)
		# except:
			# return "Please upload your file first!"
		return render_template('result.html', result=diagnosis, time=timeline)

if __name__ == "__main__":
	app.run()

