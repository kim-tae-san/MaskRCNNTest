from flask import Flask
import custom

app = Flask(__name__)

@app.route("/model", methods=["GET"])
def model():
	if request.method == "GET":
		url = request.args.get('url')
		image = url_to_image(url)
		custom.detect_and_color_splash(image)


def url_to_image(url):

	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	
	return image

if __name__=="__main__":
	app.run(debug=True)