from flask import Flask, render_template, request, redirect, flash, Response
from utils.face_utils import register_employee, recognize_face
import cv2

app = Flask(__name__)
app.secret_key = 'securekey'

# Global camera object
camera = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        emp_id = request.form['emp_id']
        name = request.form['name']
        status = register_employee(camera, emp_id, name)
        flash(status)
        return redirect('/register')
    return render_template('register.html')

@app.route('/recognize', methods=['POST'])
def recognize():
    result = recognize_face(camera)
    flash(result)
    return redirect('/')

# Live video stream
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/shutdown')
def shutdown():
    camera.release()
    return "Camera released"

if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        camera.release()
