from flask import Blueprint,render_template,Response
from .models import generate_frames,generate_detected_frames

views = Blueprint('views', __name__)

@views.route('/')
def home():
    return render_template("index.html")

@views.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@views.route('/detectvideo')
def detectvideo():
    return Response(generate_detected_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')