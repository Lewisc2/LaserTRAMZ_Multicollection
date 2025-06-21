from flask import Flask, render_template, request
import sys
import os
sys.path.append('./scripts')
import panel as pn
pn.extension()

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/analytes")
def analytes():
    return render_template('analytes.html')

@app.route("/concordia")
def concordia():
    return render_template('concordia.html')

if __name__ == '__main__':
    # Start the Panel server in a subprocess
    import subprocess
    import sys

    # Serve the Panel app on port 8006
    panel_proc = subprocess.Popen([
    sys.executable, '-m', 'panel', 'serve',
    os.path.join('scripts', 'LaserTRAMZ_MC_UPb_analytes.py'),
    os.path.join('scripts', 'LaserTRAMZ_MC_UPb_Concordia.py'),
    '--address', 'localhost', '--port', '8006',
    '--allow-websocket-origin=localhost:8000',
    '--allow-websocket-origin=localhost:8006'
])

    try:
        # Start Flask app on port 8000
        app.run(port=8000, debug=False)
    finally:
        panel_proc.terminate()