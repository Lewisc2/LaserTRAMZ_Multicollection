from flask import Flask, render_template, request
import sys
sys.path.append('./scripts')
from LaserTRAMZ_MC_UPb_analytes import calc_fncs
import panel as pn
pn.extension()

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/analytes")
def analytes():
    return render_template('analytes.html')

if __name__ == '__main__':
    # Start the Panel server in a subprocess
    import subprocess
    import sys

    # Serve the Panel app on port 8006
    panel_proc = subprocess.Popen([
        sys.executable, '-m', 'panel', 'serve',
        os.path.join('LaserTRAMZ_MC_UPb_analytes.py'),
        '--address', 'localhost', '--port', '8006', '--allow-websocket-origin=localhost:8000'
    ])

    try:
        # Start Flask app on port 5000
        app.run(port=8000, debug=True)
    finally:
        panel_proc.terminate()