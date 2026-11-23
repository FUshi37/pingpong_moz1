"""The memory supervisor must signal the actual worker, retaining its log pipe."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def test_logged_scope_child_keeps_worker_pid_and_delivers_sigint(tmp_path):
    launcher = Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim/run_wmj_v1_collect.sh'
    logfile = tmp_path/'console.log'
    ready = tmp_path/'ready'
    worker = tmp_path/'worker.py'
    worker.write_text('import signal, sys, time\nfrom pathlib import Path\n'
        'def stopped(signum, frame):\n    print("saved-on-stop", flush=True)\n    sys.exit(17)\n'
        'signal.signal(signal.SIGINT, stopped)\n'
        'Path(sys.argv[1]).write_text("ready")\ntime.sleep(30)\n')
    child = subprocess.Popen(['bash', str(launcher), '--scope-child', str(logfile),
        sys.executable, str(worker), str(ready)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        start_new_session=True)
    try:
        deadline = time.monotonic()+5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert ready.exists()
        os.kill(child.pid, signal.SIGINT)
        output = child.communicate(timeout=3)[0]
        assert child.returncode == 17
        assert b'saved-on-stop' in output
        assert 'saved-on-stop' in logfile.read_text()
    finally:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()
