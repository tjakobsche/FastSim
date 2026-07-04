# MIT License
#
# Copyright (c) 2023-2025 Hewlett Packard Enterprise Development LP 
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import traceback
import threading
import signal
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
import zmq

from controller import Controller
from logging_setup import setup_run_logs

class SliceBuffer:
    """
    Thread-safe strict-lockstep buffer for per-second slices.

    Invariants:
      - Producer (FastSim) calls publish(t, state) in increasing t.
      - Consumer (Digital Twin) calls GET(t) in order and server calls ack(t) after reply.
      - Backpressure: publish() blocks while (t - last_acked_t) >= capacity_seconds.
      - Memory is bounded: on ack(t), drop all slices < last_acked_t.
    """
    def __init__(self, capacity_seconds=900):
        self.capacity = int(capacity_seconds)
        self.buf = {}               # t -> list[str]
        self.max_published_t = -1   # largest t published so far
        self.last_acked_t = -1      # largest t acknowledged by client
        self.closed = False
        self._cv = threading.Condition()

    # Called by the simulator thread after each step
    def publish(self, t: int, running_ids):
        with self._cv:
            # Strict backpressure: don't let producer get ahead by >= capacity
            while not self.closed and (t - self.last_acked_t) >= self.capacity:
                self._cv.wait()

            if self.closed:
                return

            self.buf[t] = list(running_ids)
            if t > self.max_published_t:
                self.max_published_t = t
            self._cv.notify_all()

    # Called by the server thread when a client requests t
    def get(self, t: int):
        with self._cv:
            # Disallow going backwards (older than last ack)
            if t < self.last_acked_t:
                raise RuntimeError(f"requested t={t} < last_acked_t={self.last_acked_t}")
            # Wait until t is produced or simulation is closed
            while not self.closed and t > self.max_published_t:
                self._cv.wait()
            if self.closed and t > self.max_published_t:
                raise RuntimeError(f"simulation finished; latest available t={self.max_published_t}, requested t={t}")
            return list(self.buf[t])

    # Called by the server thread AFTER replying to GET(t)
    def ack(self, t: int):
        with self._cv:
            if t > self.last_acked_t:
                self.last_acked_t = t
                # Free memory: drop all slices strictly older than last_acked_t
                to_delete = [k for k in self.buf.keys() if k < self.last_acked_t]
                for k in to_delete:
                    self.buf.pop(k, None)
            self._cv.notify_all()

    def close(self):
        with self._cv:
            self.closed = True
            self._cv.notify_all()


class PublishingController(Controller):
    """
    Publishes an authoritative running-set slice for each whole second crossed.
    Scheduling/backfill logic is untouched; we only add post-step publishing.
    """
    def __init__(self, *args, slice_buffer: SliceBuffer, **kwargs):
        super().__init__(*args, **kwargs)
        self._slices = slice_buffer
        self._epoch = self.init_time
        self._last_pub_t = -1
        # Publish initial state (t=0)
        self._maybe_publish()

    def _maybe_publish(self):
        sec = int((self.time - self._epoch).total_seconds())
        running_ids = [job.jid for job in self.running_jobs]
        # Fill every whole second crossed since last publish
        for t in range(self._last_pub_t + 1, sec + 1):
            self._slices.publish(t, running_ids)
            self._last_pub_t = t

    def _step(self, run_main_and_resv_scheduler, sched_depth, bf, fairtree):
        super()._step(run_main_and_resv_scheduler, sched_depth, bf, fairtree)
        self._maybe_publish()


def print_sim_result(controller):
    max_submit = max(controller.job_history, key=lambda job: job.true_submit).true_submit
    job_history = [
        job for job in controller.job_history
        if ((controller.init_time + timedelta(days=2) < job.true_submit < max_submit - timedelta(days=2))
            and not job.ignore_in_eval)
    ]
    data_bd_slowdowns = [
        max(((job.runtime + job.true_job_start - job.true_submit) /
             max(job.runtime, controller.config.bd_threshold)), 1)
        for job in job_history
    ]
    sim_bd_slowdowns = [
        max((job.end - job.submit) / max(job.runtime, controller.config.bd_threshold), 1)
        for job in job_history
    ]
    data_wait_times = [
        (job.true_job_start - job.true_submit).total_seconds() / 3600.0
        for job in job_history
    ]
    sim_wait_times = [
        (job.start - job.submit).total_seconds() / 3600.0
        for job in job_history
    ]
    print(
        "True starts mean bd slowdown={}+-{} (total = {})\n".format(
            np.mean(data_bd_slowdowns), np.std(data_bd_slowdowns), np.sum(data_bd_slowdowns)
        ) +
        "Scheduling sim mean bd slowdown={}+-{} (total = {})\n".format(
            np.mean(sim_bd_slowdowns), np.std(sim_bd_slowdowns), np.sum(sim_bd_slowdowns)
        ) +
        "True starts mean wait time={}+-{} hrs (total = {} hrs)\n".format(
            np.mean(data_wait_times), np.std(data_wait_times), np.sum(data_wait_times)
        ) +
        "Scheduling sim mean wait time={}+-{}hrs (total = {} hrs)\n".format(
            np.mean(sim_wait_times), np.std(sim_wait_times), np.sum(sim_wait_times)
        )
    )


# -----------------------------
# Server loop (parallel mode)
# -----------------------------
def serve(controller: PublishingController, slices: SliceBuffer, endpoint: str):
    """
    REQ/REP server with strict lockstep:
      - GET  { "t": int }  -> { "t": int, "running_ids": [str, ...] } then ack(t)
      - INIT               -> { "init_time": ISO8601 }
      - HEALTH             -> { "latest_t": int, "last_acked_t": int, "closed": bool }
      - END                -> { "ok": true }
    """
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(endpoint)

    print(f"[FastSim] Serving slices on {endpoint}")
    print(f"[FastSim] init_time = {controller.init_time.isoformat()}")

    while True:
        req = sock.recv_json()
        op = req.get("op", "GET")
        try:
            if op == "GET":
                t = int(req["t"])
                running_ids = slices.get(t)  # blocks until t exists
                sock.send_json({"t": t, "running_ids": running_ids})
                # Strict lockstep: advance low-watermark only AFTER reply
                slices.ack(t)
            elif op == "INIT":
                sock.send_json({"init_time": controller.init_time.isoformat()})
            elif op == "HEALTH":
                sock.send_json({
                    "latest_t": slices.max_published_t,
                    "last_acked_t": slices.last_acked_t,
                    "closed": slices.closed
                })
            elif op == "END":
                sock.send_json({"ok": True})
                break
            else:
                sock.send_json({"error": f"unknown op '{op}'"})
        except Exception as e:
            sock.send_json({"error": str(e)})

    print("[FastSim] Server exiting…")



def main(args):
    if args.serve:
        # Strict-lockstep server mode
        slices = SliceBuffer(capacity_seconds=args.buffer_seconds)
        controller = PublishingController(
            args.config_file, args.output,
            slice_buffer=slices
        )

        # Simulator in background; will block itself via slices.publish when client is behind
        def _run():
            try:
                controller.run_sim(max_steps=args.max_steps)
            finally:
                slices.close()
                print("[FastSim] Simulation finished; slices closed.")

        sim_thread = threading.Thread(target=_run, daemon=True)
        sim_thread.start()

        # Handle Ctrl-C gracefully
        def _sigint(signum, frame):
            print("\n[FastSim] SIGINT: stopping server.")
            slices.close()
            sys.exit(0)

        signal.signal(signal.SIGINT, _sigint)

        serve(controller, slices, args.endpoint)
        return

    # Offline mode
    run_logs = setup_run_logs(args.output)

    controller = Controller(
        args.config_file, args.output, run_logs=run_logs
    )

    controller.run_sim(max_steps=args.max_steps)
    print_sim_result(controller)

    print('Saving Job History at End of Simulation.')
    jobs = []
    for i, job in enumerate(controller.job_history):
        print(f'Adding job {str(i).rjust(6)} of {len(controller.job_history)}', end='\r')
        try:
            job_dict = {}
            for key, value in job.__dict__.items():
                if key == 'assoc':
                    continue
                elif key in ['qos', 'partition', 'partition_qos']:
                    job_dict[key] = value.name
                elif key == 'assigned_nodes':
                    job_dict[key] = set(node.nid for node in value)
                elif key == 'node_timeline':
                    job_dict[key] = [(str(ts), cnt) for ts, cnt in value]
                elif key == 'wait_history':
                    job_dict[key] = [(str(ts), reason) for ts, reason in value]
                else:
                    job_dict[key] = value
            jobs.append(job_dict)
        except Exception:
            print(f'Error while adding job {i} of {len(controller.job_history)}')
            traceback.print_exc()
    pd.DataFrame(jobs).to_pickle(args.output)


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Configuration file
    parser.add_argument("config_file", type=str)

    # Filepath to save results (offline mode)
    parser.add_argument("--output", type=str, default="", help="Pickle Controller after sim (offline mode)")

    # Maximum number of steps in simulation
    parser.add_argument("--max_steps", type=int, default=0, help="Terminate sim after max steps (0 = run until done)")

    # Power-aware scheduling parameters
    parser.add_argument("--power_weight", type=int, default=0, help="Power priority weight")
    parser.add_argument("--power_alpha", type=float, default=2, help="Power-opt exponent alpha")
    parser.add_argument("--power_beta", type=float, default=2, help="Power-opt exponent beta")
    parser.add_argument("--power_gamma", type=float, default=2, help="Power-opt exponent gamma")
    parser.add_argument("--power_tbs", type=float, default=5, help="Time to start boosting high power jobs")
    parser.add_argument("--power_tbe", type=float, default=15, help="Time to stop boosting high power jobs")

    # Parallel (server) mode with strict lockstep
    parser.add_argument("--serve", action="store_true", help="Run as a blocking slice server for Digital Twin (strict lockstep)")
    parser.add_argument("--endpoint", type=str, default="ipc:///tmp/fastsim.sock",
                        help="ZeroMQ REQ/REP endpoint (e.g., ipc:///tmp/fastsim.sock or tcp://0.0.0.0:5555)")
    parser.add_argument("--buffer_seconds", type=int, default=900,
                        help="Maximum unacked seconds allowed before FastSim stalls (strict lockstep)")

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())