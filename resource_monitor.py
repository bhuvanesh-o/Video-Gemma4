# resource_monitor.py
"""
Samples CPU% and RAM usage of the current process (and its children — matters
because cv2/onnx/openvino can spawn worker threads/processes) at a fixed
interval on a background thread, so it doesn't block or slow down the actual
pipeline it's watching.
"""
import psutil
import threading
import time
import os


class ResourceMonitor:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.samples = []  # list of (timestamp, cpu_percent, ram_mb)
        self._stop_flag = threading.Event()
        self._thread = None

    def _sample_loop(self):
        # First call to cpu_percent() always returns 0.0 — it needs a baseline.
        # Calling it once here "primes" it before the real sampling starts.
        self.process.cpu_percent(interval=None)
        for child in self.process.children(recursive=True):
            try:
                child.cpu_percent(interval=None)
            except psutil.NoSuchProcess:
                pass

        start = time.time()
        while not self._stop_flag.is_set():
            time.sleep(self.interval)
            try:
                # cpu_percent() here is % of ONE core; can exceed 100% if
                # multi-threaded (e.g. 350% = using 3.5 cores worth of work)
                cpu = self.process.cpu_percent(interval=None)
                mem_mb = self.process.memory_info().rss / (1024 * 1024)

                # Include child processes (OpenVINO/cv2 may spawn worker threads
                # or subprocesses depending on backend — this catches those too)
                for child in self.process.children(recursive=True):
                    try:
                        cpu += child.cpu_percent(interval=None)
                        mem_mb += child.memory_info().rss / (1024 * 1024)
                    except psutil.NoSuchProcess:
                        continue

                self.samples.append((time.time() - start, cpu, mem_mb))
            except psutil.NoSuchProcess:
                break

    def start(self):
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_flag.set()
        if self._thread:
            self._thread.join()

    def summary(self):
        if not self.samples:
            return "No samples collected."
        cpu_vals = [s[1] for s in self.samples]
        mem_vals = [s[2] for s in self.samples]
        return (
            f"Duration sampled: {self.samples[-1][0]:.1f}s over {len(self.samples)} samples\n"
            f"CPU%  -> avg: {sum(cpu_vals)/len(cpu_vals):.1f}%  peak: {max(cpu_vals):.1f}%\n"
            f"RAM MB -> avg: {sum(mem_vals)/len(mem_vals):.1f} MB  peak: {max(mem_vals):.1f} MB"
        )

    def save_csv(self, path="resource_log.csv"):
        import csv
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["elapsed_sec", "cpu_percent", "ram_mb"])
            writer.writerows(self.samples)