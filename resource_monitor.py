# resource_monitor.py
"""
Samples CPU and RAM for the current process + children, and reports EXACT
peak RAM (kernel high-water-mark) instead of an approximation from sampling.

WHY THIS CHANGED: the old approach sampled CURRENT RAM every `interval`
seconds and took max(samples) as "peak" -- that misses any spike that rises
and falls between two samples. This version instead reads kernel-tracked
monotonic high-water-mark values, which can't miss a spike no matter when
you read them:

  - Per-process peak: Linux's /proc/<pid>/status -> VmHWM field.
  - Container-wide peak: cgroup v2's memory.peak / v1's memory.max_usage_in_bytes.

LIMITATION: only available on Linux (i.e. inside Docker). Native Windows has
no equivalent exposed here, so it falls back to the old sampled-max approach
there -- container peak stays None on Windows either way, same as before.

LIMITATION (process-tree peak specifically): a child that starts AND exits
entirely between samples can still be missed. Once a tracked process exits,
its LAST KNOWN VmHWM is kept (not discarded) and included in the running
sum -- so short-lived children's peak contribution isn't lost, though the
summed total is a robust upper bound rather than "everyone peaked at this
exact same instant" (different processes can peak at different times).
"""

import os
import time
import threading
import psutil


def _read_int_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


class ResourceMonitor:
    """
    samples: (timestamp, cpu_percent, process_tree_ram_mb_CURRENT, container_ram_mb_CURRENT_or_None)
    These sampled "current" values still back AVERAGE reporting (an average
    genuinely needs periodic samples). PEAKS come from:
        .exact_peak_process_tree_ram_mb
        .exact_peak_container_ram_mb
    """

    def __init__(self, interval=0.5):
        self.interval = interval
        self.samples = []

        self._running = False
        self._thread = None
        self._known_processes = {}

        # pid -> last known VmHWM (MB), kept even after the process exits.
        self._last_known_vmhwm_mb = {}

        self.exact_peak_process_tree_ram_mb = 0.0
        self.exact_peak_container_ram_mb = None

        try:
            self._root_process = psutil.Process(os.getpid())
        except psutil.Error:
            self._root_process = None

    # ------------------------------------------------------------------
    # Process tree
    # ------------------------------------------------------------------

    def _get_process_tree(self):
        if self._root_process is None:
            return []
        processes = []
        try:
            if self._root_process.is_running():
                processes.append(self._root_process)
            processes.extend(self._root_process.children(recursive=True))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
        return processes

    # ------------------------------------------------------------------
    # RAM of Python + children (CURRENT usage, for averages only)
    # ------------------------------------------------------------------

    def _get_process_tree_ram_mb(self, processes):
        total_bytes = 0
        for process in processes:
            try:
                total_bytes += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return total_bytes / (1024 ** 2)

    # ------------------------------------------------------------------
    # EXACT per-process peak RSS (Linux high-water mark)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_process_peak_rss_mb(pid):
        """Reads VmHWM from /proc/<pid>/status -- kernel's own monotonic
        peak-RSS record for this process. Linux-only; None elsewhere."""
        try:
            with open(f"/proc/{pid}/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmHWM:"):
                        return int(line.split()[1]) / 1024.0  # kB -> MB
        except (OSError, ValueError, IndexError):
            return None
        return None

    def _update_exact_process_tree_peak(self, processes):
        for process in processes:
            try:
                peak_mb = self._get_process_peak_rss_mb(process.pid)
                if peak_mb is not None:
                    self._last_known_vmhwm_mb[process.pid] = peak_mb
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

        if self._last_known_vmhwm_mb:
            self.exact_peak_process_tree_ram_mb = sum(self._last_known_vmhwm_mb.values())

    # ------------------------------------------------------------------
    # CPU of Python + children (unchanged from before)
    # ------------------------------------------------------------------

    def _get_process_tree_cpu_percent(self, processes):
        current_pids = set()
        total_cpu = 0.0
        for process in processes:
            try:
                pid = process.pid
                current_pids.add(pid)
                if pid not in self._known_processes:
                    self._known_processes[pid] = process
                    process.cpu_percent(interval=None)
                    continue
                total_cpu += self._known_processes[pid].cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

        dead_pids = [pid for pid in self._known_processes if pid not in current_pids]
        for pid in dead_pids:
            self._known_processes.pop(pid, None)
        return total_cpu

    # ------------------------------------------------------------------
    # Docker container RAM -- current (avg) and TRUE peak
    # ------------------------------------------------------------------

    @staticmethod
    def _get_container_ram_mb():
        """CURRENT container memory usage -- for average reporting only."""
        for path in ("/sys/fs/cgroup/memory.current",
                     "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
            if os.path.exists(path):
                value = _read_int_file(path)
                if value is not None:
                    return value / (1024 ** 2)
        return None

    @staticmethod
    def _get_container_peak_ram_mb():
        """TRUE peak container RAM since cgroup creation -- kernel-tracked,
        not sampled/estimated at all. None outside a real cgroup."""
        for path in ("/sys/fs/cgroup/memory.peak",
                     "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
            if os.path.exists(path):
                value = _read_int_file(path)
                if value is not None:
                    return value / (1024 ** 2)
        return None

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_once(self):
        processes = self._get_process_tree()

        cpu_pct = self._get_process_tree_cpu_percent(processes)
        ram_mb = self._get_process_tree_ram_mb(processes)
        container_ram_mb = self._get_container_ram_mb()

        self._update_exact_process_tree_peak(processes)

        container_peak = self._get_container_peak_ram_mb()
        if container_peak is not None:
            self.exact_peak_container_ram_mb = container_peak

        self.samples.append((time.time(), cpu_pct, ram_mb, container_ram_mb))

    def _run(self):
        while self._running:
            self._sample_once()
            time.sleep(self.interval)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        if self._running:
            return
        self.samples = []
        self._known_processes = {}
        self._last_known_vmhwm_mb = {}
        self.exact_peak_process_tree_ram_mb = 0.0
        self.exact_peak_container_ram_mb = None

        for process in self._get_process_tree():
            try:
                process.cpu_percent(interval=None)
                self._known_processes[process.pid] = process
            except psutil.Error:
                pass

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if not self._running:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval * 2))
        try:
            self._sample_once()
        except Exception:
            pass