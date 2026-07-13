"""
Shared resource-usage helpers for SLURM job sizing.

Used by auto_reconstruct.py and extract_buildings.py to report actual peak
memory/CPU usage at the end of a run - so every run becomes a calibration data
point for how much --mem/--cpus-per-task a similarly-sized dataset needs next
time. Also used conceptually by estimate_resources.py, which gives a rough
starting point *before* a dataset has ever been run.
"""
import os
import platform
import resource
import threading


def detect_cpu_count():
    """Cores actually allocated to this job. Falling back to os.cpu_count()
    alone under SLURM can report the whole node's physical core count even
    when only a fraction was allocated to this job, causing worker pools to
    oversubscribe cores they don't actually have. Returns (count, source)."""
    slurm_cores = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_cores and slurm_cores.isdigit():
        return int(slurm_cores), 'SLURM_CPUS_PER_TASK'
    return (os.cpu_count() or 1), 'os.cpu_count() (no SLURM allocation detected)'


def _rusage_peak_gb():
    """Fallback peak-memory estimate using only the stdlib (no psutil). Reports
    this process's peak RSS plus the largest single *terminated* child's peak
    RSS - NOT the sum of memory used by workers running concurrently, since
    POSIX's ru_maxrss for RUSAGE_CHILDREN only tracks the largest child seen so
    far, not a running total across still-alive processes. This will
    under-report true peak usage whenever multiple workers ran at once."""
    divisor = (1024 * 1024) if platform.system() == 'Linux' else (1024 * 1024 * 1024)
    self_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor
    children_peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / divisor
    return self_peak + children_peak


class MemoryMonitor:
    """Samples total RSS (this process + all live descendant processes) in a
    background thread every `interval` seconds, so the peak reported at the
    end reflects memory used while parallel workers (or a containerized
    reconstruction step) were actually running concurrently.

    Requires psutil. Without it, falls back to a cruder stdlib-only estimate
    that can under-report concurrent peaks - still better than nothing, but
    install psutil (`pip install psutil`) for an accurate number.
    """

    def __init__(self, interval=5.0):
        self.interval = interval
        self.peak_gb = 0.0
        self._stop = threading.Event()
        self._thread = None
        self._psutil = None
        try:
            import psutil
            self._psutil = psutil
        except ImportError:
            pass

    @property
    def using_psutil(self):
        return self._psutil is not None

    def _run(self):
        proc = self._psutil.Process(os.getpid())
        while not self._stop.is_set():
            try:
                total = proc.memory_info().rss
                for child in proc.children(recursive=True):
                    try:
                        total += child.memory_info().rss
                    except Exception:
                        pass
                self.peak_gb = max(self.peak_gb, total / (1024 ** 3))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        """Starts background sampling. Returns False (no-op) if psutil isn't
        available - callers should fall back to stop_and_report() anyway,
        which handles that case."""
        if not self.using_psutil:
            return False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def stop_and_report(self):
        """Stops sampling (if running) and returns (peak_gb, method_str)."""
        if self.using_psutil and self._thread:
            self._stop.set()
            self._thread.join(timeout=self.interval + 2)
            return self.peak_gb, f"psutil, main + all live child processes, sampled every {self.interval:.0f}s"
        return _rusage_peak_gb(), "resource module fallback (undercounts concurrent workers - install psutil for accuracy)"
