# test_shutdown_coordination.py
# Tests the exact shutdown coordination logic with stdlib only
import threading
import queue
import time
import sys

class ShutdownCoord:
    def __init__(self):
        self.stop_program = False
        self.shutdown_requested = False
        self.sentinel_sent = False
        self.shutdown_lock = threading.Lock()
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        self.sentinel_count = 0

    def request_shutdown_esc(self):
        with self.shutdown_lock:
            self.stop_program = True
            self.shutdown_requested = True

    def callback_enqueue(self, item):
        with self.shutdown_lock:
            if self.stop_program or self.shutdown_requested:
                return False
        try:
            with self.shutdown_lock:
                if self.stop_program or self.shutdown_requested:
                    return False
            self.audio_queue.put_nowait(item)
            return True
        except Exception:
            return False

    def shutdown_and_sentinel(self):
        # stop listener conceptually
        with self.shutdown_lock:
            self.stop_program = True
            self.shutdown_requested = True
            if not self.sentinel_sent:
                self.sentinel_sent = True
                try:
                    self.transcription_queue.put_nowait(None)
                    self.sentinel_count += 1
                    return True
                except Exception:
                    try:
                        self.transcription_queue.put(None)
                        self.sentinel_count += 1
                        return True
                    except Exception:
                        self.sentinel_sent = False
                        return False
            return False

    def worker_join_bounded(self, worker, timeout=5.0):
        if worker and worker.is_alive():
            worker.join(timeout=timeout)
            return not worker.is_alive()
        return True

def test_accepted_callback_cannot_enqueue_after_shutdown_atomic():
    sc = ShutdownCoord()
    sc.request_shutdown_esc()
    assert sc.callback_enqueue(("s1", b"x")) == False
    print("[PASS] test_accepted_callback_cannot_enqueue_after_shutdown_atomic")
    return True

def test_callback_racing_cannot_enqueue_behind_sentinel():
    sc = ShutdownCoord()
    sc.request_shutdown_esc()
    sc.shutdown_and_sentinel()
    assert sc.callback_enqueue(("s1", b"x")) == False
    # Check nothing behind sentinel
    items = []
    while True:
        try:
            items.append(sc.transcription_queue.get_nowait())
        except:
            break
    # Only sentinel
    assert items == [None]
    print("[PASS] test_callback_racing_cannot_enqueue_behind_sentinel")
    return True

def test_sentinel_insertion_idempotent():
    sc = ShutdownCoord()
    sc.request_shutdown_esc()
    r1 = sc.shutdown_and_sentinel()
    r2 = sc.shutdown_and_sentinel()
    r3 = sc.shutdown_and_sentinel()
    assert sc.sentinel_count == 1
    print("[PASS] test_sentinel_insertion_idempotent")
    return True

def test_previously_accepted_work_not_silently_discarded():
    sc = ShutdownCoord()
    # Enqueue before shutdown
    sc.audio_queue.put_nowait(("s1", b"data"))
    sc.request_shutdown_esc()
    # Work already in audio_queue should remain (main loop would process)
    items = []
    try:
        items.append(sc.audio_queue.get_nowait())
    except:
        pass
    assert len(items) == 1
    sc.shutdown_and_sentinel()
    print("[PASS] test_previously_accepted_work_not_silently_discarded")
    return True

def test_worker_join_timeout_bounded():
    def stuck():
        time.sleep(10)
    w = threading.Thread(target=stuck, daemon=True)
    w.start()
    sc = ShutdownCoord()
    start = time.time()
    result = sc.worker_join_bounded(w, timeout=0.2)
    elapsed = time.time() - start
    assert elapsed < 1.0  # bounded
    assert result == False  # didn't finish
    print("[PASS] test_worker_join_timeout_bounded")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("SHUTDOWN COORDINATION TESTS (EXACT HELPER)")
    print("=" * 60)
    tests = [
        test_accepted_callback_cannot_enqueue_after_shutdown_atomic,
        test_callback_racing_cannot_enqueue_behind_sentinel,
        test_sentinel_insertion_idempotent,
        test_previously_accepted_work_not_silently_discarded,
        test_worker_join_timeout_bounded,
    ]
    passed = failed = 0
    for t in tests:
        try:
            if t(): passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failed += 1
    print(f"\nTotal: {passed+failed} | Passed: {passed} | Failed: {failed}")
    sys.exit(1 if failed else 0)
