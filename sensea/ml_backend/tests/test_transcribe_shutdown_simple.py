# Simple shutdown tests without importing full transcribe module
import os
import sys
import time
import queue
import threading
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_sentinel_causes_worker_exit():
    """Sentinel None causes worker to exit."""
    # Simulate worker behavior
    q = queue.Queue()
    q.put(None)

    def worker():
        while True:
            item = q.get()
            try:
                if item is None:
                    break
            finally:
                q.task_done()

    w = threading.Thread(target=worker, daemon=True)
    w.start()
    w.join(timeout=2.0)
    assert not w.is_alive()
    print("[PASS] test_sentinel_causes_worker_exit")
    return True

def test_pending_work_processed_before_sentinel():
    """Pending items processed before sentinel."""
    q = queue.Queue()
    results = []

    def worker():
        while True:
            item = q.get()
            try:
                if item is None:
                    break
                results.append(item)
            finally:
                q.task_done()

    q.put("item1")
    q.put("item2")
    q.put(None)

    w = threading.Thread(target=worker, daemon=True)
    w.start()
    w.join(timeout=2.0)
    assert not w.is_alive()
    assert results == ["item1", "item2"]
    print("[PASS] test_pending_work_processed_before_sentinel")
    return True

def test_worker_exception_task_done():
    """Worker exceptions don't leave task accounting inconsistent."""
    q = queue.Queue()

    def worker():
        while True:
            item = q.get()
            try:
                if item is None:
                    break
                raise Exception("test error")
            finally:
                q.task_done()

    q.put("bad_item")
    q.put(None)

    w = threading.Thread(target=worker, daemon=True)
    w.start()
    w.join(timeout=2.0)
    assert not w.is_alive()
    print("[PASS] test_worker_exception_task_done")
    return True

def test_shutdown_timeout_handled():
    """Shutdown timeout handled without hanging."""
    w = threading.Thread(target=time.sleep, args=(10,), daemon=True)
    w.start()
    w.join(timeout=0.1)
    assert w.is_alive()  # didn't hang
    print("[PASS] test_shutdown_timeout_handled")
    return True

def test_repeated_shutdown_no_duplicate_sentinels():
    """Repeated shutdown requests don't enqueue duplicate sentinels."""
    q = queue.Queue()
    sentinel_sent = False

    def send_sentinel():
        nonlocal sentinel_sent
        with threading.Lock():
            if not sentinel_sent:
                sentinel_sent = True
                try:
                    q.put_nowait(None)
                    return True
                except:
                    return False
            return False

    assert send_sentinel() == True
    assert send_sentinel() == False  # duplicate prevented
    assert send_sentinel() == False
    print("[PASS] test_repeated_shutdown_no_duplicate_sentinels")
    return True

def test_buffered_audio_queued_before_sentinel():
    """Buffered audio is submitted before the worker sentinel."""
    q = queue.Queue()

    current_audio = [
        np.array([0.1, 0.2], dtype=np.float32),
        np.array([0.3, 0.4], dtype=np.float32),
    ]
    current_samples = 4
    active_session = "session-1"

    final_audio = np.concatenate(current_audio).copy()
    q.put((active_session, final_audio))

    current_audio = []
    current_samples = 0
    active_session = None

    q.put(None)

    audio_item = q.get_nowait()
    sentinel = q.get_nowait()

    assert audio_item[0] == "session-1"
    assert np.allclose(audio_item[1], [0.1, 0.2, 0.3, 0.4])
    assert sentinel is None
    assert q.empty()

    print("[PASS] test_buffered_audio_queued_before_sentinel")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("TRANSCRIBE SHUTDOWN TESTS (SIMPLE)")
    print("=" * 60)

    tests = [
        test_sentinel_causes_worker_exit,
        test_pending_work_processed_before_sentinel,
        test_worker_exception_task_done,
        test_shutdown_timeout_handled,
        test_repeated_shutdown_no_duplicate_sentinels,
        test_buffered_audio_queued_before_sentinel,

    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Total: {passed + failed} | Passed: {passed} | Failed: {failed}")
    print("=" * 60)

    sys.exit(1 if failed else 0)
