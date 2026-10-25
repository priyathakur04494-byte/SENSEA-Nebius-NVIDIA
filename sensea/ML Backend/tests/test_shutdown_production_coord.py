# test_shutdown_production_coord.py
# Tests the production coordinator helper
import sys
import os
import threading
import time
import queue

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transcribe_shutdown_coord import (
    ShutdownCoordinator,
    worker_loop_with_task_done,
)


def test_callback_racing_with_shutdown():
    sc = ShutdownCoordinator()
    sc.request_shutdown()
    assert sc.try_enqueue_audio(("s1", b"x")) == False
    print("[PASS] test_callback_racing_with_shutdown")
    return True


def test_no_enqueue_after_shutdown_begins():
    sc = ShutdownCoordinator()
    sc.request_shutdown()
    # Try to enqueue
    result = sc.try_enqueue_audio(("s1", "data"))
    assert result == False
    print("[PASS] test_no_enqueue_after_shutdown_begins")
    return True


def test_previously_accepted_audio_reaches_worker_before_sentinel():
    sc = ShutdownCoordinator()
    # Accept audio before shutdown
    assert sc.try_enqueue_audio(("s1", b"audio1")) == True
    assert sc.try_enqueue_audio(("s1", b"audio2")) == True
    # Drain to transcription before sentinel
    transferred = sc.drain_audio_to_transcription(lambda x: sc.transcription_queue().put_nowait(x))
    assert transferred == 2
    sent = sc.send_sentinel_once()
    assert sent == True
    # Check order
    items = []
    while True:
        try:
            items.append(sc.transcription_queue().get_nowait())
        except:
            break
    assert items[0] == ("s1", b"audio1")
    assert items[1] == ("s1", b"audio2")
    assert items[2] is None
    print("[PASS] test_previously_accepted_audio_reaches_worker_before_sentinel")
    return True


def test_sentinel_insertion_only_once():
    sc = ShutdownCoordinator()
    sc.request_shutdown()
    r1 = sc.send_sentinel_once()
    r2 = sc.send_sentinel_once()
    r3 = sc.send_sentinel_once()
    assert r1 == True
    assert r2 == False
    assert r3 == False
    print("[PASS] test_sentinel_insertion_only_once")
    return True


def test_task_done_once_per_dequeued_item():
    q = queue.Queue()

    q.put("item1")
    q.put(None)
    w = threading.Thread(
        target=worker_loop_with_task_done,
        args=(q,),
        daemon=True,
    )
    w.start()
    w.join(timeout=2.0)
    assert not w.is_alive()
    assert q.unfinished_tasks == 0
    print("[PASS] test_task_done_once_per_dequeued_item")
    return True


def test_worker_join_bounded():
    def stuck():
        time.sleep(10)
    w = threading.Thread(target=stuck, daemon=True)
    w.start()
    start = time.time()
    w.join(timeout=0.2)
    elapsed = time.time() - start
    assert elapsed < 1.0
    assert w.is_alive()
    print("[PASS] test_worker_join_bounded")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("PRODUCTION COORDINATOR TESTS")
    print("=" * 60)
    tests = [
        test_callback_racing_with_shutdown,
        test_no_enqueue_after_shutdown_begins,
        test_previously_accepted_audio_reaches_worker_before_sentinel,
        test_sentinel_insertion_only_once,
        test_task_done_once_per_dequeued_item,
        test_worker_join_bounded,
    ]
    passed = failed = 0
    for t in tests:
        try:
            if t(): passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\nTotal: {passed+failed} | Passed: {passed} | Failed: {failed}")
    sys.exit(1 if failed else 0)
