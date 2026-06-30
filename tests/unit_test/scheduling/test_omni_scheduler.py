# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

from sglang_omni.scheduling import omni_scheduler


def _make_scheduler(*, waiting_queue, running_batch):
    scheduler = object.__new__(omni_scheduler.OmniScheduler)
    scheduler._defer_prefill_during_priority_decode = True
    scheduler._prioritize_stream_prefill = True
    scheduler._priority_prefill_max_batch_size = 0
    scheduler._priority_prefill_batch_wait_s = 0.0
    scheduler._isolate_prefill_only_batches = True
    scheduler._isolated_prefill_max_batch_size = 0
    scheduler._priority_prefill_rids = set()
    scheduler.chunked_req = None
    scheduler.waiting_queue = list(waiting_queue)
    scheduler.running_batch = running_batch
    scheduler.cur_batch = None
    scheduler.last_batch = None
    scheduler._async_pending = None
    scheduler.tree_cache = SimpleNamespace()
    scheduler.chunked_prefill_size = 32768
    return scheduler




def test_priority_first_stream_queue_prioritizes_talker_streams_by_default(monkeypatch):
    monkeypatch.delenv("SGLANG_OMNI_PRIORITY_TALKER_STREAM", raising=False)
    outbox = omni_scheduler._PriorityFirstStreamQueue()
    result = omni_scheduler.OutgoingMessage(
        request_id="done",
        type="result",
        data="result",
    )
    decode_stream = omni_scheduler.OutgoingMessage(
        request_id="text",
        type="stream",
        data="decode",
        target="decode",
    )
    talker_stream = omni_scheduler.OutgoingMessage(
        request_id="audio",
        type="stream",
        data="talker",
        target="talker_ar",
    )

    outbox.put(result)
    outbox.put(decode_stream)
    outbox.put(talker_stream)

    assert outbox.get_nowait() is talker_stream
    assert outbox.get_nowait() is result
    assert outbox.get_nowait() is decode_stream


def test_priority_first_stream_queue_talker_priority_can_be_disabled(monkeypatch):
    monkeypatch.setenv("SGLANG_OMNI_PRIORITY_TALKER_STREAM", "0")
    outbox = omni_scheduler._PriorityFirstStreamQueue()
    result = omni_scheduler.OutgoingMessage(
        request_id="done",
        type="result",
        data="result",
    )
    talker_stream = omni_scheduler.OutgoingMessage(
        request_id="audio",
        type="stream",
        data="talker",
        target="talker_ar",
    )

    outbox.put(result)
    outbox.put(talker_stream)

    assert outbox.get_nowait() is result
    assert outbox.get_nowait() is talker_stream


def test_priority_first_stream_queue_talker_stream_preempts_first_stream(monkeypatch):
    monkeypatch.delenv("SGLANG_OMNI_PRIORITY_TALKER_STREAM", raising=False)
    outbox = omni_scheduler._PriorityFirstStreamQueue()
    first_text_stream = omni_scheduler.OutgoingMessage(
        request_id="fresh",
        type="stream",
        data="first",
        target="decode",
    )
    setattr(first_text_stream, omni_scheduler._PRIORITY_MARKER_ATTR, True)
    talker_stream = omni_scheduler.OutgoingMessage(
        request_id="audio",
        type="stream",
        data="talker",
        target="talker_ar",
    )

    outbox.put(first_text_stream)
    outbox.put(talker_stream)

    assert outbox.get_nowait() is talker_stream
    assert outbox.get_nowait() is first_text_stream


def test_get_new_batch_prefill_defers_while_priority_decode_runs(monkeypatch):
    hidden_req = SimpleNamespace()
    scheduler = _make_scheduler(
        waiting_queue=[hidden_req],
        running_batch=SimpleNamespace(
            reqs=[SimpleNamespace(_omni_prioritize_prefill=True)]
        ),
    )

    def fail_if_called(self):
        raise AssertionError("upstream prefill should be deferred")

    monkeypatch.setattr(
        omni_scheduler._Upstream, "get_new_batch_prefill", fail_if_called
    )

    assert scheduler.get_new_batch_prefill() is None
    assert scheduler.waiting_queue == [hidden_req]


def test_get_new_batch_prefill_defers_by_priority_request_id(monkeypatch):
    hidden_req = SimpleNamespace(rid="waiting")
    running_req = SimpleNamespace(rid="running")
    scheduler = _make_scheduler(
        waiting_queue=[hidden_req],
        running_batch=SimpleNamespace(reqs=[running_req]),
    )
    scheduler._priority_prefill_rids.add("running")

    def fail_if_called(self):
        raise AssertionError("upstream prefill should be deferred")

    monkeypatch.setattr(
        omni_scheduler._Upstream, "get_new_batch_prefill", fail_if_called
    )

    assert scheduler.get_new_batch_prefill() is None
    assert scheduler.waiting_queue == [hidden_req]


def test_get_new_batch_prefill_allows_priority_while_priority_decode_runs(
    monkeypatch,
):
    priority_req = SimpleNamespace(rid="waiting", _omni_prioritize_prefill=True)
    running_req = SimpleNamespace(rid="running", _omni_prioritize_prefill=True)
    scheduler = _make_scheduler(
        waiting_queue=[priority_req],
        running_batch=SimpleNamespace(reqs=[running_req]),
    )
    seen = {}

    def fake_upstream_prefill(self):
        seen["waiting_queue"] = list(self.waiting_queue)
        self.waiting_queue = []
        return "prefill"

    monkeypatch.setattr(
        omni_scheduler._Upstream, "get_new_batch_prefill", fake_upstream_prefill
    )

    assert scheduler.get_new_batch_prefill() == "prefill"
    assert seen["waiting_queue"] == [priority_req]
    assert scheduler.waiting_queue == []
    assert scheduler._priority_prefill_rids == {"waiting"}


def test_get_new_batch_prefill_prioritizes_waiting_stream_prefill(monkeypatch):
    hidden_req = SimpleNamespace()
    priority_req = SimpleNamespace(rid="priority", _omni_prioritize_prefill=True)
    scheduler = _make_scheduler(
        waiting_queue=[hidden_req, priority_req],
        running_batch=None,
    )
    seen = {}

    def fake_upstream_prefill(self):
        seen["waiting_queue"] = list(self.waiting_queue)
        self.waiting_queue = []
        return "prefill"

    monkeypatch.setattr(
        omni_scheduler._Upstream, "get_new_batch_prefill", fake_upstream_prefill
    )

    assert scheduler.get_new_batch_prefill() == "prefill"
    assert seen["waiting_queue"] == [priority_req]
    assert scheduler.waiting_queue == [hidden_req]
    assert scheduler._priority_prefill_rids == {"priority"}


def test_get_new_batch_prefill_coalesces_priority_requests(monkeypatch):
    priority_req = SimpleNamespace(rid="priority-a", _omni_prioritize_prefill=True)
    drained_req = SimpleNamespace(rid="priority-b", _omni_prioritize_prefill=True)
    scheduler = _make_scheduler(
        waiting_queue=[priority_req],
        running_batch=None,
    )
    scheduler._priority_prefill_batch_wait_s = 0.123
    seen = {}

    monkeypatch.setattr(
        omni_scheduler.time,
        "sleep",
        lambda wait_s: seen.setdefault("wait_s", wait_s),
    )

    scheduler.recv_requests = lambda: [drained_req]
    scheduler._take_deferred_request_payloads = lambda: []
    scheduler.process_input_requests = lambda recv: scheduler.waiting_queue.extend(
        recv
    )

    def fake_upstream_prefill(self):
        seen["waiting_queue"] = list(self.waiting_queue)
        self.waiting_queue = []
        return "prefill"

    monkeypatch.setattr(
        omni_scheduler._Upstream, "get_new_batch_prefill", fake_upstream_prefill
    )

    assert scheduler.get_new_batch_prefill() == "prefill"
    assert seen["wait_s"] == 0.123
    assert seen["waiting_queue"] == [priority_req, drained_req]
    assert scheduler.waiting_queue == []
    assert scheduler._priority_prefill_rids == {"priority-a", "priority-b"}


def test_get_new_batch_prefill_keeps_chunked_isolated_prefill_alone(monkeypatch):
    hidden_req = SimpleNamespace()
    chunked_req = SimpleNamespace(_omni_isolate_prefill_batch=True)
    scheduler = _make_scheduler(
        waiting_queue=[hidden_req],
        running_batch=None,
    )
    scheduler.chunked_req = chunked_req
    seen = {}

    def fake_upstream_prefill(self):
        seen["waiting_queue"] = list(self.waiting_queue)
        self.waiting_queue = []
        return "prefill"

    monkeypatch.setattr(
        omni_scheduler._Upstream, "get_new_batch_prefill", fake_upstream_prefill
    )

    assert scheduler.get_new_batch_prefill() == "prefill"
    assert seen["waiting_queue"] == []
    assert scheduler.waiting_queue == [hidden_req]
    assert scheduler._priority_prefill_rids == set()


def test_get_new_batch_prefill_preempts_chunked_isolated_for_priority(
    monkeypatch,
):
    hidden_req = SimpleNamespace()
    isolated_req = SimpleNamespace(_omni_isolate_prefill_batch=True)
    priority_req = SimpleNamespace(
        rid="priority",
        _omni_prioritize_prefill=True,
        extend_input_len=83,
    )
    priority_req.init_next_round_input = lambda tree_cache: None
    chunked_req = SimpleNamespace(_omni_isolate_prefill_batch=True)
    scheduler = _make_scheduler(
        waiting_queue=[hidden_req, isolated_req, priority_req],
        running_batch=None,
    )
    scheduler.chunked_req = chunked_req
    seen = {}

    def fake_upstream_prefill(self):
        seen["chunked_req"] = self.chunked_req
        seen["waiting_queue"] = list(self.waiting_queue)
        self.waiting_queue = []
        return "prefill"

    monkeypatch.setattr(
        omni_scheduler._Upstream, "get_new_batch_prefill", fake_upstream_prefill
    )

    assert scheduler.get_new_batch_prefill() == "prefill"
    assert seen["chunked_req"] is None
    assert seen["waiting_queue"] == [priority_req]
    assert scheduler.chunked_req is chunked_req
    assert scheduler.waiting_queue == [hidden_req, isolated_req]
    assert scheduler._priority_prefill_rids == {"priority"}


def test_get_new_batch_prefill_batches_waiting_isolated_prefill(monkeypatch):
    hidden_req = SimpleNamespace()
    isolated_req = SimpleNamespace(_omni_isolate_prefill_batch=True)
    chunked_req = SimpleNamespace(_omni_isolate_prefill_batch=True)
    scheduler = _make_scheduler(
        waiting_queue=[hidden_req, isolated_req],
        running_batch=None,
    )
    scheduler.chunked_req = chunked_req
    seen = {}

    def fake_upstream_prefill(self):
        seen["waiting_queue"] = list(self.waiting_queue)
        self.waiting_queue = []
        return "prefill"

    monkeypatch.setattr(
        omni_scheduler._Upstream, "get_new_batch_prefill", fake_upstream_prefill
    )

    assert scheduler.get_new_batch_prefill() == "prefill"
    assert seen["waiting_queue"] == [isolated_req]
    assert scheduler.waiting_queue == [hidden_req]
    assert scheduler._priority_prefill_rids == set()


def test_get_new_batch_prefill_batches_isolated_prefill_without_chunked_req(
    monkeypatch,
):
    hidden_req = SimpleNamespace()
    isolated_req_a = SimpleNamespace(_omni_isolate_prefill_batch=True)
    isolated_req_b = SimpleNamespace(_omni_isolate_prefill_batch=True)
    scheduler = _make_scheduler(
        waiting_queue=[hidden_req, isolated_req_a, isolated_req_b],
        running_batch=None,
    )
    seen = {}

    def fake_upstream_prefill(self):
        seen["waiting_queue"] = list(self.waiting_queue)
        self.waiting_queue = [isolated_req_b]
        return "prefill"

    monkeypatch.setattr(
        omni_scheduler._Upstream, "get_new_batch_prefill", fake_upstream_prefill
    )

    assert scheduler.get_new_batch_prefill() == "prefill"
    assert seen["waiting_queue"] == [isolated_req_a, isolated_req_b]
    assert scheduler.waiting_queue == [isolated_req_b, hidden_req]
    assert scheduler._priority_prefill_rids == set()


def test_get_new_batch_prefill_caps_isolated_prefill_batch(monkeypatch):
    hidden_req = SimpleNamespace()
    isolated_req_a = SimpleNamespace(_omni_isolate_prefill_batch=True)
    isolated_req_b = SimpleNamespace(_omni_isolate_prefill_batch=True)
    scheduler = _make_scheduler(
        waiting_queue=[hidden_req, isolated_req_a, isolated_req_b],
        running_batch=None,
    )
    scheduler._isolated_prefill_max_batch_size = 1
    seen = {}

    def fake_upstream_prefill(self):
        seen["waiting_queue"] = list(self.waiting_queue)
        self.waiting_queue = []
        return "prefill"

    monkeypatch.setattr(
        omni_scheduler._Upstream, "get_new_batch_prefill", fake_upstream_prefill
    )

    assert scheduler.get_new_batch_prefill() == "prefill"
    assert seen["waiting_queue"] == [isolated_req_a]
    assert scheduler.waiting_queue == [isolated_req_b, hidden_req]
    assert scheduler._priority_prefill_rids == set()

def test_get_num_allocatable_reqs_caps_mamba_by_available_req_slots(monkeypatch):
    scheduler = object.__new__(omni_scheduler.OmniScheduler)
    scheduler.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(mambaish_config=object())
    )
    scheduler.req_to_token_pool = SimpleNamespace(available_size=lambda: 3)

    def fake_upstream_num_allocatable(self, running_bs):
        assert running_bs == 5
        return 8

    monkeypatch.setattr(
        omni_scheduler._Upstream,
        "get_num_allocatable_reqs",
        fake_upstream_num_allocatable,
    )

    assert scheduler.get_num_allocatable_reqs(5) == 3


def test_get_num_allocatable_reqs_keeps_upstream_limit_for_non_mamba(monkeypatch):
    scheduler = object.__new__(omni_scheduler.OmniScheduler)
    scheduler.tp_worker = SimpleNamespace(model_runner=SimpleNamespace())
    scheduler.req_to_token_pool = SimpleNamespace(available_size=lambda: 3)

    monkeypatch.setattr(
        omni_scheduler._Upstream,
        "get_num_allocatable_reqs",
        lambda self, running_bs: 8,
    )

    assert scheduler.get_num_allocatable_reqs(5) == 8

def test_process_batch_result_prefill_only_without_token_finishes_empty(monkeypatch):
    scheduler = object.__new__(omni_scheduler.OmniScheduler)
    scheduler.tree_cache = SimpleNamespace()

    seen = {"released": [], "streamed": None, "routed": []}
    monkeypatch.setattr(
        omni_scheduler._Upstream,
        "process_batch_result",
        lambda self, batch, result: (_ for _ in ()).throw(
            AssertionError("prefill-only no-token result should not enter upstream")
        ),
    )
    monkeypatch.setattr(
        omni_scheduler,
        "release_kv_cache",
        lambda req, tree_cache: seen["released"].append((req, tree_cache)),
    )
    scheduler.maybe_collect_routed_experts = lambda req: seen["routed"].append(req)
    scheduler.stream_output = lambda reqs, return_logprob, skip_req=None: seen.update(
        streamed=(list(reqs), return_logprob, skip_req)
    )

    req = SimpleNamespace(
        rid="prefill-only",
        output_ids=[],
        is_chunked=0,
        is_retracted=False,
        time_stats=SimpleNamespace(prefill_finished_ts=0.0, completion_time=0.0),
        finished_reason=None,
    )
    req.finished = lambda: req.finished_reason is not None

    def check_finished(new_accepted_len=1):
        assert new_accepted_len == 0
        req.finished_reason = SimpleNamespace(to_json=lambda: {"type": "length", "length": 0})

    req.check_finished = check_finished
    batch = SimpleNamespace(
        reqs=[req],
        is_prefill_only=True,
        return_logprob=False,
        decoding_reqs=[],
    )
    result = SimpleNamespace(next_token_ids=None, copy_done=None)

    scheduler.process_batch_result(batch, result)

    assert req.output_ids == []
    assert req.finished_reason.to_json() == {"type": "length", "length": 0}
    assert req.time_stats.prefill_finished_ts > 0.0
    assert req.time_stats.completion_time > 0.0
    assert seen["routed"] == [req]
    assert seen["released"] == [(req, scheduler.tree_cache)]
    assert seen["streamed"] == ([req], False, None)




def test_omni_scheduler_protects_latest_rtc_prefix_cache():
    scheduler = object.__new__(omni_scheduler.OmniScheduler)

    class FakeTreeCache:
        def __init__(self):
            self.root_node = object()
            self.matched_node = object()
            self.locked = []
            self.unlocked = []
            self.seen_key = None

        def match_prefix(self, key):
            self.seen_key = key
            return SimpleNamespace(
                device_indices=[11, 12, 13],
                last_device_node=self.matched_node,
            )

        def inc_lock_ref(self, node):
            self.locked.append(node)

        def dec_lock_ref(self, node):
            self.unlocked.append(node)

    tree_cache = FakeTreeCache()
    old_node = object()
    scheduler.tree_cache = tree_cache
    scheduler._omni_protected_prefix_nodes = {"rtc:req-0": old_node}
    req = SimpleNamespace(
        _omni_protect_latest_prefix_cache=True,
        _omni_rtc_cache_namespace="rtc:req-0",
        origin_input_ids=[1, 2],
        output_ids=[3, 4],
        extra_key="media-cache:audio=rtc:req-0:audio",
    )

    omni_scheduler.OmniScheduler._protect_omni_latest_prefix_cache(scheduler, req)

    assert tree_cache.unlocked == [old_node]
    assert tree_cache.locked == [tree_cache.matched_node]
    assert scheduler._omni_protected_prefix_nodes == {
        "rtc:req-0": [tree_cache.matched_node]
    }
    assert list(tree_cache.seen_key.token_ids) == [1, 2, 3, 4]
    assert tree_cache.seen_key.extra_key == "media-cache:audio=rtc:req-0:audio"


def test_omni_scheduler_keeps_recent_rtc_prefix_cache_depth(monkeypatch):
    scheduler = object.__new__(omni_scheduler.OmniScheduler)
    monkeypatch.setenv("QWEN35_RTC_PROTECT_PRERUN_PREFIX_CACHE_DEPTH", "2")

    class FakeTreeCache:
        def __init__(self):
            self.root_node = object()
            self.nodes = [object(), object(), object()]
            self.locked = []
            self.unlocked = []
            self.index = 0

        def match_prefix(self, key):
            del key
            node = self.nodes[self.index]
            self.index += 1
            return SimpleNamespace(device_indices=[1], last_device_node=node)

        def inc_lock_ref(self, node):
            self.locked.append(node)

        def dec_lock_ref(self, node):
            self.unlocked.append(node)

    tree_cache = FakeTreeCache()
    scheduler.tree_cache = tree_cache
    scheduler._omni_protected_prefix_nodes = {}
    req = SimpleNamespace(
        _omni_protect_latest_prefix_cache=True,
        _omni_rtc_cache_namespace="rtc:req-0",
        origin_input_ids=[1],
        output_ids=[],
        extra_key="media-cache:audio=rtc:req-0:audio",
    )

    for _ in range(3):
        omni_scheduler.OmniScheduler._protect_omni_latest_prefix_cache(scheduler, req)

    assert tree_cache.locked == tree_cache.nodes
    assert tree_cache.unlocked == [tree_cache.nodes[0]]
    assert scheduler._omni_protected_prefix_nodes == {
        "rtc:req-0": tree_cache.nodes[1:]
    }


def test_omni_scheduler_releases_protected_rtc_prefix_when_actual_finishes(
    monkeypatch,
):
    scheduler = object.__new__(omni_scheduler.OmniScheduler)
    protected_node = object()
    protected_node_2 = object()
    released = []
    scheduler.tree_cache = SimpleNamespace(
        dec_lock_ref=lambda node: released.append(node)
    )
    scheduler._omni_protected_prefix_nodes = {
        "rtc:req-0": [protected_node, protected_node_2]
    }

    req = SimpleNamespace(
        _omni_rtc_cache_namespace="rtc:req-0",
        _omni_release_protected_prefix_cache_on_finish=True,
        is_retracted=False,
        finished_reason=None,
    )
    req.finished = lambda: req.finished_reason is not None

    def fake_upstream_process_batch_result(self, batch, result):
        del self, batch, result
        req.finished_reason = object()

    monkeypatch.setattr(
        omni_scheduler._Upstream,
        "process_batch_result",
        fake_upstream_process_batch_result,
    )

    scheduler.process_batch_result(
        SimpleNamespace(reqs=[req], is_prefill_only=False),
        SimpleNamespace(next_token_ids=[1]),
    )

    assert released == [protected_node, protected_node_2]
    assert scheduler._omni_protected_prefix_nodes == {}
    assert req._omni_release_protected_prefix_cache_on_finish is False


