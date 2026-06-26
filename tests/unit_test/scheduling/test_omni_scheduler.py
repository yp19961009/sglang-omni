# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

from sglang_omni.scheduling import omni_scheduler


def _make_scheduler(*, waiting_queue, running_batch):
    scheduler = object.__new__(omni_scheduler.OmniScheduler)
    scheduler._defer_prefill_during_priority_decode = True
    scheduler._prioritize_stream_prefill = True
    scheduler._priority_prefill_max_batch_size = 0
    scheduler._isolate_prefill_only_batches = True
    scheduler._priority_prefill_rids = set()
    scheduler.chunked_req = None
    scheduler.waiting_queue = list(waiting_queue)
    scheduler.running_batch = running_batch
    scheduler.cur_batch = None
    scheduler.last_batch = None
    scheduler._async_pending = None
    return scheduler


def test_get_new_batch_prefill_defers_while_priority_decode_runs(monkeypatch):
    priority_req = SimpleNamespace(_omni_prioritize_prefill=True)
    scheduler = _make_scheduler(
        waiting_queue=[priority_req],
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
    assert scheduler.waiting_queue == [priority_req]


def test_get_new_batch_prefill_defers_by_priority_request_id(monkeypatch):
    priority_req = SimpleNamespace(rid="waiting", _omni_prioritize_prefill=True)
    running_req = SimpleNamespace(rid="running")
    scheduler = _make_scheduler(
        waiting_queue=[priority_req],
        running_batch=SimpleNamespace(reqs=[running_req]),
    )
    scheduler._priority_prefill_rids.add("running")

    def fail_if_called(self):
        raise AssertionError("upstream prefill should be deferred")

    monkeypatch.setattr(
        omni_scheduler._Upstream, "get_new_batch_prefill", fail_if_called
    )

    assert scheduler.get_new_batch_prefill() is None
    assert scheduler.waiting_queue == [priority_req]


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


def test_get_new_batch_prefill_keeps_chunked_isolated_prefill_alone(monkeypatch):
    hidden_req = SimpleNamespace()
    priority_req = SimpleNamespace(rid="priority", _omni_prioritize_prefill=True)
    chunked_req = SimpleNamespace(_omni_isolate_prefill_batch=True)
    scheduler = _make_scheduler(
        waiting_queue=[hidden_req, priority_req],
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
    assert scheduler.waiting_queue == [hidden_req, priority_req]
    assert scheduler._priority_prefill_rids == set()
