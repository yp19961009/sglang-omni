# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
from collections import Counter

from sglang_omni.relay import nixl


class _FakeNixlAgent:
    def __init__(self, batches: list[dict[str, list[bytes]]]) -> None:
        self._batches = list(batches)
        self.calls = 0

    def get_new_notifs(self) -> dict[str, list[bytes]]:
        self.calls += 1
        if self._batches:
            return self._batches.pop(0)
        return {}


def _connection_with_notifications(
    batches: list[dict[str, list[bytes]]],
) -> nixl.Connection:
    conn = object.__new__(nixl.Connection)
    conn.name = "fake"
    conn._nixl = _FakeNixlAgent(batches)
    conn._remote_agents = {}
    conn._notification_lock = asyncio.Lock()
    conn._pending_notifications = Counter()
    return conn


def test_nixl_connection_caches_notifications_for_other_waiters() -> None:
    async def _drive() -> None:
        conn = _connection_with_notifications(
            [{"agent": [b"done:other", b"done:target"]}]
        )

        await conn.wait_for_notification(b"done:target", timeout=0.1)

        assert conn._pending_notifications[b"done:other"] == 1
        assert conn._nixl.calls == 1

        await conn.wait_for_notification(b"done:other", timeout=0.1)

        assert not conn._pending_notifications
        assert conn._nixl.calls == 1

    asyncio.run(_drive())


def test_nixl_put_operation_waits_for_its_unique_notification() -> None:
    async def _drive() -> None:
        conn = _connection_with_notifications(
            [
                {"agent": [b"done:unrelated"]},
                {"agent": [b"done:target"]},
            ]
        )
        released = []
        op = nixl.PutOperation(
            connection=conn,
            metadata={},
            expected_notification=b"done:target",
            on_completion_cb=lambda: released.append(True),
        )

        await op.wait_for_completion(timeout=0.1)

        assert released == [True]
        assert conn._pending_notifications[b"done:unrelated"] == 1

    asyncio.run(_drive())
