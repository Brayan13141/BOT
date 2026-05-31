# Execution Cost Ledger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a fill-centric, append-only cost ledger that records the economic reality of every execution: fees, rebates, slippage, and reserved slots for latency and inventory costs, with a derived `net_cost` invariant enforced at every layer.

**Architecture:** `CostLedgerEntry` is a new domain event (frozen dataclass, extends `BaseEvent`) — one per fill, stored append-only in a fifth EventStore stream. `CostLedger` is a stateless class that computes `CostLedgerEntry` from fill data + `arrival_price`, and aggregates per-order summaries via `OrderCostSummary`. No mutable state anywhere.

**Tech Stack:** Python 3.12, `decimal.Decimal`, `dataclasses`, `sqlite3`, existing `live/event_models.py`, `live/serializers.py`, `live/event_store.py`, `pytest`, `tmp_path` fixture.

---

## Attribution Rules (fixed — no future changes without a new plan)

| Component | Nature | Definition |
|-----------|--------|------------|
| `arrival_price` | estimated | midprice at `OrderCreated.event_ts_ms`; caller-provided from market data |
| `slippage` | estimated | `(fill_price - arrival_price) × fill_qty` for BUY; `(arrival_price - fill_price) × fill_qty` for SELL; **positive = cost** (filled at worse price); negative = price improvement |
| `maker_rebate` | observed | `fill.fee` when `fee_model == MAKER`; `Decimal("0")` otherwise |
| `taker_fee` | observed | `fill.fee` when `fee_model == TAKER`; `Decimal("0")` otherwise |
| `latency_cost` | reserved | `Decimal("0")` in Phase 1B. Requires L2 tick data. Field exists for future use. |
| `inventory_cost` | reserved | `Decimal("0")` in Phase 1B. Future: Binance perpetual funding rate × notional × hold_hours / 8. |
| `net_cost` | derived | `taker_fee - maker_rebate + slippage + latency_cost + inventory_cost` |

**Fee convention (Binance paper trading sim):** `fill.fee` is always `>= 0`. MAKER fills produce a `maker_rebate` (reduces `net_cost`); TAKER fills produce a `taker_fee` (increases `net_cost`). They are mutually exclusive per fill.

**`net_cost` can be negative** when a MAKER fill's rebate exceeds its slippage — this represents net income from that fill. This is correct and expected.

---

## Accounting Invariant

```
net_cost = taker_fee - maker_rebate + slippage + latency_cost + inventory_cost
```

This identity MUST hold for every `CostLedgerEntry`. Tests assert it explicitly.

At order level:
```
total_net_cost = total_taker_fee - total_maker_rebate + total_slippage
              + total_latency_cost + total_inventory_cost
```

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `live/event_models.py` | Modify | Add `CostLedgerEntry` frozen dataclass |
| `live/serializers.py` | Modify | Register `CostLedgerEntry` in `_EVENT_REGISTRY` + `_DECIMAL_FIELDS` |
| `live/event_store.py` | Modify | Add `cost_ledger` SQLite table, `cost_ledger.jsonl` stream, `append_cost_entry()`, `get_cost_entries_by_order()` |
| `live/cost_ledger.py` | Create | `OrderCostSummary` (frozen dataclass) + `CostLedger` (stateless class) |
| `tests/test_cost_ledger.py` | Create | All 17 tests |

---

## Task 1: `CostLedgerEntry` event type + serializer registration

**Files:**
- Modify: `live/event_models.py`
- Modify: `live/serializers.py`
- Create: `tests/test_cost_ledger.py`

- [ ] **Step 1: Write the failing tests for CostLedgerEntry**

Create `tests/test_cost_ledger.py`:

```python
"""Tests for Execution Cost Ledger: CostLedgerEntry, CostLedger, OrderCostSummary."""
import uuid
from decimal import Decimal

import pytest

from live.event_models import CostLedgerEntry
from live.serializers import event_to_json, json_to_event


def _make_cost_entry(
    *,
    fill_id: str = "fill-001",
    order_id: str = "order-001",
    side: str = "BUY",
    fill_price: str = "65100",
    fill_qty: str = "0.01",
    arrival_price: str = "65000",
    fee_model: str = "TAKER",
    fee: str = "0.26040",
    event_ts_ms: int = 1_001_000,
) -> CostLedgerEntry:
    """Build a CostLedgerEntry via CostLedger.compute_fill_cost (once Task 3 exists).
    For Task 1, construct directly to test the dataclass itself."""
    from live.cost_ledger import CostLedger
    return CostLedger.compute_fill_cost(
        fill_id=fill_id,
        order_id=order_id,
        side=side,
        fill_price=Decimal(fill_price),
        fill_qty=Decimal(fill_qty),
        arrival_price=Decimal(arrival_price),
        fee_model=fee_model,
        fee=Decimal(fee),
        fill_ts_ms=event_ts_ms,
        event_id=str(uuid.uuid4()),
    )


# ── Task 1: CostLedgerEntry dataclass ────────────────────────────────────────

def test_cost_ledger_entry_is_frozen():
    entry = _make_cost_entry()
    with pytest.raises((AttributeError, TypeError)):
        entry.net_cost = Decimal("0")  # type: ignore[misc]


def test_cost_ledger_entry_serializes_round_trip():
    entry = _make_cost_entry(
        fill_price="65100.12345678",
        arrival_price="65000.00000001",
        fee="0.26040493",
    )
    json_str = event_to_json(entry)
    recovered = json_to_event(json_str)

    assert isinstance(recovered, CostLedgerEntry)
    assert recovered.fill_price == entry.fill_price
    assert recovered.arrival_price == entry.arrival_price
    assert recovered.slippage == entry.slippage
    assert recovered.taker_fee == entry.taker_fee
    assert recovered.maker_rebate == entry.maker_rebate
    assert recovered.net_cost == entry.net_cost


def test_cost_ledger_entry_has_correct_event_type_in_json():
    entry = _make_cost_entry()
    import json
    data = json.loads(event_to_json(entry))
    assert data["__event_type__"] == "CostLedgerEntry"
```

- [ ] **Step 2: Run tests — expect ImportError (CostLedgerEntry doesn't exist yet)**

```
cd C:/Users/Lenovo/Documents/TRADING-BOT
venv\Scripts\python -m pytest tests/test_cost_ledger.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'CostLedgerEntry'`

- [ ] **Step 3: Add `CostLedgerEntry` to `live/event_models.py`**

Append after the `PersistedFill` class (end of file):

```python


@dataclass(frozen=True, kw_only=True)
class CostLedgerEntry(BaseEvent):
    """
    Economic cost record for a single fill. Append-only.

    Attribution rules:
        arrival_price  — midprice at OrderCreated.event_ts_ms (caller-provided)
        slippage       — (fill_price - arrival_price) × fill_qty for BUY
                         (arrival_price - fill_price) × fill_qty for SELL
                         positive = cost; negative = price improvement
        maker_rebate   — fill.fee when fee_model == MAKER, else Decimal("0")
        taker_fee      — fill.fee when fee_model == TAKER, else Decimal("0")
        latency_cost   — Decimal("0") in Phase 1B (requires L2 tick data)
        inventory_cost — Decimal("0") in Phase 1B (future: funding accrual)

    Invariant (enforced by CostLedger.compute_fill_cost, tested explicitly):
        net_cost == taker_fee - maker_rebate + slippage + latency_cost + inventory_cost
    """
    order_id: str
    fill_id: str
    side: str              # OrderSide.value: "BUY" | "SELL"
    fill_price: Decimal
    fill_qty: Decimal      # always positive (unsigned)
    arrival_price: Decimal
    slippage: Decimal
    fee_model: str         # FeeModel.value: "MAKER" | "TAKER"
    maker_rebate: Decimal  # >= 0
    taker_fee: Decimal     # >= 0
    latency_cost: Decimal  # Decimal("0") in Phase 1B
    inventory_cost: Decimal  # Decimal("0") in Phase 1B
    net_cost: Decimal
```

- [ ] **Step 4: Register `CostLedgerEntry` in `live/serializers.py`**

Add to the import block (after `PersistedFill`):
```python
from live.event_models import (
    BaseEvent,
    CostLedgerEntry,          # ← add this line
    OrderAcknowledged,
    ...
)
```

Add to `_EVENT_REGISTRY` (after `"PersistedFill"`):
```python
    "CostLedgerEntry":  CostLedgerEntry,
```

Add to `_DECIMAL_FIELDS` (after `"PersistedFill"` entry):
```python
    "CostLedgerEntry": frozenset({
        "fill_price", "fill_qty", "arrival_price", "slippage",
        "maker_rebate", "taker_fee", "latency_cost", "inventory_cost", "net_cost",
    }),
```

- [ ] **Step 5: Run the 3 tests — expect ImportError for `CostLedger` (not yet created)**

```
venv\Scripts\python -m pytest tests/test_cost_ledger.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'CostLedger' from 'live.cost_ledger'` (since `_make_cost_entry` imports it). The dataclass + serializer wiring is correct at this point — we just need Task 3 to proceed.

- [ ] **Step 6: Commit**

```
git add live/event_models.py live/serializers.py tests/test_cost_ledger.py
git commit -m "feat: add CostLedgerEntry event type + serializer registration"
```

---

## Task 2: EventStore fifth stream — `cost_ledger`

**Files:**
- Modify: `live/event_store.py`
- Modify: `tests/test_cost_ledger.py`

- [ ] **Step 1: Add EventStore tests to `tests/test_cost_ledger.py`**

Append to `tests/test_cost_ledger.py`:

```python
# ── Task 2: EventStore cost stream ───────────────────────────────────────────

from pathlib import Path
from live.event_store import EventStore


def test_cost_entry_persisted_and_retrieved(tmp_path):
    entry = _make_cost_entry(order_id="order-store-001")
    with EventStore(tmp_path) as store:
        store.append_cost_entry(entry)
        retrieved = store.get_cost_entries_by_order("order-store-001")

    assert len(retrieved) == 1
    assert isinstance(retrieved[0], CostLedgerEntry)
    assert retrieved[0].fill_id == "fill-001"
    assert retrieved[0].order_id == "order-store-001"


def test_cost_entry_duplicate_event_id_rejected(tmp_path):
    import sqlite3
    entry = _make_cost_entry(order_id="order-dup-001")
    with EventStore(tmp_path) as store:
        store.append_cost_entry(entry)
        with pytest.raises(sqlite3.IntegrityError):
            store.append_cost_entry(entry)  # same event_id → UNIQUE constraint fails


def test_cost_entries_ordered_by_event_ts_ms(tmp_path):
    import uuid
    from live.cost_ledger import CostLedger
    cost_early = CostLedger.compute_fill_cost(
        fill_id="fill-early", order_id="order-ord-001", side="BUY",
        fill_price=Decimal("65100"), fill_qty=Decimal("0.01"),
        arrival_price=Decimal("65000"), fee_model="TAKER", fee=Decimal("0.26"),
        fill_ts_ms=1_000, event_id=str(uuid.uuid4()),
    )
    cost_late = CostLedger.compute_fill_cost(
        fill_id="fill-late", order_id="order-ord-001", side="BUY",
        fill_price=Decimal("65200"), fill_qty=Decimal("0.01"),
        arrival_price=Decimal("65000"), fee_model="MAKER", fee=Decimal("0.13"),
        fill_ts_ms=2_000, event_id=str(uuid.uuid4()),
    )
    with EventStore(tmp_path) as store:
        store.append_cost_entry(cost_late)   # append out-of-order
        store.append_cost_entry(cost_early)
        retrieved = store.get_cost_entries_by_order("order-ord-001")

    assert retrieved[0].fill_id == "fill-early"
    assert retrieved[1].fill_id == "fill-late"
```

- [ ] **Step 2: Run — expect `AttributeError: 'EventStore' object has no attribute 'append_cost_entry'`**

```
venv\Scripts\python -m pytest tests/test_cost_ledger.py::test_cost_entry_persisted_and_retrieved -v
```

- [ ] **Step 3: Add `cost_ledger` schema to `_SCHEMA` in `live/event_store.py`**

In `_SCHEMA`, after the `fills` table block, append:

```python
"""
...existing schema...

CREATE TABLE IF NOT EXISTS cost_ledger (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id         TEXT    NOT NULL UNIQUE,
    schema_version   INTEGER NOT NULL DEFAULT 1,
    order_id         TEXT    NOT NULL,
    fill_id          TEXT    NOT NULL UNIQUE,
    side             TEXT    NOT NULL,
    fill_price       TEXT    NOT NULL,
    fill_qty         TEXT    NOT NULL,
    arrival_price    TEXT    NOT NULL,
    slippage         TEXT    NOT NULL,
    fee_model        TEXT    NOT NULL,
    maker_rebate     TEXT    NOT NULL,
    taker_fee        TEXT    NOT NULL,
    latency_cost     TEXT    NOT NULL,
    inventory_cost   TEXT    NOT NULL,
    net_cost         TEXT    NOT NULL,
    event_ts_ms      INTEGER NOT NULL,
    causation_id     TEXT,
    correlation_id   TEXT,
    payload_json     TEXT    NOT NULL,
    created_at       INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cost_ledger_order
    ON cost_ledger(order_id, event_ts_ms);
"""
```

- [ ] **Step 4: Add `_f_costs` file handle and import to `EventStore.__init__`**

In `live/event_store.py`, add import at top of file (after existing imports):
```python
from live.event_models import (
    BaseEvent,
    CostLedgerEntry,           # ← add
    PersistedFill,
    PersistedTransition,
    RawExchangeEvent,
)
```

In `EventStore.__init__`, after `self._f_fills = open(...)`:
```python
            self._f_costs  = open(logs_dir / "cost_ledger.jsonl",    "a", encoding="utf-8")
```

In `EventStore.close()`, add `self._f_costs` to the tuple:
```python
        for f in (self._f_raw, self._f_domain, self._f_trans, self._f_fills, self._f_costs):
```

- [ ] **Step 5: Add `append_cost_entry()` and `get_cost_entries_by_order()` to EventStore**

After `append_fill()` method, insert:

```python
    def append_cost_entry(self, event: CostLedgerEntry) -> None:
        """Persist an economic cost record for a single fill."""
        json_str = event_to_json(event)
        self._conn.execute(
            """INSERT INTO cost_ledger
               (event_id, schema_version, order_id, fill_id, side,
                fill_price, fill_qty, arrival_price, slippage,
                fee_model, maker_rebate, taker_fee, latency_cost, inventory_cost,
                net_cost, event_ts_ms, causation_id, correlation_id,
                payload_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                event.event_id, event.schema_version,
                event.order_id, event.fill_id, event.side,
                str(event.fill_price), str(event.fill_qty),
                str(event.arrival_price), str(event.slippage),
                event.fee_model,
                str(event.maker_rebate), str(event.taker_fee),
                str(event.latency_cost), str(event.inventory_cost),
                str(event.net_cost),
                event.event_ts_ms, event.causation_id, event.correlation_id,
                json_str, _now_ms(),
            ),
        )
        self._conn.commit()
        self._f_costs.write(json_str + "\n")
        self._f_costs.flush()

    def get_cost_entries_by_order(self, order_id: str) -> list[CostLedgerEntry]:
        """Return all cost entries for an order, ordered by event_ts_ms ASC."""
        cur = self._conn.execute(
            """SELECT payload_json FROM cost_ledger
               WHERE order_id = ?
               ORDER BY event_ts_ms ASC, id ASC""",
            (order_id,),
        )
        return [json_to_event(row[0]) for row in cur]  # type: ignore[return-value]
```

- [ ] **Step 6: Run EventStore tests (still expect ImportError for CostLedger in `_make_cost_entry`)**

```
venv\Scripts\python -m pytest tests/test_cost_ledger.py -k "cost_entry" -v 2>&1 | head -30
```

The EventStore tests will still fail because `_make_cost_entry` can't import `CostLedger`. Task 3 fixes this.

- [ ] **Step 7: Commit**

```
git add live/event_models.py live/event_store.py live/serializers.py tests/test_cost_ledger.py
git commit -m "feat: add cost_ledger stream to EventStore"
```

---

## Task 3: `CostLedger.compute_fill_cost()` + `OrderCostSummary`

**Files:**
- Create: `live/cost_ledger.py`
- Modify: `tests/test_cost_ledger.py`

- [ ] **Step 1: Add compute_fill_cost tests to `tests/test_cost_ledger.py`**

Append:

```python
# ── Task 3: CostLedger.compute_fill_cost() ───────────────────────────────────

from live.cost_ledger import CostLedger, OrderCostSummary


def test_taker_fill_generates_fee_not_rebate():
    entry = _make_cost_entry(fee_model="TAKER", fee="0.26040")
    assert entry.taker_fee == Decimal("0.26040")
    assert entry.maker_rebate == Decimal("0")


def test_maker_fill_generates_rebate_not_fee():
    entry = _make_cost_entry(fee_model="MAKER", fee="0.13020")
    assert entry.maker_rebate == Decimal("0.13020")
    assert entry.taker_fee == Decimal("0")


def test_slippage_buy_positive_when_filled_above_arrival():
    # BUY: fill_price 65100 > arrival_price 65000 → paid more → positive slippage
    entry = _make_cost_entry(side="BUY", fill_price="65100", fill_qty="0.01",
                              arrival_price="65000")
    expected_slippage = (Decimal("65100") - Decimal("65000")) * Decimal("0.01")
    assert entry.slippage == expected_slippage  # 1.0 USDT


def test_slippage_buy_negative_when_filled_below_arrival():
    # BUY: fill_price 64900 < arrival_price 65000 → price improvement → negative slippage
    entry = _make_cost_entry(side="BUY", fill_price="64900", fill_qty="0.01",
                              arrival_price="65000")
    expected_slippage = (Decimal("64900") - Decimal("65000")) * Decimal("0.01")
    assert entry.slippage == expected_slippage  # -1.0 USDT


def test_slippage_sell_positive_when_filled_below_arrival():
    # SELL: fill_price 64900 < arrival_price 65000 → received less → positive slippage
    entry = _make_cost_entry(side="SELL", fill_price="64900", fill_qty="0.01",
                              arrival_price="65000")
    expected_slippage = (Decimal("65000") - Decimal("64900")) * Decimal("0.01")
    assert entry.slippage == expected_slippage  # 1.0 USDT


def test_slippage_sell_negative_when_filled_above_arrival():
    # SELL: fill_price 65100 > arrival_price 65000 → price improvement → negative slippage
    entry = _make_cost_entry(side="SELL", fill_price="65100", fill_qty="0.01",
                              arrival_price="65000")
    expected_slippage = (Decimal("65000") - Decimal("65100")) * Decimal("0.01")
    assert entry.slippage == expected_slippage  # -1.0 USDT


def test_net_cost_identity_taker():
    entry = _make_cost_entry(fee_model="TAKER", fee="0.26040",
                              fill_price="65100", arrival_price="65000")
    expected = (entry.taker_fee - entry.maker_rebate + entry.slippage
                + entry.latency_cost + entry.inventory_cost)
    assert entry.net_cost == expected


def test_net_cost_identity_maker():
    entry = _make_cost_entry(fee_model="MAKER", fee="0.13020",
                              fill_price="65100", arrival_price="65000")
    expected = (entry.taker_fee - entry.maker_rebate + entry.slippage
                + entry.latency_cost + entry.inventory_cost)
    assert entry.net_cost == expected


def test_net_cost_maker_can_be_negative_when_rebate_exceeds_slippage():
    # MAKER fill at arrival_price exactly → slippage = 0 → net_cost = -rebate < 0
    entry = _make_cost_entry(fee_model="MAKER", fee="0.13020",
                              fill_price="65000", arrival_price="65000",
                              fill_qty="0.01")
    assert entry.slippage == Decimal("0")
    assert entry.net_cost < Decimal("0")   # receiving income from rebate


def test_latency_cost_is_zero_phase_1b():
    entry = _make_cost_entry()
    assert entry.latency_cost == Decimal("0")


def test_inventory_cost_is_zero_phase_1b():
    entry = _make_cost_entry()
    assert entry.inventory_cost == Decimal("0")
```

- [ ] **Step 2: Run — expect `ModuleNotFoundError: No module named 'live.cost_ledger'`**

```
venv\Scripts\python -m pytest tests/test_cost_ledger.py -v 2>&1 | head -20
```

- [ ] **Step 3: Create `live/cost_ledger.py`**

```python
"""
Execution Cost Ledger — stateless cost computation for fills.

CostLedger.compute_fill_cost(): produces one CostLedgerEntry per fill.
CostLedger.summarize_order():   aggregates list[CostLedgerEntry] → OrderCostSummary.

All financial quantities: Decimal. Float prohibited.

Attribution rules (fixed — see plan 2026-05-27-execution-cost-ledger.md):
    arrival_price  = midprice at OrderCreated.event_ts_ms (caller-provided)
    slippage       = (fill_price - arrival_price) × fill_qty for BUY
                     (arrival_price - fill_price) × fill_qty for SELL
    maker_rebate   = fill.fee if MAKER, else 0
    taker_fee      = fill.fee if TAKER, else 0
    latency_cost   = Decimal("0") — Phase 1B reserved
    inventory_cost = Decimal("0") — Phase 1B reserved
    net_cost       = taker_fee - maker_rebate + slippage + latency_cost + inventory_cost
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from decimal import Decimal

from live.event_models import CostLedgerEntry

_ZERO = Decimal("0")


@dataclass(frozen=True)
class OrderCostSummary:
    """
    Aggregated costs for all fills of one order. Derived — never persisted directly.

    Invariant:
        total_net_cost == total_taker_fee - total_maker_rebate + total_slippage
                        + total_latency_cost + total_inventory_cost
    """
    order_id: str
    fill_count: int
    total_fill_qty: Decimal
    total_maker_rebate: Decimal
    total_taker_fee: Decimal
    total_slippage: Decimal
    total_latency_cost: Decimal
    total_inventory_cost: Decimal
    total_net_cost: Decimal


class CostLedger:
    """Stateless cost computation. No persistent state."""

    @staticmethod
    def compute_fill_cost(
        *,
        fill_id: str,
        order_id: str,
        side: str,            # "BUY" | "SELL"
        fill_price: Decimal,
        fill_qty: Decimal,    # always positive
        arrival_price: Decimal,
        fee_model: str,       # "MAKER" | "TAKER"
        fee: Decimal,         # actual fee from exchange (>= 0)
        fill_ts_ms: int,
        event_id: str | None = None,
        causation_id: str | None = None,
        correlation_id: str | None = None,
    ) -> CostLedgerEntry:
        """
        Compute the economic cost record for a single fill.

        Args:
            fill_id:       execution_id from exchange (matches PersistedFill.fill_id)
            order_id:      parent order identifier
            side:          "BUY" or "SELL"
            fill_price:    actual execution price
            fill_qty:      filled quantity (unsigned, positive)
            arrival_price: midprice benchmark at order creation time
            fee_model:     "MAKER" or "TAKER"
            fee:           actual fee amount >= 0 (rebate for MAKER, cost for TAKER)
            fill_ts_ms:    canonical exchange timestamp (int64 ms)
            event_id:      UUID4 string; auto-generated if None
            causation_id:  event_id of the triggering event (e.g. OrderFillReceived)
            correlation_id: order_id for grouping (convention)
        """
        if side == "BUY":
            slippage = (fill_price - arrival_price) * fill_qty
        else:  # SELL
            slippage = (arrival_price - fill_price) * fill_qty

        if fee_model == "MAKER":
            maker_rebate = fee
            taker_fee = _ZERO
        else:  # TAKER
            taker_fee = fee
            maker_rebate = _ZERO

        latency_cost = _ZERO
        inventory_cost = _ZERO
        net_cost = taker_fee - maker_rebate + slippage + latency_cost + inventory_cost

        return CostLedgerEntry(
            event_id=event_id or str(uuid.uuid4()),
            schema_version=1,
            event_ts_ms=fill_ts_ms,
            causation_id=causation_id,
            correlation_id=correlation_id,
            order_id=order_id,
            fill_id=fill_id,
            side=side,
            fill_price=fill_price,
            fill_qty=fill_qty,
            arrival_price=arrival_price,
            slippage=slippage,
            fee_model=fee_model,
            maker_rebate=maker_rebate,
            taker_fee=taker_fee,
            latency_cost=latency_cost,
            inventory_cost=inventory_cost,
            net_cost=net_cost,
        )

    @staticmethod
    def summarize_order(entries: list[CostLedgerEntry]) -> OrderCostSummary:
        """
        Aggregate all CostLedgerEntries for one order into an OrderCostSummary.

        Raises:
            ValueError: if entries is empty.
            ValueError: if entries contain more than one distinct order_id.
        """
        if not entries:
            raise ValueError("Cannot summarize empty entries list.")
        order_ids = {e.order_id for e in entries}
        if len(order_ids) > 1:
            raise ValueError(
                f"All entries must belong to one order. Got: {order_ids}"
            )

        total_fill_qty = sum((e.fill_qty for e in entries), _ZERO)
        total_maker_rebate = sum((e.maker_rebate for e in entries), _ZERO)
        total_taker_fee = sum((e.taker_fee for e in entries), _ZERO)
        total_slippage = sum((e.slippage for e in entries), _ZERO)
        total_latency_cost = sum((e.latency_cost for e in entries), _ZERO)
        total_inventory_cost = sum((e.inventory_cost for e in entries), _ZERO)
        total_net_cost = (
            total_taker_fee - total_maker_rebate + total_slippage
            + total_latency_cost + total_inventory_cost
        )

        return OrderCostSummary(
            order_id=entries[0].order_id,
            fill_count=len(entries),
            total_fill_qty=total_fill_qty,
            total_maker_rebate=total_maker_rebate,
            total_taker_fee=total_taker_fee,
            total_slippage=total_slippage,
            total_latency_cost=total_latency_cost,
            total_inventory_cost=total_inventory_cost,
            total_net_cost=total_net_cost,
        )
```

- [ ] **Step 4: Run all tests so far**

```
venv\Scripts\python -m pytest tests/test_cost_ledger.py -v 2>&1 | tail -25
```

Expected: all Task 1 + Task 3 tests PASS. Task 2 (EventStore) tests also run and should PASS now. Count: ~14 tests PASS.

- [ ] **Step 5: Commit**

```
git add live/cost_ledger.py tests/test_cost_ledger.py
git commit -m "feat: implement CostLedger.compute_fill_cost + OrderCostSummary"
```

---

## Task 4: `CostLedger.summarize_order()` — aggregation tests

**Files:**
- Modify: `tests/test_cost_ledger.py`

- [ ] **Step 1: Add aggregation tests**

Append to `tests/test_cost_ledger.py`:

```python
# ── Task 4: CostLedger.summarize_order() ─────────────────────────────────────

def test_partial_fill_cost_aggregation_two_taker_fills():
    import uuid
    fill1 = CostLedger.compute_fill_cost(
        fill_id="fill-p1", order_id="order-agg-001", side="BUY",
        fill_price=Decimal("65100"), fill_qty=Decimal("0.01"),
        arrival_price=Decimal("65000"), fee_model="TAKER", fee=Decimal("0.26040"),
        fill_ts_ms=1_000, event_id=str(uuid.uuid4()),
    )
    fill2 = CostLedger.compute_fill_cost(
        fill_id="fill-p2", order_id="order-agg-001", side="BUY",
        fill_price=Decimal("65150"), fill_qty=Decimal("0.01"),
        arrival_price=Decimal("65000"), fee_model="TAKER", fee=Decimal("0.26060"),
        fill_ts_ms=2_000, event_id=str(uuid.uuid4()),
    )
    summary = CostLedger.summarize_order([fill1, fill2])

    assert summary.order_id == "order-agg-001"
    assert summary.fill_count == 2
    assert summary.total_fill_qty == Decimal("0.02")
    assert summary.total_taker_fee == Decimal("0.26040") + Decimal("0.26060")
    assert summary.total_maker_rebate == Decimal("0")


def test_mixed_liquidity_order_costs():
    import uuid
    maker_fill = CostLedger.compute_fill_cost(
        fill_id="fill-m1", order_id="order-mixed-001", side="BUY",
        fill_price=Decimal("65000"), fill_qty=Decimal("0.01"),
        arrival_price=Decimal("65000"), fee_model="MAKER", fee=Decimal("0.13000"),
        fill_ts_ms=1_000, event_id=str(uuid.uuid4()),
    )
    taker_fill = CostLedger.compute_fill_cost(
        fill_id="fill-t1", order_id="order-mixed-001", side="BUY",
        fill_price=Decimal("65200"), fill_qty=Decimal("0.01"),
        arrival_price=Decimal("65000"), fee_model="TAKER", fee=Decimal("0.26080"),
        fill_ts_ms=2_000, event_id=str(uuid.uuid4()),
    )
    summary = CostLedger.summarize_order([maker_fill, taker_fill])

    assert summary.total_maker_rebate == Decimal("0.13000")
    assert summary.total_taker_fee == Decimal("0.26080")
    assert summary.fill_count == 2


def test_summarize_order_net_cost_identity():
    import uuid
    fill1 = CostLedger.compute_fill_cost(
        fill_id="fill-id1", order_id="order-id-001", side="SELL",
        fill_price=Decimal("64800"), fill_qty=Decimal("0.02"),
        arrival_price=Decimal("65000"), fee_model="TAKER", fee=Decimal("0.51840"),
        fill_ts_ms=1_000, event_id=str(uuid.uuid4()),
    )
    fill2 = CostLedger.compute_fill_cost(
        fill_id="fill-id2", order_id="order-id-001", side="SELL",
        fill_price=Decimal("64900"), fill_qty=Decimal("0.01"),
        arrival_price=Decimal("65000"), fee_model="MAKER", fee=Decimal("0.12980"),
        fill_ts_ms=2_000, event_id=str(uuid.uuid4()),
    )
    summary = CostLedger.summarize_order([fill1, fill2])

    expected_total_net_cost = (
        summary.total_taker_fee - summary.total_maker_rebate
        + summary.total_slippage
        + summary.total_latency_cost
        + summary.total_inventory_cost
    )
    assert summary.total_net_cost == expected_total_net_cost


def test_summarize_single_fill():
    entry = _make_cost_entry(order_id="order-single-001")
    summary = CostLedger.summarize_order([entry])
    assert summary.fill_count == 1
    assert summary.total_net_cost == entry.net_cost


def test_summarize_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        CostLedger.summarize_order([])


def test_summarize_mixed_order_ids_raises():
    import uuid
    e1 = CostLedger.compute_fill_cost(
        fill_id="fill-x1", order_id="order-A", side="BUY",
        fill_price=Decimal("65000"), fill_qty=Decimal("0.01"),
        arrival_price=Decimal("65000"), fee_model="TAKER", fee=Decimal("0.26"),
        fill_ts_ms=1_000, event_id=str(uuid.uuid4()),
    )
    e2 = CostLedger.compute_fill_cost(
        fill_id="fill-x2", order_id="order-B", side="BUY",
        fill_price=Decimal("65000"), fill_qty=Decimal("0.01"),
        arrival_price=Decimal("65000"), fee_model="TAKER", fee=Decimal("0.26"),
        fill_ts_ms=2_000, event_id=str(uuid.uuid4()),
    )
    with pytest.raises(ValueError, match="one order"):
        CostLedger.summarize_order([e1, e2])
```

- [ ] **Step 2: Run all tests**

```
venv\Scripts\python -m pytest tests/test_cost_ledger.py -v 2>&1 | tail -30
```

Expected: all tests PASS (Task 1 + 2 + 3 + 4 = ~20 tests).

- [ ] **Step 3: Commit**

```
git add tests/test_cost_ledger.py
git commit -m "test: add aggregation tests for CostLedger.summarize_order"
```

---

## Task 5: Decimal precision + net cost identity + integration (replay determinism)

**Files:**
- Modify: `tests/test_cost_ledger.py`

- [ ] **Step 1: Add the final 3 tests**

Append to `tests/test_cost_ledger.py`:

```python
# ── Task 5: Decimal precision + integration ───────────────────────────────────

def test_decimal_precision_preserved_through_event_store(tmp_path):
    """Decimal with many significant figures survives JSONL + SQLite round-trip."""
    import uuid
    precise_fee = Decimal("0.00012345678901234")
    precise_fill_price = Decimal("65123.45678901")
    precise_arrival = Decimal("65000.00000001")

    entry = CostLedger.compute_fill_cost(
        fill_id="fill-prec", order_id="order-prec-001", side="BUY",
        fill_price=precise_fill_price, fill_qty=Decimal("0.001"),
        arrival_price=precise_arrival, fee_model="TAKER", fee=precise_fee,
        fill_ts_ms=1_000, event_id=str(uuid.uuid4()),
    )

    with EventStore(tmp_path) as store:
        store.append_cost_entry(entry)
        retrieved = store.get_cost_entries_by_order("order-prec-001")

    assert len(retrieved) == 1
    r = retrieved[0]
    assert r.fill_price == precise_fill_price
    assert r.arrival_price == precise_arrival
    assert r.taker_fee == precise_fee
    assert r.slippage == entry.slippage
    assert r.net_cost == entry.net_cost


def test_net_cost_identity_holds_after_store_round_trip(tmp_path):
    """
    After persisting to EventStore and retrieving, the net_cost invariant must still hold:
    net_cost == taker_fee - maker_rebate + slippage + latency_cost + inventory_cost
    """
    import uuid
    entry = CostLedger.compute_fill_cost(
        fill_id="fill-inv", order_id="order-inv-001", side="SELL",
        fill_price=Decimal("64950"), fill_qty=Decimal("0.015"),
        arrival_price=Decimal("65000"), fee_model="MAKER", fee=Decimal("0.19485"),
        fill_ts_ms=1_000, event_id=str(uuid.uuid4()),
    )

    with EventStore(tmp_path) as store:
        store.append_cost_entry(entry)
        retrieved = store.get_cost_entries_by_order("order-inv-001")[0]

    expected_net_cost = (
        retrieved.taker_fee - retrieved.maker_rebate
        + retrieved.slippage
        + retrieved.latency_cost
        + retrieved.inventory_cost
    )
    assert retrieved.net_cost == expected_net_cost


def test_replay_reconstructs_identical_cost_ledger(tmp_path):
    """
    Recomputing CostLedgerEntry from a retrieved entry's own fields produces
    the same net_cost, slippage, fees — proving compute_fill_cost is deterministic
    and that Decimal round-trip through EventStore loses no precision.
    """
    import uuid

    original = CostLedger.compute_fill_cost(
        fill_id="fill-replay", order_id="order-replay-001", side="BUY",
        fill_price=Decimal("65150.25"), fill_qty=Decimal("0.02"),
        arrival_price=Decimal("65000"), fee_model="TAKER",
        fee=Decimal("0.52120200"),
        fill_ts_ms=1_001_000, event_id=str(uuid.uuid4()),
        correlation_id="order-replay-001",
    )

    with EventStore(tmp_path) as store:
        store.append_cost_entry(original)
        retrieved = store.get_cost_entries_by_order("order-replay-001")[0]

    # Recompute from the retrieved entry's own fields
    recomputed = CostLedger.compute_fill_cost(
        fill_id=retrieved.fill_id,
        order_id=retrieved.order_id,
        side=retrieved.side,
        fill_price=retrieved.fill_price,
        fill_qty=retrieved.fill_qty,
        arrival_price=retrieved.arrival_price,
        fee_model=retrieved.fee_model,
        # Recover fee: for TAKER fee = taker_fee; for MAKER fee = maker_rebate
        fee=retrieved.taker_fee if retrieved.fee_model == "TAKER" else retrieved.maker_rebate,
        fill_ts_ms=retrieved.event_ts_ms,
        event_id=str(uuid.uuid4()),  # new event_id — identity check below uses economic fields only
    )

    assert recomputed.net_cost == original.net_cost
    assert recomputed.slippage == original.slippage
    assert recomputed.taker_fee == original.taker_fee
    assert recomputed.maker_rebate == original.maker_rebate
    assert recomputed.latency_cost == Decimal("0")
    assert recomputed.inventory_cost == Decimal("0")
```

- [ ] **Step 2: Run the full test suite**

```
venv\Scripts\python -m pytest tests/test_cost_ledger.py -v
```

Expected output (all PASS, ~23 tests):
```
tests/test_cost_ledger.py::test_cost_ledger_entry_is_frozen PASSED
tests/test_cost_ledger.py::test_cost_ledger_entry_serializes_round_trip PASSED
tests/test_cost_ledger.py::test_cost_ledger_entry_has_correct_event_type_in_json PASSED
tests/test_cost_ledger.py::test_cost_entry_persisted_and_retrieved PASSED
tests/test_cost_ledger.py::test_cost_entry_duplicate_event_id_rejected PASSED
tests/test_cost_ledger.py::test_cost_entries_ordered_by_event_ts_ms PASSED
tests/test_cost_ledger.py::test_taker_fill_generates_fee_not_rebate PASSED
tests/test_cost_ledger.py::test_maker_fill_generates_rebate_not_fee PASSED
tests/test_cost_ledger.py::test_slippage_buy_positive_when_filled_above_arrival PASSED
tests/test_cost_ledger.py::test_slippage_buy_negative_when_filled_below_arrival PASSED
tests/test_cost_ledger.py::test_slippage_sell_positive_when_filled_below_arrival PASSED
tests/test_cost_ledger.py::test_slippage_sell_negative_when_filled_above_arrival PASSED
tests/test_cost_ledger.py::test_net_cost_identity_taker PASSED
tests/test_cost_ledger.py::test_net_cost_identity_maker PASSED
tests/test_cost_ledger.py::test_net_cost_maker_can_be_negative_when_rebate_exceeds_slippage PASSED
tests/test_cost_ledger.py::test_latency_cost_is_zero_phase_1b PASSED
tests/test_cost_ledger.py::test_inventory_cost_is_zero_phase_1b PASSED
tests/test_cost_ledger.py::test_partial_fill_cost_aggregation_two_taker_fills PASSED
tests/test_cost_ledger.py::test_mixed_liquidity_order_costs PASSED
tests/test_cost_ledger.py::test_summarize_order_net_cost_identity PASSED
tests/test_cost_ledger.py::test_summarize_single_fill PASSED
tests/test_cost_ledger.py::test_summarize_empty_raises PASSED
tests/test_cost_ledger.py::test_summarize_mixed_order_ids_raises PASSED
tests/test_cost_ledger.py::test_decimal_precision_preserved_through_event_store PASSED
tests/test_cost_ledger.py::test_net_cost_identity_holds_after_store_round_trip PASSED
tests/test_cost_ledger.py::test_replay_reconstructs_identical_cost_ledger PASSED
```

- [ ] **Step 3: Run the full existing test suite to check for regressions**

```
venv\Scripts\python -m pytest tests/ -v 2>&1 | tail -20
```

Expected: all existing 164 tests + 26 new = ~190 tests PASS, 0 FAIL.

- [ ] **Step 4: Commit**

```
git add tests/test_cost_ledger.py
git commit -m "test: Decimal precision + net_cost identity + replay determinism tests"
```

---

## Self-Review

### Spec coverage check

| Requirement from spec | Covered by |
|-----------------------|-----------|
| Cost taxonomy (gross_pnl, maker_rebate, taker_fee, slippage, latency_cost, inventory_cost, net_pnl) | `CostLedgerEntry` fields. Note: `gross_pnl`/`net_pnl` intentionally deferred to Phase 1D (requires entry+exit matching). `net_cost` replaces `net_pnl` at the fill level. |
| Decimal for everything | `CostLedgerEntry` — all financial fields are `Decimal`. `_DECIMAL_FIELDS` registered. Tests verify round-trips. |
| Granularity: fill-centric | `CostLedgerEntry` is per fill. `OrderCostSummary` is derived. ✅ |
| Accounting invariant | Documented in dataclass docstring, enforced in `compute_fill_cost()`, tested in 3 separate tests. ✅ |
| Relation with Event Store | `CostLedgerEntry` extends `BaseEvent`. `append_cost_entry()` + `get_cost_entries_by_order()`. Append-only. `UNIQUE(event_id)` not silenced. ✅ |
| Attribution rules: slippage | `arrival_price` benchmark, total USDT, signed by side. Documented + tested 4 cases. ✅ |
| Attribution rules: latency cost | `Decimal("0")` reserved field. Phase 1B constraint documented. ✅ |
| Attribution rules: inventory cost | `Decimal("0")` reserved field. Phase 1B constraint documented. ✅ |
| Replay determinism | `test_replay_reconstructs_identical_cost_ledger` — recompute from retrieved entry = identical economic fields. ✅ |
| `observed / estimated / derived / counterfactual` taxonomy | Documented in plan header table. Not enforced in code (unnecessary complexity for Phase 1B). ✅ |

### gross_pnl / net_pnl omission

Bryan's invariant was: `net_pnl = gross_pnl + maker_rebate - taker_fee - slippage - ...`

`gross_pnl` requires knowing both entry and exit fill prices — i.e., a round trip. A single fill doesn't have a realized PnL. The ledger here records **execution costs per fill**. The `net_pnl` calculation at the trade level (entry order + exit order) belongs to Phase 1D (Strategy Harness), which has the context of which orders form a round trip.

`net_cost` in this plan = `taker_fee - maker_rebate + slippage + 0 + 0` — which is the cost side of Bryan's invariant, without requiring gross_pnl. When Phase 1D computes round-trip PnL:

```
net_pnl = gross_pnl - (entry_order_summary.total_net_cost + exit_order_summary.total_net_cost)
```

This is exactly Bryan's invariant, split across two phases.

### Placeholder scan

No TBDs, no TODOs in implementation code. All test code is complete and runnable. ✅

### Type consistency

| Symbol | Defined in | Used in |
|--------|-----------|---------|
| `CostLedgerEntry` | Task 1 / event_models.py | Tasks 2, 3, 4, 5 |
| `OrderCostSummary` | Task 3 / cost_ledger.py | Tasks 4, 5 |
| `CostLedger.compute_fill_cost()` | Task 3 | Tasks 1 (`_make_cost_entry`), 4, 5 |
| `CostLedger.summarize_order()` | Task 3 | Task 4 |
| `EventStore.append_cost_entry()` | Task 2 | Tasks 2, 5 |
| `EventStore.get_cost_entries_by_order()` | Task 2 | Tasks 2, 5 |

All consistent. ✅
