"""Tests for Execution Cost Ledger: CostLedgerEntry, CostLedger, OrderCostSummary."""
import json
import sqlite3
import uuid
from decimal import Decimal
from pathlib import Path

import pytest

from live.cost_ledger import CostLedger, OrderCostSummary
from live.event_models import CostLedgerEntry
from live.event_store import EventStore
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
    data = json.loads(event_to_json(entry))
    assert data["__event_type__"] == "CostLedgerEntry"


# ── Task 2: EventStore cost stream ───────────────────────────────────────────

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
    entry = _make_cost_entry(order_id="order-dup-001")
    with EventStore(tmp_path) as store:
        store.append_cost_entry(entry)
        with pytest.raises(sqlite3.IntegrityError):
            store.append_cost_entry(entry)  # same event_id → UNIQUE constraint fails


def test_cost_entries_ordered_by_event_ts_ms(tmp_path):
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


# ── Task 3: CostLedger.compute_fill_cost() ───────────────────────────────────

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


# ── Task 4: CostLedger.summarize_order() ─────────────────────────────────────

def test_partial_fill_cost_aggregation_two_taker_fills():
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


# ── Task 5: Decimal precision + integration ───────────────────────────────────

def test_decimal_precision_preserved_through_event_store(tmp_path):
    """Decimal with many significant figures survives JSONL + SQLite round-trip."""
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
    """After persisting to EventStore and retrieving, net_cost invariant must still hold."""
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
