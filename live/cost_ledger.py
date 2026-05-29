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
