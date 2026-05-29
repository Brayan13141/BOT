"""
Domain event models for execution event sourcing.

All events are immutable frozen dataclasses.
All financial quantities (Decimal) are stored as Python Decimal in memory.
Serialization to/from str is handled by serializers.py — never use float here.

BaseEvent fields on every event:
    event_id:        UUID4 — globally unique per event instance
    schema_version:  int — 1 for all v1 events; increment on breaking changes
    event_ts_ms:     int — canonical exchange timestamp (Unix ms, int64)
    causation_id:    str | None — event_id that triggered this event
    correlation_id:  str | None — groups related events (use order_id for order events)

Child classes use kw_only=True to allow required fields after BaseEvent's optional fields.

Doctrinal rules:
    - No mutable state. Ever.
    - No float. Ever.
    - No Order objects. Events are serializable primitives only.
    - Enum values stored as .value strings ("BUY", "MAKER", "FILLED"), not enum instances.
      This keeps event_models.py independent of order_types.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class BaseEvent:
    """Root of the event hierarchy. Every persisted event extends this."""
    event_id: str
    schema_version: int
    event_ts_ms: int
    causation_id: str | None = None
    correlation_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class RawExchangeEvent(BaseEvent):
    """
    Exact bytes/JSON received from the exchange — never normalised.
    Stored as-is for reproducible parsing and bug diagnosis.

    CRITICAL: Never derive execution decisions from this event.
    Use domain events (OrderSubmitted etc.) for all business logic.
    """
    exchange: str           # "binance"
    stream: str             # "executionReport", "!trade@arr"
    local_receive_ts_ms: int
    payload_json: str       # raw JSON string from exchange WS
    checksum: str | None = None


@dataclass(frozen=True, kw_only=True)
class OrderSubmitted(BaseEvent):
    """
    An order was created locally and the submit request was sent to exchange.

    event_ts_ms  = when osm.submit() was called (submitted to exchange).
    created_ts_ms = when osm.create_order() was called (Order object created).

    Both timestamps are needed for exact replay through the OSM.
    """
    order_id: str
    symbol: str
    side: str           # OrderSide.value: "BUY" | "SELL"
    order_type: str     # OrderType.value: "LIMIT" | "MARKET"
    qty: Decimal
    created_ts_ms: int  # for osm.create_order() replay
    limit_price: Decimal | None = None


@dataclass(frozen=True, kw_only=True)
class OrderAcknowledged(BaseEvent):
    """Exchange confirmed the order is resting in the book.

    event_ts_ms  = when ACK was received (= osm.acknowledge ts).
    submitted_ts_ms = when submit was sent (for ack_latency_ms = event_ts_ms - submitted_ts_ms).
    """
    order_id: str
    submitted_ts_ms: int


@dataclass(frozen=True, kw_only=True)
class OrderFillReceived(BaseEvent):
    """A fill (partial or full) received from exchange.

    fill_id = execution_id from exchange (used by OSM for idempotency).
    fee is the actual fee charged, in fee_asset units.
    fee_model = "MAKER" | "TAKER" (determines fee rate and liquidity_role).
    event_ts_ms = fill timestamp from exchange.
    """
    order_id: str
    fill_id: str        # execution_id → passed to osm.fill(execution_id=...)
    price: Decimal
    qty: Decimal
    fee: Decimal        # actual fee charged, as reported by exchange
    fee_asset: str      # "USDT" for USDT-margined perps
    fee_model: str      # FeeModel.value: "MAKER" | "TAKER"


@dataclass(frozen=True, kw_only=True)
class OrderCancelled(BaseEvent):
    """Order was cancelled by user or system.

    trigger = "USER_CANCEL" | "SYSTEM_CANCEL" (passed to osm.cancel).
    """
    order_id: str
    trigger: str


@dataclass(frozen=True, kw_only=True)
class OrderExpired(BaseEvent):
    """Order expired due to TTL without being fully filled."""
    order_id: str


@dataclass(frozen=True, kw_only=True)
class OrderRejected(BaseEvent):
    """Exchange rejected the order (insufficient margin, invalid params, etc.)."""
    order_id: str
    reason: str | None = None


@dataclass(frozen=True, kw_only=True)
class PersistedTransition(BaseEvent):
    """
    OSM state machine transition — audit log entry for every state change.

    event_ts_ms = same exchange timestamp as the triggering event.
    from_state / to_state = OrderState.value strings.
    trigger = OSM trigger string (e.g. "FULL_FILL", "USER_CANCEL").
    bid_price / ask_price = optional spread snapshot at transition time.
    """
    order_id: str
    from_state: str         # OrderState.value
    to_state: str           # OrderState.value
    trigger: str
    bid_price: Decimal | None = None
    ask_price: Decimal | None = None


@dataclass(frozen=True, kw_only=True)
class PersistedFill(BaseEvent):
    """
    Financial ledger entry for a single fill.

    Created AFTER the OSM processes an OrderFillReceived, so liquidity_role
    is available (derived by OSM from the full fill history of the order).
    trade_id is the exchange's trade identifier (if provided).
    """
    order_id: str
    fill_id: str                # matches OrderFillReceived.fill_id
    price: Decimal
    qty: Decimal
    fee: Decimal
    fee_asset: str
    fee_model: str              # FeeModel.value
    liquidity_role: str         # LiquidityRole.value (MAKER | TAKER | MIXED)
    trade_id: str | None = None


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
