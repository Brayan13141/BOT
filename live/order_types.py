"""
Order types — enums and immutable data contracts for the Order State Machine.

Canonical time: event_ts_ms (int64 Unix epoch milliseconds from exchange).
recv_monotonic_ns: optional diagnostics — stored but NEVER used in logic.
Financial quantities: Decimal. Float prohibited in financial fields.

Doctrinal rules:
- TOUCH != FILL: no fill is inferred from price touching a level
- Order state derives from execution events only, never from candle data
- ACKNOWLEDGED requires exchange ACK, not market data inference
- LiquidityRole is derived from FillEvent.fee_model — never stored mutable
- All transitions are recorded in order.transitions (event-sourced)
- Terminal states: FILLED, CANCELLED, EXPIRED, REJECTED
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Optional


class OrderState(str, Enum):
    CREATED          = "CREATED"
    SUBMITTED        = "SUBMITTED"
    ACKNOWLEDGED     = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED           = "FILLED"
    CANCELLED        = "CANCELLED"
    EXPIRED          = "EXPIRED"
    REJECTED         = "REJECTED"


class OrderSide(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    LIMIT  = "LIMIT"
    MARKET = "MARKET"


class FeeModel(str, Enum):
    MAKER = "MAKER"
    TAKER = "TAKER"


class LiquidityRole(str, Enum):
    """
    Derived from fills — never stored as mutable order field.
    MAKER:  all fills have FeeModel.MAKER
    TAKER:  all fills have FeeModel.TAKER
    MIXED:  order has both maker and taker fills (e.g. partial rest + reprice)
    """
    MAKER = "MAKER"
    TAKER = "TAKER"
    MIXED = "MIXED"


@dataclass(frozen=True)
class FillEvent:
    """
    Immutable record of a single fill execution.

    execution_id: unique ID from exchange — used for idempotency.
    event_ts_ms:  canonical exchange clock (int64 ms).
    price, qty:   Decimal — financial fields never float.
    recv_monotonic_ns: optional local profiling — never enters logic.
    """
    execution_id:      str
    order_id:          str
    event_ts_ms:       int
    price:             Decimal
    qty:               Decimal
    fee_model:         FeeModel
    recv_monotonic_ns: Optional[int] = None


@dataclass(frozen=True)
class StateTransition:
    """
    Immutable record of a single state transition.

    event_ts_ms:      canonical exchange clock (int64 ms).
    bid_price/ask_price: Decimal snapshot at transition time (optional, audit only).
    recv_monotonic_ns: optional local profiling — never enters logic.
    """
    from_state:        OrderState
    to_state:          OrderState
    event_ts_ms:       int
    trigger:           str
    bid_price:         Optional[Decimal] = None
    ask_price:         Optional[Decimal] = None
    recv_monotonic_ns: Optional[int]     = None


@dataclass
class Order:
    """
    Mutable order lifecycle object. Mutated exclusively by OrderStateMachine.

    Canonical clock: event_ts_ms (int64 Unix ms from exchange).
    Financial fields: Decimal only. Float prohibited.
    recv_monotonic_ns is stored for profiling but NEVER used in any conditional.

    Latency (diagnostics only):
        ack_latency_ms = acknowledged_ts_ms - submitted_ts_ms

    WARNING: Order mutation is only legal through OrderStateMachine.
    Direct attribute assignment bypasses transition enforcement and
    breaks event-source integrity.
    """
    order_id:      str
    symbol:        str
    side:          OrderSide
    order_type:    OrderType
    limit_price:   Optional[Decimal]   # None for MARKET orders
    qty:           Decimal             # original total quantity
    created_ts_ms: int                 # when order object was created locally

    # Lifecycle timestamps — set only by OrderStateMachine
    submitted_ts_ms:    Optional[int]  = field(default=None)
    acknowledged_ts_ms: Optional[int]  = field(default=None)
    first_fill_ts_ms:   Optional[int]  = field(default=None)
    last_fill_ts_ms:    Optional[int]  = field(default=None)
    terminal_ts_ms:     Optional[int]  = field(default=None)

    # Inventory state — Decimal
    filled_qty:      Decimal           = field(default_factory=lambda: Decimal("0"))
    remaining_qty:   Decimal           = field(default_factory=lambda: Decimal("0"))
    avg_fill_price:  Optional[Decimal] = field(default=None)

    # Current state
    state: OrderState = field(default=OrderState.CREATED)

    # Event-sourced history (append-only, mutated by OSM only)
    fills:       list = field(default_factory=list)  # list[FillEvent]
    transitions: list = field(default_factory=list)  # list[StateTransition]

    # Idempotency: execution IDs seen on this order
    processed_execution_ids: set = field(default_factory=set)  # set[str]

    def __post_init__(self) -> None:
        self.remaining_qty = self.qty

    @property
    def liquidity_role(self) -> LiquidityRole:
        """
        Derived from fills — never stored mutable.

        No fills → MAKER (no taker evidence yet).
        All MAKER → MAKER.
        All TAKER → TAKER.
        Mixed    → MIXED (e.g. partial resting fill + reprice to taker).
        """
        if not self.fills:
            return LiquidityRole.MAKER
        models = {f.fee_model for f in self.fills}
        if models == {FeeModel.MAKER}:
            return LiquidityRole.MAKER
        if models == {FeeModel.TAKER}:
            return LiquidityRole.TAKER
        return LiquidityRole.MIXED
