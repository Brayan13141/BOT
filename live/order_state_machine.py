"""
Order State Machine — enforces valid order lifecycle transitions.

Stateless service. Mutates Order objects, appends to order.transitions.
Never reads from candles or infers state from market data.

Canonical clock: event_ts_ms (int64 Unix ms from exchange).
Financial quantities: Decimal.
recv_monotonic_ns: optional param, stored in transitions, NEVER in logic.

Valid transition graph:
  CREATED          → SUBMITTED                      (SUBMIT)
  SUBMITTED        → ACKNOWLEDGED                   (ACK_RECEIVED)
  SUBMITTED        → REJECTED                       (EXCHANGE_REJECT)
  SUBMITTED        → FILLED                         (IMMEDIATE_TAKER_FILL)
  SUBMITTED        → PARTIALLY_FILLED               (IMMEDIATE_TAKER_PARTIAL_FILL)
  ACKNOWLEDGED     → PARTIALLY_FILLED               (PARTIAL_FILL)
  ACKNOWLEDGED     → FILLED                         (FULL_FILL)
  ACKNOWLEDGED     → CANCELLED                      (USER_CANCEL, SYSTEM_CANCEL)
  ACKNOWLEDGED     → EXPIRED                        (TTL_EXPIRED)
  PARTIALLY_FILLED → PARTIALLY_FILLED               (PARTIAL_FILL)
  PARTIALLY_FILLED → FILLED                         (FULL_FILL)
  PARTIALLY_FILLED → CANCELLED                      (USER_CANCEL, SYSTEM_CANCEL)
  PARTIALLY_FILLED → EXPIRED                        (TTL_EXPIRED)

Terminal states (no further transitions):
  FILLED, CANCELLED, EXPIRED, REJECTED

Taker fee classification is per-fill (FillEvent.fee_model).
LiquidityRole is derived from fills — no TAKER_CONVERTED state.
"""
from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Optional

from live.order_types import (
    FeeModel,
    FillEvent,
    Order,
    OrderSide,
    OrderState,
    OrderType,
    StateTransition,
)

# ── Exceptions ────────────────────────────────────────────────────────────────


class InvalidTransitionError(Exception):
    """Raised when a requested state transition is not in the valid transition table."""


class CausalityViolationError(Exception):
    """
    Raised when event_ts_ms < last transition timestamp.
    Causal ordering of events is a hard invariant — replay determinism depends on it.
    """


class OutOfOrderEventError(InvalidTransitionError):
    """
    Raised when an exchange event arrives for a state the order hasn't reached yet.
    Common in live: fill arrives before ACK due to WS race conditions.

    Attributes:
        order_id:              identifies the affected order
        current_state:         order's state when event was received
        received_event_type:   the event type that arrived (e.g. 'PARTIAL_FILL')
        exchange_event_ts:     event_ts_ms from exchange
        local_receive_ts:      recv_monotonic_ns if available (None otherwise)
    """
    def __init__(
        self,
        order_id: str,
        current_state: OrderState,
        received_event_type: str,
        exchange_event_ts: int,
        local_receive_ts: Optional[int] = None,
    ) -> None:
        self.order_id = order_id
        self.current_state = current_state
        self.received_event_type = received_event_type
        self.exchange_event_ts = exchange_event_ts
        self.local_receive_ts = local_receive_ts
        super().__init__(
            f"Out-of-order event for order {order_id!r}: "
            f"received '{received_event_type}' while in state {current_state.value}. "
            f"exchange_event_ts={exchange_event_ts}, local_receive_ts={local_receive_ts}"
        )


class DuplicateEventError(Exception):
    """
    Raised when an execution_id has already been processed for this order.
    WS feeds duplicate execution reports — idempotency is enforced here.
    """


# ── Transition table ──────────────────────────────────────────────────────────

_VALID_TRANSITIONS: dict[tuple[OrderState, OrderState], set[str]] = {
    (OrderState.CREATED,          OrderState.SUBMITTED):        {"SUBMIT"},
    (OrderState.SUBMITTED,        OrderState.ACKNOWLEDGED):     {"ACK_RECEIVED"},
    (OrderState.SUBMITTED,        OrderState.REJECTED):         {"EXCHANGE_REJECT"},
    (OrderState.SUBMITTED,        OrderState.FILLED):           {"IMMEDIATE_TAKER_FILL"},
    (OrderState.SUBMITTED,        OrderState.PARTIALLY_FILLED): {"IMMEDIATE_TAKER_PARTIAL_FILL"},
    (OrderState.ACKNOWLEDGED,     OrderState.PARTIALLY_FILLED): {"PARTIAL_FILL"},
    (OrderState.ACKNOWLEDGED,     OrderState.FILLED):           {"FULL_FILL"},
    (OrderState.ACKNOWLEDGED,     OrderState.CANCELLED):        {"USER_CANCEL", "SYSTEM_CANCEL"},
    (OrderState.ACKNOWLEDGED,     OrderState.EXPIRED):          {"TTL_EXPIRED"},
    (OrderState.PARTIALLY_FILLED, OrderState.PARTIALLY_FILLED): {"PARTIAL_FILL"},
    (OrderState.PARTIALLY_FILLED, OrderState.FILLED):           {"FULL_FILL"},
    (OrderState.PARTIALLY_FILLED, OrderState.CANCELLED):        {"USER_CANCEL", "SYSTEM_CANCEL"},
    (OrderState.PARTIALLY_FILLED, OrderState.EXPIRED):          {"TTL_EXPIRED"},
}

_TERMINAL_STATES: frozenset[OrderState] = frozenset({
    OrderState.FILLED,
    OrderState.CANCELLED,
    OrderState.EXPIRED,
    OrderState.REJECTED,
})

# States from which fill() is valid (normal path + immediate taker path)
_FILLABLE_STATES: frozenset[OrderState] = frozenset({
    OrderState.ACKNOWLEDGED,
    OrderState.PARTIALLY_FILLED,
    OrderState.SUBMITTED,  # IMMEDIATE_TAKER_FILL path
})


# ── OrderStateMachine ─────────────────────────────────────────────────────────

class OrderStateMachine:
    """
    Stateless service that enforces order lifecycle transitions.

    Every public method:
    1. Validates causal timestamp ordering
    2. Validates the transition via _assert_transition()
    3. Updates the relevant Order fields
    4. Appends a StateTransition to order.transitions
    """

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        qty: Decimal,
        event_ts_ms: int,
        limit_price: Optional[Decimal] = None,
        order_id: Optional[str] = None,
    ) -> Order:
        """Create an Order in CREATED state. Records ORDER_CREATED transition."""
        if order_type == OrderType.LIMIT and limit_price is None:
            raise ValueError("LIMIT orders require limit_price")
        if order_type == OrderType.MARKET and limit_price is not None:
            raise ValueError("MARKET orders must not have limit_price")
        if qty <= Decimal("0"):
            raise ValueError(f"qty must be positive, got {qty}")

        oid = order_id or str(uuid.uuid4())
        order = Order(
            order_id=oid,
            symbol=symbol,
            side=side,
            order_type=order_type,
            limit_price=limit_price,
            qty=qty,
            created_ts_ms=event_ts_ms,
        )
        order.transitions.append(StateTransition(
            from_state=OrderState.CREATED,
            to_state=OrderState.CREATED,
            event_ts_ms=event_ts_ms,
            trigger="ORDER_CREATED",
        ))
        return order

    def submit(
        self,
        order: Order,
        event_ts_ms: int,
        recv_monotonic_ns: Optional[int] = None,
    ) -> None:
        """CREATED → SUBMITTED: order sent to exchange."""
        self._assert_monotonic_timestamp(order, event_ts_ms)
        self._assert_transition(order, OrderState.SUBMITTED, "SUBMIT")
        order.submitted_ts_ms = event_ts_ms
        order.state = OrderState.SUBMITTED
        order.transitions.append(StateTransition(
            from_state=OrderState.CREATED,
            to_state=OrderState.SUBMITTED,
            event_ts_ms=event_ts_ms,
            trigger="SUBMIT",
            recv_monotonic_ns=recv_monotonic_ns,
        ))

    def acknowledge(
        self,
        order: Order,
        event_ts_ms: int,
        recv_monotonic_ns: Optional[int] = None,
    ) -> None:
        """
        SUBMITTED → ACKNOWLEDGED: exchange ACK received.

        Contract: ACKNOWLEDGED is ONLY set here. It is never inferred from
        market data (candles, trade feed, price levels).
        ack_latency_ms = acknowledged_ts_ms - submitted_ts_ms
        """
        self._assert_monotonic_timestamp(order, event_ts_ms)
        if order.state != OrderState.SUBMITTED:
            raise OutOfOrderEventError(
                order_id=order.order_id,
                current_state=order.state,
                received_event_type="ACK_RECEIVED",
                exchange_event_ts=event_ts_ms,
                local_receive_ts=recv_monotonic_ns,
            )
        order.acknowledged_ts_ms = event_ts_ms
        order.state = OrderState.ACKNOWLEDGED
        order.transitions.append(StateTransition(
            from_state=OrderState.SUBMITTED,
            to_state=OrderState.ACKNOWLEDGED,
            event_ts_ms=event_ts_ms,
            trigger="ACK_RECEIVED",
            recv_monotonic_ns=recv_monotonic_ns,
        ))

    def reject(
        self,
        order: Order,
        event_ts_ms: int,
        recv_monotonic_ns: Optional[int] = None,
    ) -> None:
        """SUBMITTED → REJECTED: exchange rejected the order."""
        self._assert_monotonic_timestamp(order, event_ts_ms)
        self._assert_transition(order, OrderState.REJECTED, "EXCHANGE_REJECT")
        from_state = order.state
        order.terminal_ts_ms = event_ts_ms
        order.state = OrderState.REJECTED
        order.transitions.append(StateTransition(
            from_state=from_state,
            to_state=OrderState.REJECTED,
            event_ts_ms=event_ts_ms,
            trigger="EXCHANGE_REJECT",
            recv_monotonic_ns=recv_monotonic_ns,
        ))

    def fill(
        self,
        order: Order,
        fill_price: Decimal,
        fill_qty: Decimal,
        event_ts_ms: int,
        fee_model: FeeModel,
        execution_id: str,
        recv_monotonic_ns: Optional[int] = None,
        bid_price: Optional[Decimal] = None,
        ask_price: Optional[Decimal] = None,
    ) -> None:
        """
        Apply a fill event to an order in a fillable state.

        Doctrinal rule: TOUCH != FILL. This method is only called when an
        execution event is received from the exchange. It is never called
        because a candle's high/low touched the limit_price.

        Valid source states: ACKNOWLEDGED, PARTIALLY_FILLED, SUBMITTED (immediate taker only).

        Immediate taker path (SUBMITTED → FILLED/PARTIALLY_FILLED):
          Trigger: IMMEDIATE_TAKER_FILL (full) or IMMEDIATE_TAKER_PARTIAL_FILL (partial).

        Normal path (ACKNOWLEDGED/PARTIALLY_FILLED → FILLED/PARTIALLY_FILLED):
          Trigger: FULL_FILL (full) or PARTIAL_FILL (partial).

        LiquidityRole is derived from order.fills — not stored separately.
        A fill with FeeModel.TAKER on a previously MAKER order produces LiquidityRole.MIXED.
        """
        self._assert_monotonic_timestamp(order, event_ts_ms)

        # Idempotency
        if execution_id in order.processed_execution_ids:
            raise DuplicateEventError(
                f"execution_id {execution_id!r} already processed for order {order.order_id!r}"
            )

        if order.state not in _FILLABLE_STATES:
            if order.state in _TERMINAL_STATES:
                raise InvalidTransitionError(
                    f"Order {order.order_id!r} is in terminal state {order.state}. "
                    f"No further transitions are allowed."
                )
            raise OutOfOrderEventError(
                order_id=order.order_id,
                current_state=order.state,
                received_event_type="FILL",
                exchange_event_ts=event_ts_ms,
                local_receive_ts=recv_monotonic_ns,
            )

        if fill_qty <= Decimal("0"):
            raise ValueError(f"fill_qty must be positive, got {fill_qty}")
        if fill_qty > order.remaining_qty:
            raise ValueError(
                f"fill_qty {fill_qty} exceeds remaining_qty {order.remaining_qty}"
            )

        # Determine trigger prefix based on source state
        immediate = order.state == OrderState.SUBMITTED

        fill_event = FillEvent(
            execution_id=execution_id,
            order_id=order.order_id,
            event_ts_ms=event_ts_ms,
            price=fill_price,
            qty=fill_qty,
            fee_model=fee_model,
            recv_monotonic_ns=recv_monotonic_ns,
        )
        order.fills.append(fill_event)
        order.processed_execution_ids.add(execution_id)

        if order.first_fill_ts_ms is None:
            order.first_fill_ts_ms = event_ts_ms
        order.last_fill_ts_ms = event_ts_ms

        prev_filled = order.filled_qty
        order.filled_qty += fill_qty
        order.remaining_qty -= fill_qty

        if order.avg_fill_price is None:
            order.avg_fill_price = fill_price
        else:
            order.avg_fill_price = (
                (order.avg_fill_price * prev_filled + fill_price * fill_qty)
                / order.filled_qty
            )

        from_state = order.state
        if order.remaining_qty == Decimal("0"):
            order.terminal_ts_ms = event_ts_ms
            order.state = OrderState.FILLED
            trigger = "IMMEDIATE_TAKER_FILL" if immediate else "FULL_FILL"
            order.transitions.append(StateTransition(
                from_state=from_state,
                to_state=OrderState.FILLED,
                event_ts_ms=event_ts_ms,
                trigger=trigger,
                bid_price=bid_price,
                ask_price=ask_price,
            ))
        else:
            order.state = OrderState.PARTIALLY_FILLED
            trigger = "IMMEDIATE_TAKER_PARTIAL_FILL" if immediate else "PARTIAL_FILL"
            order.transitions.append(StateTransition(
                from_state=from_state,
                to_state=OrderState.PARTIALLY_FILLED,
                event_ts_ms=event_ts_ms,
                trigger=trigger,
                bid_price=bid_price,
                ask_price=ask_price,
            ))

    def cancel(
        self,
        order: Order,
        event_ts_ms: int,
        trigger: str = "USER_CANCEL",
        recv_monotonic_ns: Optional[int] = None,
    ) -> None:
        """ACKNOWLEDGED / PARTIALLY_FILLED → CANCELLED."""
        if trigger not in ("USER_CANCEL", "SYSTEM_CANCEL"):
            raise ValueError(
                f"Invalid cancel trigger: '{trigger}'. Must be USER_CANCEL or SYSTEM_CANCEL."
            )
        self._assert_monotonic_timestamp(order, event_ts_ms)
        self._assert_transition(order, OrderState.CANCELLED, trigger)
        from_state = order.state
        order.terminal_ts_ms = event_ts_ms
        order.state = OrderState.CANCELLED
        order.transitions.append(StateTransition(
            from_state=from_state,
            to_state=OrderState.CANCELLED,
            event_ts_ms=event_ts_ms,
            trigger=trigger,
            recv_monotonic_ns=recv_monotonic_ns,
        ))

    def expire(
        self,
        order: Order,
        event_ts_ms: int,
        recv_monotonic_ns: Optional[int] = None,
    ) -> None:
        """ACKNOWLEDGED / PARTIALLY_FILLED → EXPIRED: TTL elapsed without full fill."""
        self._assert_monotonic_timestamp(order, event_ts_ms)
        self._assert_transition(order, OrderState.EXPIRED, "TTL_EXPIRED")
        from_state = order.state
        order.terminal_ts_ms = event_ts_ms
        order.state = OrderState.EXPIRED
        order.transitions.append(StateTransition(
            from_state=from_state,
            to_state=OrderState.EXPIRED,
            event_ts_ms=event_ts_ms,
            trigger="TTL_EXPIRED",
            recv_monotonic_ns=recv_monotonic_ns,
        ))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _assert_monotonic_timestamp(self, order: Order, event_ts_ms: int) -> None:
        """
        Enforce causal ordering: event_ts_ms must be >= last recorded transition ts.
        Raises CausalityViolationError if violated.
        """
        if not order.transitions:
            return
        last_ts = order.transitions[-1].event_ts_ms
        if event_ts_ms < last_ts:
            raise CausalityViolationError(
                f"event_ts_ms {event_ts_ms} < last transition ts {last_ts} "
                f"for order {order.order_id!r}. Causal ordering violated."
            )

    def _assert_transition(
        self,
        order: Order,
        to_state: OrderState,
        trigger: str,
    ) -> None:
        if order.state in _TERMINAL_STATES:
            raise InvalidTransitionError(
                f"Order {order.order_id!r} is in terminal state {order.state}. "
                f"No further transitions are allowed."
            )
        key = (order.state, to_state)
        if key not in _VALID_TRANSITIONS or trigger not in _VALID_TRANSITIONS[key]:
            raise InvalidTransitionError(
                f"Invalid transition for order {order.order_id!r}: "
                f"{order.state} → {to_state} via trigger '{trigger}'"
            )
