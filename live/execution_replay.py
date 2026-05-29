"""
Execution Replay — reconstruct Order state from persisted domain events.

Replays domain events (OrderSubmitted, OrderAcknowledged, etc.) through a fresh
OrderStateMachine instance to rebuild the complete Order lifecycle.

Guarantees:
    - Deterministic: same events → same Order state, every time.
    - No side effects: replay never writes to EventStore.
    - Isolated: uses a fresh OSM per call; no shared mutable state.
    - Decimal-exact: financial quantities preserve full precision.

This module depends on:
    - live.event_models: domain event dataclasses
    - live.event_store: EventStore.get_domain_events_by_order()
    - live.order_types: enums (OrderSide, OrderType, FeeModel)
    - live.order_state_machine: OrderStateMachine

It does NOT depend on live.storage (market data) or live.replay (market replay).
"""
from __future__ import annotations

from live.event_models import (
    BaseEvent,
    OrderAcknowledged,
    OrderCancelled,
    OrderExpired,
    OrderFillReceived,
    OrderRejected,
    OrderSubmitted,
)
from live.event_store import EventStore
from live.order_state_machine import OrderStateMachine
from live.order_types import FeeModel, Order, OrderSide, OrderType


class ExecutionReplay:
    """
    Reconstruct an Order's state by replaying its domain events through OSM.

    Usage:
        replay = ExecutionReplay(event_store=store)
        order = replay.replay_order("order-001")
        assert order.state == OrderState.FILLED

    The replay algorithm:
        1. Fetch all domain events for order_id, ordered by event_ts_ms ASC.
        2. For each event, call the corresponding OSM method.
        3. Return the fully reconstructed Order.

    OrderSubmitted drives TWO OSM calls:
        - osm.create_order(..., event_ts_ms=event.created_ts_ms)
        - osm.submit(order, event_ts_ms=event.event_ts_ms)
      Both timestamps are stored in OrderSubmitted specifically for this replay.
    """

    def __init__(self, event_store: EventStore) -> None:
        self._store = event_store

    def replay_order(self, order_id: str) -> Order:
        """
        Reconstruct the Order lifecycle from persisted domain events.

        Raises:
            ValueError: if no domain events are found for order_id.
            InvalidTransitionError / CausalityViolationError: if stored events
                are inconsistent (should never happen with a healthy event log).
        """
        events: list[BaseEvent] = self._store.get_domain_events_by_order(order_id)
        if not events:
            raise ValueError(
                f"No domain events found for order_id={order_id!r}. "
                "Cannot replay an order with no history."
            )

        osm = OrderStateMachine()
        order: Order | None = None

        for event in events:
            if isinstance(event, OrderSubmitted):
                order = osm.create_order(
                    symbol=event.symbol,
                    side=OrderSide(event.side),
                    order_type=OrderType(event.order_type),
                    qty=event.qty,
                    limit_price=event.limit_price,
                    event_ts_ms=event.created_ts_ms,
                    order_id=event.order_id,
                )
                osm.submit(order, event_ts_ms=event.event_ts_ms)

            elif isinstance(event, OrderAcknowledged):
                assert order is not None
                osm.acknowledge(order, event_ts_ms=event.event_ts_ms)

            elif isinstance(event, OrderFillReceived):
                assert order is not None
                osm.fill(
                    order,
                    fill_price=event.price,
                    fill_qty=event.qty,
                    event_ts_ms=event.event_ts_ms,
                    fee_model=FeeModel(event.fee_model),
                    execution_id=event.fill_id,
                )

            elif isinstance(event, OrderCancelled):
                assert order is not None
                osm.cancel(order, event_ts_ms=event.event_ts_ms, trigger=event.trigger)

            elif isinstance(event, OrderExpired):
                assert order is not None
                osm.expire(order, event_ts_ms=event.event_ts_ms)

            elif isinstance(event, OrderRejected):
                assert order is not None
                osm.reject(order, event_ts_ms=event.event_ts_ms)

        assert order is not None  # guaranteed since events is non-empty
        return order
