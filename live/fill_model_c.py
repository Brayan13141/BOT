"""
FillModelC — queue-aware, tape-driven fill engine for execution simulation.

Simulates fills against the aggTrades stream (ts, price, qty, side, agg_trade_id)
instead of OHLCV ticks. Kills the TOUCH != FILL fantasy that invalidated Exp08a:
a resting maker fills only when real opposing aggressor volume trades through its
level, after an explicit, calibrated queue_ahead.

Doctrine:
- The tape is ground truth. Only observed trades drive fills.
- queue_ahead is the only unobservable, made an explicit calibration knob.
- TOUCH != FILL. Maker fills require opposing volume through the level.
- No look-ahead: only trades with agg_trade_id > active_since_agg_trade_id.
- Passive-price invariant: maker fills always at limit price, never the print price.
- No L2 reconstruction, no synthetic spread, no estimated queue.

Stateless between calls; stateful only within a single evaluate() fold.
Deterministic given (order, trades, queue_ahead, active_since_agg_trade_id).
"""
from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from decimal import Decimal

from live.fee_schedule import FEE_PRECISION, MAKER_REBATE_RATE, TAKER_FEE_RATE
from live.fill_model_a import FillDecision
from live.order_types import FeeModel, Order, OrderSide, OrderState, OrderType

_TERMINAL_STATES = frozenset({
    OrderState.FILLED,
    OrderState.CANCELLED,
    OrderState.EXPIRED,
    OrderState.REJECTED,
})


@dataclass(frozen=True)
class AggTrade:
    """
    Single aggregated trade print from the exchange tape.

    side is the TAKER side: BUY = taker lifted the ask, SELL = taker hit the bid.
    agg_trade_id is strictly increasing (source guarantee) and anchors causality.
    """
    timestamp_ms: int
    price:        Decimal
    qty:          Decimal
    side:         OrderSide
    agg_trade_id: int


@dataclass(frozen=True)
class FillEvaluationResult:
    """
    Self-contained result of FillModelC.evaluate(). Contains zero L2-inference fields.

    volume_through:  total opposite-aggressor volume through P (maker); 0 for taker.
    queue_ahead:     the simulation hypothesis (model config), always present.
    queue_consumed:  how much of queue_ahead was eaten (<= queue_ahead); 0 for taker.
    queue_remaining: queue_ahead - queue_consumed.
    prints_consumed: qualifying prints the fold processed (maker + taker).
    """
    fills:                     list[FillDecision]
    active_since_agg_trade_id: int
    volume_through:            Decimal
    queue_ahead:               Decimal
    queue_consumed:            Decimal
    queue_remaining:           Decimal
    filled_qty:                Decimal
    remaining_qty:             Decimal
    fully_filled:              bool
    prints_consumed:           int


class FillModelC:
    """Queue-aware fill engine over aggTrades. Configure via queue_ahead (BTC)."""

    def __init__(
        self,
        queue_ahead: Decimal,                                    # REQUIRED, no default
        execution_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if queue_ahead < Decimal("0"):
            raise ValueError(f"queue_ahead must be >= 0, got {queue_ahead}")
        self.queue_ahead = queue_ahead
        self._execution_id_factory = execution_id_factory

    def _new_id(self) -> str:
        if self._execution_id_factory is not None:
            return self._execution_id_factory()
        return str(uuid.uuid4())

    def evaluate(
        self,
        order: Order,
        trades: Iterable[AggTrade],
        *,
        active_since_agg_trade_id: int,
    ) -> FillEvaluationResult:
        """
        Fold over trades with agg_trade_id > active_since_agg_trade_id and return fills.

        MARKET orders take the TAKER walk. (LIMIT routing added in later tasks.)

        Raises:
            ValueError: if order is in a terminal state.
        """
        if order.state in _TERMINAL_STATES:
            raise ValueError(
                f"Cannot evaluate fill for terminal order {order.order_id!r} "
                f"(state={order.state.value})"
            )

        window = [t for t in trades if t.agg_trade_id > active_since_agg_trade_id]

        if order.order_type == OrderType.MARKET:
            return self._taker_walk(order, window, active_since_agg_trade_id)
        # LIMIT routing is added in Task 3 / Task 4. Placeholder until then:
        return self._taker_walk(order, window, active_since_agg_trade_id)

    def _taker_walk(
        self,
        order: Order,
        window: list[AggTrade],
        active_since_agg_trade_id: int,
    ) -> FillEvaluationResult:
        """Same-side print walk. No queue. Fills at realized print price (TAKER)."""
        remaining = order.remaining_qty
        fills: list[FillDecision] = []
        prints_consumed = 0
        for t in window:
            if remaining <= Decimal("0"):
                break
            if t.side != order.side:          # same-side only: BUY order consumes BUY prints
                continue
            fill_qty = min(t.qty, remaining)
            fee = (t.price * fill_qty * TAKER_FEE_RATE).quantize(FEE_PRECISION)
            fills.append(FillDecision(
                fill_price=t.price,
                fill_qty=fill_qty,
                fee_model=FeeModel.TAKER,
                fee=fee,
                execution_id=self._new_id(),
                event_ts_ms=t.timestamp_ms,
            ))
            remaining -= fill_qty
            prints_consumed += 1

        filled = order.remaining_qty - remaining
        return FillEvaluationResult(
            fills=fills,
            active_since_agg_trade_id=active_since_agg_trade_id,
            volume_through=Decimal("0"),
            queue_ahead=self.queue_ahead,
            queue_consumed=Decimal("0"),
            queue_remaining=self.queue_ahead,
            filled_qty=filled,
            remaining_qty=remaining,
            fully_filled=(remaining == Decimal("0")),
            prints_consumed=prints_consumed,
        )
