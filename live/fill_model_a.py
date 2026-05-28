"""
FillModelA — deterministic, queue-free fill engine for paper trading simulation.

Rules:
    MARKET BUY  → fill at tick.ask, FeeModel.TAKER, always (volume > 0)
    MARKET SELL → fill at tick.bid, FeeModel.TAKER, always (volume > 0)
    LIMIT BUY   → fill if tick.price <= limit_price, at limit_price, FeeModel.MAKER
    LIMIT SELL  → fill if tick.price >= limit_price, at limit_price, FeeModel.MAKER
    Partial     → fill_qty = min(order.remaining_qty, tick.volume)

No randomness. No queue position. No order book depth.
Goal: validate the complete simulation pipeline end-to-end.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from decimal import Decimal

from live.fee_schedule import FEE_PRECISION, MAKER_REBATE_RATE, TAKER_FEE_RATE
from live.order_types import FeeModel, Order, OrderSide, OrderState, OrderType

_TERMINAL_STATES = frozenset({
    OrderState.FILLED,
    OrderState.CANCELLED,
    OrderState.EXPIRED,
    OrderState.REJECTED,
})


@dataclass(frozen=True)
class Tick:
    """
    Single market data snapshot.

    price:  last trade price (used for LIMIT fill condition)
    bid:    best bid (MARKET SELL fills here)
    ask:    best ask (MARKET BUY fills here)
    volume: available volume at this level (caps fill_qty for partial fills)
    """
    timestamp_ms: int
    price:        Decimal
    bid:          Decimal
    ask:          Decimal
    volume:       Decimal


@dataclass(frozen=True)
class FillDecision:
    """
    Result of FillModelA.evaluate(). Describes a single fill to execute.

    fill_price:   price at which the fill executes
    fill_qty:     quantity filled (may be < order.remaining_qty for partials)
    fee_model:    MAKER or TAKER
    fee:          actual fee amount (>= 0). For MAKER this is the rebate.
    execution_id: UUID4 string — pass to osm.fill(execution_id=...) for idempotency
    event_ts_ms:  tick.timestamp_ms — canonical exchange clock for this fill
    """
    fill_price:   Decimal
    fill_qty:     Decimal
    fee_model:    FeeModel
    fee:          Decimal
    execution_id: str
    event_ts_ms:  int


class FillModelA:
    """Stateless fill engine. No persistent state."""

    @staticmethod
    def evaluate(order: Order, tick: Tick) -> FillDecision | None:
        """
        Evaluate whether order fills against tick. Returns FillDecision or None.

        Raises:
            ValueError: if order is in a terminal state.
        """
        if order.state in _TERMINAL_STATES:
            raise ValueError(
                f"Cannot evaluate fill for terminal order {order.order_id!r} "
                f"(state={order.state.value})"
            )

        if tick.volume <= Decimal("0"):
            return None

        if order.order_type == OrderType.MARKET:
            fill_price = tick.ask if order.side == OrderSide.BUY else tick.bid
            fee_model  = FeeModel.TAKER
        else:  # LIMIT
            assert order.limit_price is not None
            if order.side == OrderSide.BUY:
                if tick.price > order.limit_price:
                    return None
            else:  # SELL
                if tick.price < order.limit_price:
                    return None
            fill_price = order.limit_price
            fee_model  = FeeModel.MAKER

        fill_qty = min(order.remaining_qty, tick.volume)
        rate     = TAKER_FEE_RATE if fee_model == FeeModel.TAKER else MAKER_REBATE_RATE
        fee      = (fill_price * fill_qty * rate).quantize(FEE_PRECISION)

        return FillDecision(
            fill_price=fill_price,
            fill_qty=fill_qty,
            fee_model=fee_model,
            fee=fee,
            execution_id=str(uuid.uuid4()),
            event_ts_ms=tick.timestamp_ms,
        )
