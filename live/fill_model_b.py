"""
FillModelB — Volume-Aware fill engine for paper trading simulation.

Extends FillModelA by limiting fill quantity to a configurable fraction
(participation_rate) of the available tick volume. Models that a retail
order competes with other participants for the same candle volume.

Design: delegates entirely to FillModelA via a capped tick.
    FillModelA: fill_qty = min(remaining_qty, tick.volume)
    FillModelB: fill_qty = min(remaining_qty, tick.volume * participation_rate)
              -> achieved by passing tick with volume=effective_liquidity to FillModelA

All fill conditions (price checks, bid/ask routing, fee model) are maintained
in FillModelA. No logic duplication.
No RNG. No queue position. Deterministic given (order, tick, participation_rate).
At participation_rate=1.0, degenerates to FillModelA behaviour.
"""
from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal

from live.fill_model_a import FillDecision, FillModelA, Tick
from live.order_types import Order


class FillModelB:
    """
    Volume-Aware fill engine. Configure via participation_rate.

    participation_rate: fraction of tick.volume available to this order.
    Default 0.01 (1%) — calibration parameter, not market-derived.
    Valid range: (0, 1]. At 1.0, equivalent to FillModelA.
    """

    def __init__(
        self,
        participation_rate: Decimal = Decimal("0.01"),  # calibration parameter — not market-derived
        execution_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if participation_rate <= Decimal("0") or participation_rate > Decimal("1"):
            raise ValueError(
                f"participation_rate must be in (0, 1], got {participation_rate}"
            )
        self.participation_rate = participation_rate
        self._execution_id_factory = execution_id_factory

    def evaluate(self, order: Order, tick: Tick) -> FillDecision | None:
        """
        Cap tick volume by participation_rate, then delegate to FillModelA.

        effective_liquidity = tick.volume * participation_rate
        All fill logic (price condition, bid/ask routing, fee) handled by FillModelA.

        Raises:
            ValueError: if order is in a terminal state (raised by FillModelA).
        """
        effective_liquidity = tick.volume * self.participation_rate
        if effective_liquidity <= Decimal("0"):
            return None
        capped = Tick(
            timestamp_ms=tick.timestamp_ms,
            price=tick.price,
            bid=tick.bid,
            ask=tick.ask,
            volume=effective_liquidity,
        )
        return FillModelA.evaluate(order, capped, execution_id_factory=self._execution_id_factory)
