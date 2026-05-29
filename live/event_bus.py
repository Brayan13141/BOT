"""
Event Bus — async pub/sub backbone for the paper trading system.

Architecture: ingest → normalize → publish → subscribers
Nothing reads directly from the exchange. Everything goes through events.
"""
import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Awaitable
import time


class EventType(str, Enum):
    TRADE = "TRADE"
    CANDLE_M1 = "CANDLE_M1"
    CANDLE_M15 = "CANDLE_M15"
    CANDLE_H1 = "CANDLE_H1"
    CANDLE_H4 = "CANDLE_H4"
    FUNDING = "FUNDING"
    HEARTBEAT = "HEARTBEAT"


@dataclass
class Event:
    type: EventType
    ts_event: int    # exchange timestamp, milliseconds UTC
    ts_local: int    # local receive time, milliseconds UTC
    data: dict = field(default_factory=dict)


Handler = Callable[["Event"], Awaitable[None]]


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[EventType, list[Handler]] = defaultdict(list)

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        self._subs[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: Handler) -> None:
        try:
            self._subs[event_type].remove(handler)
        except ValueError:
            pass

    async def publish(self, event: Event) -> None:
        for handler in self._subs.get(event.type, []):
            try:
                await handler(event)
            except Exception as exc:
                print(f"[EventBus] handler error on {event.type}: {exc}")

    @staticmethod
    def now_ms() -> int:
        return int(time.time() * 1000)
