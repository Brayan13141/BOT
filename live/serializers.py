"""
Serializers for execution domain events.

Responsibilities:
    - Convert event dataclasses to/from JSON strings (one line per event in JSONL).
    - Serialize Decimal as str, deserialize str back to Decimal.
    - Embed and recover __event_type__ for polymorphic deserialization.

Decimal protocol (mandatory):
    serialize:   str(decimal_value)
    deserialize: Decimal(string_value)
    NEVER use float at any point in the pipeline.

Adding a new event type:
    1. Add the class to live/event_models.py.
    2. Add an entry to _EVENT_REGISTRY below.
    3. Add the Decimal field names to _DECIMAL_FIELDS below.
    That's all — serialization is generic.
"""
from __future__ import annotations

import dataclasses
import json
from decimal import Decimal

from live.event_models import (
    BaseEvent,
    CostLedgerEntry,
    OrderAcknowledged,
    OrderCancelled,
    OrderExpired,
    OrderFillReceived,
    OrderRejected,
    OrderSubmitted,
    PersistedFill,
    PersistedTransition,
    RawExchangeEvent,
)

# ── Event type registry ───────────────────────────────────────────────────────
# Maps __event_type__ string → dataclass. Add new event types here.

_EVENT_REGISTRY: dict[str, type[BaseEvent]] = {
    "RawExchangeEvent": RawExchangeEvent,
    "OrderSubmitted":   OrderSubmitted,
    "OrderAcknowledged": OrderAcknowledged,
    "OrderFillReceived": OrderFillReceived,
    "OrderCancelled":   OrderCancelled,
    "OrderExpired":     OrderExpired,
    "OrderRejected":    OrderRejected,
    "PersistedTransition": PersistedTransition,
    "PersistedFill":    PersistedFill,
    "CostLedgerEntry":  CostLedgerEntry,
}

# ── Decimal field registry ────────────────────────────────────────────────────
# Maps event type name → set of field names that hold Decimal values.
# Optional[Decimal] fields are included here; None values are handled gracefully.

_DECIMAL_FIELDS: dict[str, frozenset[str]] = {
    "RawExchangeEvent":  frozenset(),
    "OrderSubmitted":    frozenset({"qty", "limit_price"}),
    "OrderAcknowledged": frozenset(),
    "OrderFillReceived": frozenset({"price", "qty", "fee"}),
    "OrderCancelled":    frozenset(),
    "OrderExpired":      frozenset(),
    "OrderRejected":     frozenset(),
    "PersistedTransition": frozenset({"bid_price", "ask_price"}),
    "PersistedFill":     frozenset({"price", "qty", "fee"}),
    "CostLedgerEntry":   frozenset({
        "fill_price", "fill_qty", "arrival_price", "slippage",
        "maker_rebate", "taker_fee", "latency_cost", "inventory_cost", "net_cost",
    }),
}


# ── Public API ────────────────────────────────────────────────────────────────

def event_to_json(event: BaseEvent) -> str:
    """
    Serialize a domain event to a single-line JSON string.

    - Adds __event_type__ for round-trip deserialization.
    - Converts all Decimal fields to str (preserving full precision).
    - None values are serialized as JSON null (not omitted).
    """
    event_type = type(event).__name__
    data = dataclasses.asdict(event)
    decimal_fields = _DECIMAL_FIELDS.get(event_type, frozenset())

    for field_name in decimal_fields:
        value = data.get(field_name)
        if value is not None:
            # dataclasses.asdict() returns Decimal as-is (not converted)
            data[field_name] = str(value)

    data["__event_type__"] = event_type
    return json.dumps(data, ensure_ascii=False)


def json_to_event(json_str: str) -> BaseEvent:
    """
    Deserialize a JSON string back to the correct event dataclass.

    - Reads __event_type__ to find the right class in _EVENT_REGISTRY.
    - Converts Decimal field strings back to Decimal instances.
    - Raises ValueError if __event_type__ is not registered.
    """
    data = json.loads(json_str)
    event_type = data.pop("__event_type__")
    if event_type not in _EVENT_REGISTRY:
        raise ValueError(
            f"Unknown event type {event_type!r}. "
            "Register it in _EVENT_REGISTRY and _DECIMAL_FIELDS in serializers.py."
        )
    cls = _EVENT_REGISTRY[event_type]
    assert event_type in _DECIMAL_FIELDS, (
        f"{event_type!r} is in _EVENT_REGISTRY but missing from _DECIMAL_FIELDS"
    )

    decimal_fields = _DECIMAL_FIELDS[event_type]
    for field_name in decimal_fields:
        value = data.get(field_name)
        if value is not None:
            data[field_name] = Decimal(value)

    return cls(**data)
