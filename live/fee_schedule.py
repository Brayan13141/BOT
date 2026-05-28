"""
Fee schedule — single source of truth for exchange fee/rebate rates.

All FillModel implementations MUST import from here.
FillModelB/C may add venue-specific schedules; never duplicate these constants.

Source: Binance USDT-M perpetual futures (VIP 0).
"""
from decimal import Decimal

TAKER_FEE_RATE    = Decimal("0.0004")   # 0.04% — taker fee
MAKER_REBATE_RATE = Decimal("0.0001")   # 0.01% — maker rebate (income, stored positive)
FEE_PRECISION     = Decimal("0.00000001")
