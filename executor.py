"""
CLOB V2 order executor.

Places LIMIT orders only (0% maker fee) via py-clob-client-v2.
This is the only module that touches real money.

Migrated to V2 on April 28, 2026 (Polymarket exchange upgrade).
V1 py-clob-client no longer works after this date.

Setup:
    1. Create Polymarket account (email login recommended)
    2. Export private key from https://reveal.magic.link/polymarket
    3. Set POLY_PRIVATE_KEY in .env

Usage:
    executor = ClobExecutor.from_env()      # load keys from .env
    executor = ClobExecutor.paper()          # paper mode, no keys needed
    order = executor.buy_yes(token_id, price, size)
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

log = logging.getLogger("polyweather")

HOST = "https://clob.polymarket.com"
CHAIN_ID = 137  # Polygon mainnet

# ─── Paper mode executor (no real trades) ──────────────────

@dataclass
class PaperOrder:
    token_id: str
    side: str
    price: float
    size: float
    order_id: str = "paper"
    status: str = "paper"


class PaperExecutor:
    """Simulates order placement without touching real money."""

    def buy_yes(self, token_id: str, price: float, size: float) -> PaperOrder:
        shares = round(size / price, 2)
        log.info(f"[PAPER] BUY YES {shares:.1f} shares @ ${price:.3f} = ${size:.2f}")
        return PaperOrder(token_id=token_id, side="BUY", price=price, size=size)

    def sell_yes(self, token_id: str, price: float, shares: float) -> PaperOrder:
        log.info(f"[PAPER] SELL YES {shares:.1f} shares @ ${price:.3f}")
        return PaperOrder(token_id=token_id, side="SELL", price=price, size=shares * price)

    def get_order_book(self, token_id: str) -> Optional[dict]:
        """Read-only: fetch real orderbook even in paper mode."""
        try:
            from py_clob_client_v2 import ClobClient
            client = ClobClient(host=HOST, chain_id=CHAIN_ID)
            return client.get_order_book(token_id)
        except Exception as e:
            log.debug(f"[PAPER] Orderbook fetch failed: {e}")
            return None

    @property
    def is_live(self) -> bool:
        return False


# ─── Live CLOB V2 executor ────────────────────────────────

class ClobExecutor:
    """
    Places real LIMIT GTC orders on Polymarket via py-clob-client-v2.
    All orders are LIMIT (maker) → 0% trading fee.

    V2 changes from V1:
    - No funder/signature_type params — V2 handles proxy wallets internally
    - Two-step auth: create_or_derive_api_key() → pass creds to second client
    - Side.BUY/Side.SELL enums instead of strings
    - PartialCreateOrderOptions instead of raw dict
    - pUSD collateral instead of USDC.e
    """

    def __init__(self, private_key: str):
        try:
            from py_clob_client_v2 import (
                ClobClient, ApiCreds, OrderArgs, OrderType,
                PartialCreateOrderOptions, Side, MarketOrderArgs,
            )
            self._OrderArgs = OrderArgs
            self._OrderType = OrderType
            self._Side = Side
            self._Options = PartialCreateOrderOptions
        except ImportError:
            raise ImportError(
                "pip install py_clob_client_v2\n"
                "Old py-clob-client no longer works after April 28 2026 upgrade."
            )

        # Step 1: L1 auth — derive API credentials
        client_l1 = ClobClient(host=HOST, chain_id=CHAIN_ID, key=private_key)
        creds = client_l1.create_or_derive_api_key()
        log.info("[CLOB V2] API credentials derived")

        # Step 2: L2 auth — full client with creds
        self.client = ClobClient(
            host=HOST,
            chain_id=CHAIN_ID,
            key=private_key,
            creds=creds,
        )
        log.info("[CLOB V2] Connected to Polymarket CLOB V2 (live mode)")

    def buy_yes(self, token_id: str, price: float, size: float) -> dict:
        """
        Place a LIMIT BUY order for YES shares.
        price: limit price (e.g. 0.12 = 12¢)
        size:  total USD to spend
        Returns order response dict.
        """
        shares = round(size / price, 2)
        log.info(f"[LIVE] BUY YES {shares:.1f} shares @ ${price:.3f} = ${size:.2f}")

        try:
            resp = self.client.create_and_post_order(
                order_args=self._OrderArgs(
                    token_id=token_id,
                    price=price,
                    side=self._Side.BUY,
                    size=shares,
                ),
                options=self._Options(tick_size="0.01"),
                order_type=self._OrderType.GTC,
            )
            order_id = resp.get("orderID", resp.get("id", "unknown"))
            log.info(f"[LIVE] Order placed: {order_id}")
            return resp
        except Exception as e:
            log.error(f"[LIVE] Order failed: {e}")
            return {"error": str(e)}

    def sell_yes(self, token_id: str, price: float, shares: float) -> dict:
        """Place a LIMIT SELL order for YES shares."""
        log.info(f"[LIVE] SELL YES {shares:.1f} shares @ ${price:.3f}")

        try:
            resp = self.client.create_and_post_order(
                order_args=self._OrderArgs(
                    token_id=token_id,
                    price=price,
                    side=self._Side.SELL,
                    size=shares,
                ),
                options=self._Options(tick_size="0.01"),
                order_type=self._OrderType.GTC,
            )
            order_id = resp.get("orderID", resp.get("id", "unknown"))
            log.info(f"[LIVE] Sell order placed: {order_id}")
            return resp
        except Exception as e:
            log.error(f"[LIVE] Sell order failed: {e}")
            return {"error": str(e)}

    def cancel_order(self, order_id: str) -> dict:
        """Cancel an open order."""
        try:
            return self.client.cancel(order_id)
        except Exception as e:
            log.error(f"[LIVE] Cancel failed: {e}")
            return {"error": str(e)}

    def get_open_orders(self) -> list:
        """Get all open orders."""
        try:
            return self.client.get_orders()
        except Exception as e:
            log.error(f"[LIVE] Get orders failed: {e}")
            return []

    def get_order_book(self, token_id: str) -> Optional[dict]:
        """Read-only: get real orderbook for a token."""
        try:
            return self.client.get_order_book(token_id)
        except Exception as e:
            log.debug(f"[LIVE] Orderbook fetch failed: {e}")
            return None

    def get_balance(self) -> Optional[float]:
        """Get pUSD balance."""
        try:
            bal = self.client.get_balance()
            return float(bal) if bal else None
        except Exception as e:
            log.debug(f"[LIVE] Balance fetch failed: {e}")
            return None

    @property
    def is_live(self) -> bool:
        return True

    # ─── Factory methods ───────────────────────────────────

    @classmethod
    def from_env(cls) -> 'ClobExecutor':
        """Load credentials from .env file or environment variables."""
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()

        pk = os.environ.get("POLY_PRIVATE_KEY", "")

        if not pk:
            raise ValueError(
                "Set POLY_PRIVATE_KEY in .env file.\n"
                "Get your private key from https://reveal.magic.link/polymarket"
            )

        return cls(private_key=pk)

    @classmethod
    def paper(cls) -> PaperExecutor:
        """Return paper-mode executor (no real trades)."""
        return PaperExecutor()


# ─── Quick test ────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Paper mode test
    print("=== Paper mode ===")
    ex = ClobExecutor.paper()
    ex.buy_yes("test-token-123", price=0.12, size=5.0)
    ex.sell_yes("test-token-123", price=0.45, shares=41.6)
    print(f"Is live: {ex.is_live}")

    # V2 connection test (uncomment after setting .env)
    # print("\n=== Live mode test ===")
    # ex = ClobExecutor.from_env()
    # print(f"Is live: {ex.is_live}")
    # book = ex.get_order_book("<some-token-id>")
    # print(f"Orderbook: {book}")