import os
import unittest
from unittest.mock import AsyncMock, patch

from metac_bot.infrastructure import market_data


class MarketDataTests(unittest.IsolatedAsyncioTestCase):
    async def test_yahoo_daily_series_has_dated_values(self) -> None:
        response = {
            "chart": {
                "result": [
                    {
                        "meta": {
                            "currency": "USD",
                            "exchangeTimezoneName": "America/New_York",
                        },
                        "timestamp": [0, 86400],
                        "indicators": {"quote": [{"close": [1.5, None]}]},
                    }
                ]
            }
        }
        with patch.object(
            market_data, "_get", new=AsyncMock(return_value=response)
        ) as get:
            result = await market_data.yahoo_series("AAPL")
        self.assertEqual(
            result["observations"],
            [{"time": "1970-01-01T00:00:00+00:00", "close": 1.5}],
        )
        get.assert_awaited_once_with(
            "https://query1.finance.yahoo.com/v8/finance/chart/AAPL",
            {"range": "1y", "interval": "1d"},
        )
        self.assertIn("in progress", result["note"])

    async def test_fred_requires_key_and_never_returns_it(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "FRED_API_KEY"):
                await market_data.fred_series("CPIAUCSL")
        with (
            patch.dict(os.environ, {"FRED_API_KEY": "sensitive"}, clear=True),
            patch.object(
                market_data,
                "_get",
                new=AsyncMock(
                    return_value={
                        "observations": [{"date": "2026-01-01", "value": "2.5"}]
                    }
                ),
            ) as get,
        ):
            result = await market_data.fred_series("CPIAUCSL")
        get.assert_awaited_once_with(
            "https://api.stlouisfed.org/fred/series/observations",
            {
                "series_id": "CPIAUCSL",
                "api_key": "sensitive",
                "file_type": "json",
                "sort_order": "desc",
                "limit": 100,
            },
        )
        self.assertNotIn("sensitive", str(result))
        self.assertEqual(result["observations"][0]["value"], "2.5")

    async def test_market_failures_do_not_hide_other_sources(self) -> None:
        async def fetch(source: str, query: str) -> dict:
            if source == "predictit":
                raise RuntimeError("secret")
            return {"provider": source, "candidates": []}

        with patch.object(market_data, "_market_source", side_effect=fetch):
            result = await market_data.prediction_markets("election")
        self.assertEqual(
            result["sources"][2], {"provider": "predictit", "error": "RuntimeError"}
        )
        self.assertNotIn("secret", str(result))
        self.assertIn("unverified", result["note"])

    async def test_kalshi_keeps_distinct_bid_and_ask(self) -> None:
        with patch.object(
            market_data,
            "_get",
            new=AsyncMock(
                return_value={
                    "market": {
                        "ticker": "ABC",
                        "yes_bid_dollars": "0.20",
                        "yes_ask_dollars": "0.30",
                    }
                }
            ),
        ):
            result = await market_data.kalshi_market("ABC")
        self.assertEqual(result["market"]["yes_bid_dollars"], "0.20")
        self.assertEqual(result["market"]["yes_ask_dollars"], "0.30")

    async def test_invalid_identifier_is_rejected_before_network(self) -> None:
        with patch.object(market_data, "_get") as get:
            with self.assertRaises(ValueError):
                await market_data.yahoo_series("../../foo")
            get.assert_not_called()

    async def test_provider_candidate_parsing(self) -> None:
        async def get(url: str, params: dict | None = None) -> object:
            if "kalshi" in url:
                return {
                    "events": [
                        {
                            "title": "Election winner?",
                            "event_ticker": "E",
                            "markets": [{"ticker": "E-X", "yes_bid_dollars": "0.40"}],
                        }
                    ],
                    "cursor": "",
                }
            if "manifold" in url:
                return [
                    {
                        "question": "Election?",
                        "probability": 0.42,
                        "url": "https://manifold.markets/example",
                    }
                ]
            if "polymarket" in url:
                return {
                    "events": [
                        {
                            "title": "Election",
                            "slug": "election",
                            "markets": [
                                {
                                    "question": "Election?",
                                    "outcomes": '["Yes", "No"]',
                                    "outcomePrices": '["0.4", "0.6"]',
                                }
                            ],
                        }
                    ]
                }
            return {
                "markets": [
                    {
                        "name": "Election winner?",
                        "contracts": [{"lastTradePrice": 0.55}],
                    },
                    {"name": "Unrelated"},
                ]
            }

        with patch.object(market_data, "_get", side_effect=get):
            result = await market_data.prediction_markets("Election")
        manifold, polymarket, predictit, kalshi = result["sources"]
        self.assertEqual(manifold["candidates"][0]["probability"], 0.42)
        self.assertEqual(
            polymarket["candidates"][0]["url"], "https://polymarket.com/event/election"
        )
        self.assertEqual(
            predictit["candidates"][0]["contracts"][0]["lastTradePrice"], 0.55
        )
        self.assertEqual(kalshi["candidates"][0]["markets"][0]["ticker"], "E-X")
        self.assertFalse(kalshi["catalogue_incomplete"])

    async def test_kalshi_scan_is_bounded_and_discloses_truncation(self) -> None:
        with patch.object(
            market_data,
            "_get",
            new=AsyncMock(return_value={"events": [], "cursor": "next"}),
        ) as get:
            result = await market_data._market_source("kalshi", "election")
        self.assertEqual(get.await_count, 3)
        self.assertTrue(result["catalogue_incomplete"])
