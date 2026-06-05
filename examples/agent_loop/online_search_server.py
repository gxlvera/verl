#!/usr/bin/env python3
"""Online search server compatible with Search-R1 / veRL retriever calls.

The HTTP contract follows Search-R1's retriever server pattern:

Request:
    {"queries": ["..."], "topk": 3, "return_scores": true}

Response:
    {"result": [[{"document": {"title": "...", "contents": "...", "url": "..."}, "score": 1.0}]]}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    queries: list[str] = Field(default_factory=list)
    topk: int | None = None
    return_scores: bool = True


@dataclass(frozen=True)
class SearchResult:
    title: str
    snippet: str
    url: str
    source: str
    score: float

    def as_document(self) -> dict[str, str]:
        contents = self.snippet.strip()
        if self.title.strip():
            contents = f"{self.title.strip()}\n{contents}" if contents else self.title.strip()
        if self.url.strip():
            contents = f"{contents}\nURL: {self.url.strip()}" if contents else f"URL: {self.url.strip()}"
        return {
            "title": self.title,
            "contents": contents,
            "url": self.url,
            "source": self.source,
        }


class LruCache:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self._data: OrderedDict[tuple[str, int], list[SearchResult]] = OrderedDict()

    def get(self, key: tuple[str, int]) -> list[SearchResult] | None:
        if self.max_size <= 0 or key not in self._data:
            return None
        value = self._data.pop(key)
        self._data[key] = value
        return value

    def set(self, key: tuple[str, int], value: list[SearchResult]) -> None:
        if self.max_size <= 0:
            return
        if key in self._data:
            self._data.pop(key)
        self._data[key] = value
        while len(self._data) > self.max_size:
            self._data.popitem(last=False)


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _fetch_json(url: str, headers: dict[str, str], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            payload = response.read().decode(charset, errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"HTTP {exc.code} from search provider: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Search provider request failed: {exc.reason}") from exc
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise RuntimeError("Search provider returned non-object JSON.")
    return data


class OnlineSearchClient:
    def __init__(
        self,
        provider: str,
        topk: int,
        timeout: float,
        search_url: str,
        serp_api_key: str,
        google_api_key: str,
        google_cse_id: str,
        brave_api_key: str,
        bing_api_key: str,
    ):
        self.provider = provider.lower()
        self.topk = topk
        self.timeout = timeout
        self.search_url = search_url
        self.serp_api_key = serp_api_key
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id
        self.brave_api_key = brave_api_key
        self.bing_api_key = bing_api_key

    def search(self, query: str, topk: int) -> list[SearchResult]:
        query = query.strip()
        if not query:
            return []
        if self.provider == "serpapi":
            return self._search_serpapi(query, topk)
        if self.provider == "google":
            return self._search_google(query, topk)
        if self.provider == "brave":
            return self._search_brave(query, topk)
        if self.provider == "bing":
            return self._search_bing(query, topk)
        raise RuntimeError(f"Unsupported provider: {self.provider}")

    def _search_serpapi(self, query: str, topk: int) -> list[SearchResult]:
        if not self.serp_api_key:
            raise RuntimeError("SERPAPI_API_KEY is required for provider=serpapi.")
        params = {
            "q": query,
            "api_key": self.serp_api_key,
            "engine": _env("SERPAPI_ENGINE", "google"),
            "num": str(topk),
        }
        url = self.search_url or "https://serpapi.com/search.json"
        data = _fetch_json(f"{url}?{urllib.parse.urlencode(params)}", {}, self.timeout)
        organic = data.get("organic_results") or []
        results = []
        for rank, item in enumerate(organic[:topk], start=1):
            results.append(
                SearchResult(
                    title=str(item.get("title") or ""),
                    snippet=str(item.get("snippet") or item.get("description") or ""),
                    url=str(item.get("link") or ""),
                    source="serpapi",
                    score=1.0 / rank,
                )
            )
        return results

    def _search_google(self, query: str, topk: int) -> list[SearchResult]:
        if not self.google_api_key or not self.google_cse_id:
            raise RuntimeError("GOOGLE_API_KEY and GOOGLE_CSE_ID are required for provider=google.")
        params = {
            "key": self.google_api_key,
            "cx": self.google_cse_id,
            "q": query,
            "num": str(min(topk, 10)),
        }
        url = self.search_url or "https://www.googleapis.com/customsearch/v1"
        data = _fetch_json(f"{url}?{urllib.parse.urlencode(params)}", {}, self.timeout)
        results = []
        for rank, item in enumerate((data.get("items") or [])[:topk], start=1):
            results.append(
                SearchResult(
                    title=str(item.get("title") or ""),
                    snippet=str(item.get("snippet") or ""),
                    url=str(item.get("link") or ""),
                    source="google_cse",
                    score=1.0 / rank,
                )
            )
        return results

    def _search_brave(self, query: str, topk: int) -> list[SearchResult]:
        if not self.brave_api_key:
            raise RuntimeError("BRAVE_SEARCH_API_KEY is required for provider=brave.")
        params = {"q": query, "count": str(min(topk, 20))}
        url = self.search_url or "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.brave_api_key,
        }
        data = _fetch_json(f"{url}?{urllib.parse.urlencode(params)}", headers, self.timeout)
        web = data.get("web") or {}
        results = []
        for rank, item in enumerate((web.get("results") or [])[:topk], start=1):
            results.append(
                SearchResult(
                    title=str(item.get("title") or ""),
                    snippet=str(item.get("description") or ""),
                    url=str(item.get("url") or ""),
                    source="brave",
                    score=1.0 / rank,
                )
            )
        return results

    def _search_bing(self, query: str, topk: int) -> list[SearchResult]:
        if not self.bing_api_key:
            raise RuntimeError("BING_SEARCH_API_KEY is required for provider=bing.")
        params = {"q": query, "count": str(min(topk, 50))}
        url = self.search_url or "https://api.bing.microsoft.com/v7.0/search"
        headers = {"Ocp-Apim-Subscription-Key": self.bing_api_key}
        data = _fetch_json(f"{url}?{urllib.parse.urlencode(params)}", headers, self.timeout)
        web_pages = data.get("webPages") or {}
        results = []
        for rank, item in enumerate((web_pages.get("value") or [])[:topk], start=1):
            results.append(
                SearchResult(
                    title=str(item.get("name") or ""),
                    snippet=str(item.get("snippet") or ""),
                    url=str(item.get("url") or ""),
                    source="bing",
                    score=1.0 / rank,
                )
            )
        return results


def build_app(client: OnlineSearchClient, cache: LruCache) -> FastAPI:
    app = FastAPI(title="Search-R1 compatible online search server")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "provider": client.provider, "default_topk": client.topk}

    @app.post("/retrieve")
    async def retrieve(request: RetrieveRequest) -> dict[str, Any]:
        topk = request.topk or client.topk
        if topk <= 0:
            raise HTTPException(status_code=400, detail="topk must be positive.")

        started = time.perf_counter()
        batch: list[list[dict[str, Any]]] = []
        for query in request.queries:
            key = (query.strip(), topk)
            results = cache.get(key)
            if results is None:
                try:
                    results = await asyncio.to_thread(client.search, query, topk)
                except Exception as exc:
                    raise HTTPException(status_code=502, detail=str(exc)) from exc
                cache.set(key, results)

            items = []
            for result in results[:topk]:
                item: dict[str, Any] = {"document": result.as_document()}
                if request.return_scores:
                    item["score"] = result.score
                items.append(item)
            batch.append(items)

        return {
            "result": batch,
            "meta": {
                "provider": client.provider,
                "num_queries": len(request.queries),
                "latency_s": round(time.perf_counter() - started, 6),
            },
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default=_env("ONLINE_SEARCH_PROVIDER", "serpapi"))
    parser.add_argument("--host", default=_env("ONLINE_SEARCH_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(_env("ONLINE_SEARCH_PORT", "8000")))
    parser.add_argument("--topk", type=int, default=int(_env("ONLINE_SEARCH_TOPK", "3")))
    parser.add_argument("--timeout", type=float, default=float(_env("ONLINE_SEARCH_TIMEOUT", "15")))
    parser.add_argument("--cache-size", type=int, default=int(_env("ONLINE_SEARCH_CACHE_SIZE", "10000")))
    parser.add_argument("--search-url", default=_env("ONLINE_SEARCH_URL_OVERRIDE", ""))
    parser.add_argument("--serp-api-key", default=_env("SERPAPI_API_KEY", _env("SERP_API_KEY", "")))
    parser.add_argument("--google-api-key", default=_env("GOOGLE_API_KEY", ""))
    parser.add_argument("--google-cse-id", default=_env("GOOGLE_CSE_ID", ""))
    parser.add_argument("--brave-api-key", default=_env("BRAVE_SEARCH_API_KEY", ""))
    parser.add_argument("--bing-api-key", default=_env("BING_SEARCH_API_KEY", ""))
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = parse_args()
    client = OnlineSearchClient(
        provider=args.provider,
        topk=args.topk,
        timeout=args.timeout,
        search_url=args.search_url,
        serp_api_key=args.serp_api_key,
        google_api_key=args.google_api_key,
        google_cse_id=args.google_cse_id,
        brave_api_key=args.brave_api_key,
        bing_api_key=args.bing_api_key,
    )
    app = build_app(client, LruCache(args.cache_size))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
