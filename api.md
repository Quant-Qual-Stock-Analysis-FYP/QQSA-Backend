# Backend API Documentation

**Base URL:** `/api/`  
**Auth:** JWT (Bearer) where noted. Analysis and markets endpoints have no explicit auth in code; chatbot and user profile require auth.

---

## 1. Users (`/api/users/`)

| Method | Endpoint | Function | Auth | Payload | Return |
|--------|----------|----------|------|---------|--------|
| **POST** | `/api/users/register/` | Register new user | No | `application/json`: `username`, `email`, `password` | 201: User object `{ id, username, email }` (no password). 400: validation errors. |
| **POST** | `/api/users/login/` | Obtain JWT | No | `application/json`: `username`, `password` | 200: `{ access, refresh }` (JWT strings). 401: invalid credentials. |
| **POST** | `/api/users/token/refresh/` | Refresh access token | No | `application/json`: `refresh` (refresh token) | 200: `{ access }`. 401: invalid/expired refresh. |
| **GET** | `/api/users/profile/` | Get current user | Yes (JWT) | — | 200: `{ id, username, email }`. 401: unauthorized. |
| **PUT** / **PATCH** | `/api/users/profile/` | Update current user | Yes (JWT) | `application/json`: `username`, `email`, `password` (optional) | 200: updated user `{ id, username, email }`. 400: validation errors. |

---

## 2. Chat (`/api/chat/`)

All chat endpoints require **JWT** (Bearer).

| Method | Endpoint | Function | Payload | Return |
|--------|----------|----------|---------|--------|
| **GET** | `/api/chat/rooms/` | List current user’s chat rooms | — | 200: Array of `{ id, title, created_at, updated_at, message_count, last_message }`. `last_message`: `{ content, role, timestamp }` or null. |
| **POST** | `/api/chat/rooms/` | Create chat room | `application/json`: `title` (optional) | 201: `{ id, title, created_at, updated_at }`. |
| **GET** | `/api/chat/rooms/<pk>/` | Get chat room with messages | — | 200: `{ id, title, created_at, updated_at, messages, message_count }`. `messages`: array of `{ id, role, content, timestamp }`. 404: not found. |
| **PUT** / **PATCH** | `/api/chat/rooms/<pk>/` | Update chat room | `application/json`: `title` | 200: updated room. 400/404. |
| **DELETE** | `/api/chat/rooms/<pk>/` | Delete chat room | — | 204 or 200. 400: cannot delete last room. |
| **GET** | `/api/chat/rooms/<room_id>/messages/` | Get messages in room | — | 200: Array of `{ id, role, content, timestamp }`. 404: room not found. |
| **POST** | `/api/chat/rooms/<room_id>/messages/` | Send message, get AI reply | `application/json`: `question` (string) | 200: `{ question, answer }`. 400: missing `question`. 404: room not found. |
| **DELETE** | `/api/chat/messages/<pk>/` | Delete one message | — | 200: `{ message: "Message deleted successfully" }`. 404: not found. |
| **GET** | `/api/chat/recommended-questions/` | Get recommended questions | — | 200: `{ questions: [ { id, question }, ... ] }`. |

---

## 3. Markets (`/api/`)

| Method | Endpoint | Function | Payload | Return |
|--------|----------|----------|---------|--------|
| **GET** | `/api/stocks/` | List all stocks | — | 200: Array of `{ id, symbol, name, sector, details, last_updated, news }`. `news`: array of `{ title, link, publisher, icon_url, created_at }`. |
| **GET** | `/api/stocks/<id>/` | Get stock by primary key | — | 200: Single stock object (same shape). 404: not found. |
| **GET** | `/api/stocks/by_symbol/?symbol=<SYMBOL>` | Get stock by symbol | Query: `symbol` (required) | 200: Single stock object. 400: missing symbol. 404: not found. |
| **GET** | `/api/stocks/chart/?symbol=<SYMBOL>` | Get 1D intraday chart | Query: `symbol` (required) | 200: `{ meta: { symbol, market_status, is_market_open, data_date, range_desc, server_timestamp }, data: [ { time, price }, ... ] }`. 400: missing symbol. 404/500: not found or Yahoo error. |

---

## 4. Analysis (`/api/analysis/`)

| Method | Endpoint | Function | Payload | Return |
|--------|----------|----------|---------|--------|
| **GET** | `/api/analysis/results/` | List all stored analysis results | — | 200: Array of analysis objects. Each item = full analysis payload below + `stock_rank`, `etf_rank`. |
| **GET** | `/api/analysis/results/<symbol>/` | Get analysis for one symbol | Path: `symbol` (e.g. AAPL) | 200: Single analysis object (see **Analysis object**). 400: invalid symbol. 404: stock or analysis not found. |
| **GET** | `/api/analysis/scores/` | List scores for leaderboard | — | 200: Array of `{ stock_symbol, stock_rank, etf_rank, overall_score, risk_score, sentiment_score, technical_score, fundamental_score }`. |
| **GET** | `/api/analysis/market-context/` | Get market context | — | 200: `{ market_cycle, market_score, market_bias, market_reason }`. `market_cycle`: "Bull" \| "Bear" \| "Base". `market_reason`: array of strings. |
| **POST** | `/api/analysis/portfolio/` | Sparse portfolio optimisation | `application/json`: `top_n` (required, int), `horizon` (optional, "short" \| "mid" \| "long"), `use_hedge` (optional, bool), `risk_level` (optional, "low" \| "mid" \| "high") | 200: Portfolio result (see **Portfolio result**). 400: validation error (e.g. missing `top_n`). |
| **POST** | `/api/analysis/documents/sentiment/` | Upload RAG sentiment doc | **Option A** `multipart/form-data`: `symbol`, `file` (.docx or .pdf). **Option B** `application/json`: `symbol`, `content` | 201: `{ status: "ok" }`. 400: invalid symbol/content/file. 404: stock not found. |
| **POST** | `/api/analysis/documents/fundamental/` | Upload RAG fundamental doc | Same as sentiment | 201: `{ status: "ok" }`. Same errors. |
| **GET** | `/api/analysis/documents/sentiment/list/` | List manual sentiment docs | — | 200: Array of `{ id, symbol, filename, source, created_at }`. |
| **GET** | `/api/analysis/documents/fundamental/list/` | List manual fundamental docs | — | 200: Same shape as sentiment list. |
| **DELETE** | `/api/analysis/documents/sentiment/<doc_id>/` | Delete sentiment doc | Path: `doc_id` (int) | 200: `{ status: "ok" }`. 404: not found or not manual. |
| **DELETE** | `/api/analysis/documents/fundamental/<doc_id>/` | Delete fundamental doc | Path: `doc_id` (int) | 200: `{ status: "ok" }`. 404: same. |

---

### Analysis object (single symbol)

Returned by `GET /api/analysis/results/<symbol>/` and each element of `GET /api/analysis/results/`.

```json
{
  "tags": ["string"],
  "label": "stock" | "etf",
  "conflict_note": "string | null",
  "scores": {
    "stability": 0-100,
    "sentiment": 0-100,
    "technical": 0-100,
    "fundamental": 0-100,
    "overall_score": 0-100
  },
  "signal": "Strong Buy" | "Buy" | "Hold" | "Sell",
  "symbol": "SYMBOL",
  "summary": "string",
  "analysis": {
    "stability": { "score", "reason", "data": { "score", "detail", "formula" } },
    "sentiment": { "score", "reason", "data": { "sentiment", "score", "source", "reason" } },
    "technical": { "score", "reason", "data": { "score", "status", "indicators", "formula", "alpha": { "name", "expression", "score" } } },
    "fundamental": { "score", "reason", "data": { "sentiment", "score", "source", "reason" } }
  },
  "strategy": "string"
}
```

Plus `stock_rank`, `etf_rank` when from list/detail.

---

### Portfolio result

Returned by `POST /api/analysis/portfolio/`.

```json
{
  "inputs": {
    "top_n": 5,
    "horizon": "mid",
    "use_hedge": true,
    "risk_level": "mid"
  },
  "portfolio": [
    { "symbol": "AAPL", "weight": 0.18, "sector": "Technology", "role": "Alpha Generator" }
  ],
  "allocation_analysis": {
    "sector_distribution": { "Technology": 0.4, "Healthcare": 0.3, ... },
    "is_concentrated": false,
    "dominant_sector": "Technology"
  },
  "analysis": {
    "scenario_analysis": { "bull": 12.5, "base": 8.0, "bear": -5.2 }
  },
  "summary": {
    "cvar_5pct": 2.1,
    "hedge_symbol": "TLT",
    "portfolio_risk": "Medium",
    "explanation": "string"
  }
}
```

---

## 5. Error responses

- **400:** `{ "error": "message" }` (validation, bad request).
- **404:** `{ "error": "message" }` (not found).
- **500:** `{ "error": "message" }` (server error).

---

## 6. Auth header (where required)

```http
Authorization: Bearer <access_token>
```

Use the `access` value from `POST /api/users/login/` or `POST /api/users/token/refresh/`.

---

You can paste this into `docs/API.md` or append it to your README; adjust base URL or auth notes if your project uses a global auth or different prefix.