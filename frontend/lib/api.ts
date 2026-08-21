import type {
  ChatApiResponse,
  ChatHistoryTurn,
  DeleteUserResponse,
  HealthResponse,
  ProfileResponse,
  SeedDemoResponse,
  TransactionsResponse,
} from "./types";

/**
 * Backend base URL. Set NEXT_PUBLIC_API_URL in the Vercel project (or
 * frontend/.env.local for local dev) to point at your deployed Render
 * backend -- see DEPLOYMENT.md. The NEXT_PUBLIC_ prefix is required by
 * Next.js for any env var read in browser-side code (this file runs in
 * the browser, not just on the server) -- without it, the value would
 * be undefined at runtime no matter what's set in Vercel's dashboard.
 */
export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/+$/, "") || "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    // A network-level failure (backend unreachable, CORS rejection, DNS,
    // offline) throws a plain TypeError from fetch() with no useful
    // message -- normalize it into the same ApiError shape as an HTTP
    // error response, so every caller has exactly one error type to
    // handle instead of two.
    throw new ApiError(
      `Could not reach the backend at ${API_BASE_URL}. Is it running, and is NEXT_PUBLIC_API_URL set correctly?`,
      0,
    );
  }

  if (!res.ok) {
    let detail = res.statusText || `Request failed (${res.status})`;
    try {
      const body = (await res.json()) as { detail?: string };
      if (body?.detail) detail = body.detail;
    } catch {
      // Response body wasn't JSON (e.g. a proxy's plain-text 502) -- the
      // statusText fallback above already covers this.
    }
    throw new ApiError(detail, res.status);
  }

  return (await res.json()) as T;
}

export interface ChatStreamCallbacks {
  onToken: (text: string) => void;
  onRestart: () => void;
  onDone: (response: ChatApiResponse) => void;
  onError: (message: string) => void;
}

/**
 * Consumes POST /api/chat/stream's Server-Sent Events -- see
 * backend/app/routers/chat.py's `_sse`/`_event_stream` for the exact
 * wire format parsed below (an `event:` line, a `data:` line holding
 * one JSON object, then a blank line). Deliberately not the browser's
 * built-in `EventSource`: that API can only send GET requests with no
 * body, and this endpoint needs the user's message and history in a
 * POST body -- so this reads the fetch() response body as a stream
 * directly instead and parses the same wire format by hand.
 *
 * Every event type has its own callback (rather than one generic
 * `onEvent(name, data)`) so a caller can't forget to handle "error" and
 * silently swallow a failed stream -- see ChatShell.tsx's usage.
 */
async function chatStream(
  userId: string,
  message: string,
  history: ChatHistoryTurn[],
  callbacks: ChatStreamCallbacks,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}/api/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, message, history }),
    });
  } catch {
    callbacks.onError(
      `Could not reach the backend at ${API_BASE_URL}. Is it running, and is NEXT_PUBLIC_API_URL set correctly?`,
    );
    return;
  }

  if (!res.ok || !res.body) {
    let detail = res.statusText || `Request failed (${res.status})`;
    try {
      const body = (await res.json()) as { detail?: string };
      if (body?.detail) detail = body.detail;
    } catch {
      // Response body wasn't JSON -- the statusText fallback above covers it.
    }
    callbacks.onError(detail);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // Events are separated by a blank line. A network chunk boundary can
    // land mid-event, so only fully-received events (everything up to
    // the last "\n\n" seen so far) are parsed on each pass -- a trailing
    // partial event stays in `buffer` for the next read to complete.
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const rawEvent = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);

      const lines = rawEvent.split("\n");
      const eventLine = lines.find((l) => l.startsWith("event: "));
      const dataLine = lines.find((l) => l.startsWith("data: "));

      if (eventLine && dataLine) {
        const eventName = eventLine.slice("event: ".length);
        try {
          const data = JSON.parse(dataLine.slice("data: ".length));
          if (eventName === "token") callbacks.onToken((data as { text: string }).text);
          else if (eventName === "restart") callbacks.onRestart();
          else if (eventName === "error") callbacks.onError((data as { message: string }).message);
          else if (eventName === "done") callbacks.onDone(data as ChatApiResponse);
        } catch {
          // A malformed event is dropped rather than crashing the whole
          // stream -- one bad chunk shouldn't take down an otherwise
          // fine response.
        }
      }

      boundary = buffer.indexOf("\n\n");
    }
  }
}

export const api = {
  health: () => request<HealthResponse>("/api/health"),

  chat: (userId: string, message: string, history: ChatHistoryTurn[] = []) =>
    request<ChatApiResponse>("/api/chat", {
      method: "POST",
      body: JSON.stringify({ user_id: userId, message, history }),
    }),

  chatStream,

  getProfile: (userId: string) =>
    request<ProfileResponse>(`/api/users/${encodeURIComponent(userId)}/profile`),

  getTransactions: (userId: string, limit = 100) =>
    request<TransactionsResponse>(
      `/api/users/${encodeURIComponent(userId)}/transactions?limit=${limit}`,
    ),

  deleteUser: (userId: string) =>
    request<DeleteUserResponse>(`/api/users/${encodeURIComponent(userId)}`, {
      method: "DELETE",
    }),

  seedDemoData: () =>
    request<SeedDemoResponse>("/api/users/seed-demo-data", { method: "POST" }),
};
