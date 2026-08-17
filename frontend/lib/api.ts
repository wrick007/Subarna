import type {
  ChatApiResponse,
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

export const api = {
  health: () => request<HealthResponse>("/api/health"),

  chat: (userId: string, message: string) =>
    request<ChatApiResponse>("/api/chat", {
      method: "POST",
      body: JSON.stringify({ user_id: userId, message }),
    }),

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
