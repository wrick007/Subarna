/**
 * Wire types mirroring backend/app/api_schemas.py exactly. Kept as a
 * single hand-written file rather than codegen: the API surface is
 * small and stable enough that a generator would be more ceremony than
 * the six interfaces below, and any drift is caught immediately (a
 * missing/renamed field fails to compile, not fails silently at
 * runtime) the moment a component tries to read a field that isn't here.
 */

export interface EvidenceItem {
  source_id: string;
  date: string;
  description: string;
  amount: number;
  currency: string;
  category: string;
  document: string;
  relevance: number;
  retrieval_stage: string;
  keyword_score: number | null;
  vector_score: number | null;
  rerank_score: number | null;
}

export interface Retrieval {
  stage: string;
  note: string;
  vector_search_used: boolean;
  keyword_search_used: boolean;
  rerank_used: boolean;
  query_rewrite_used: boolean;
  evidence: EvidenceItem[];
}

export interface Calculation {
  metric: string;
  value: number;
  currency: string;
  period: string;
  formula: string;
  inputs: Record<string, unknown>;
  source_ids: string[];
}

export interface ChatApiResponse {
  response: string;
  is_casual: boolean;
  intent: string;
  risk_level: string;
  critic_passed: boolean;
  critic_retries_used: number;
  critic_errors: string[];
  critic_unsupported_claims: string[];
  retrieval: Retrieval | null;
  calculations: Calculation[];
  skipped_calculations: string[];
  specialists_used: string[];
  specialist_outputs: Record<string, unknown>;
  latency_ms: number;
}

export interface ProfileResponse {
  user_id: string;
  has_profile: boolean;
  profile: Record<string, unknown> | null;
}

export interface Transaction {
  date: string;
  description: string;
  amount: number;
  currency: string;
  category: string;
  account: string;
  type: string;
  source_id: string;
}

export interface TransactionsResponse {
  user_id: string;
  count: number;
  transactions: Transaction[];
}

export interface DeleteUserResponse {
  user_id: string;
  deleted: boolean;
}

export interface SeedDemoResponse {
  user_id: string;
  transactions_seeded: number;
  vector_indexed: number;
}

export interface HealthResponse {
  status: string;
  llm_configured: boolean;
  provider: string | null;
  model: string | null;
  llm_error: string | null;
  warm_up: Record<string, boolean>;
}

/** Local chat-thread state -- not a wire type. `meta` is only present on
 * a completed assistant reply (holds the full ChatApiResponse so
 * VerifiedStrip/EvidenceDrawer can render it); `pending`/`error` cover
 * the in-flight and failed states for the message the UI is currently
 * waiting on. */
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: number;
  meta?: ChatApiResponse;
  pending?: boolean;
  error?: string;
}
