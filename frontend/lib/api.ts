/** API service layer for the demo hybrid recommender portal. */

import type {
  SampleUser,
  UserProfile,
  ItemOut,
  ItemDetail,
  SearchResponse,
  SimilarResponse,
  RecommendationResponse,
  RetrainResponse,
  RatingRequest,
  RatingResponse,
  GraphResponse,
  ExplanationResponse,
} from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...options?.headers },
    ...options,
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail || `API error: ${res.status}`);
  }
  return res.json();
}

// ── Users ───────────────────────────────────────────────────────────────

export async function getUsers(minRatings?: number, maxRatings?: number): Promise<SampleUser[]> {
  const params = new URLSearchParams();
  if (minRatings != null) params.set("min_ratings", String(minRatings));
  if (maxRatings != null) params.set("max_ratings", String(maxRatings));
  const qs = params.toString();
  return fetchJSON<SampleUser[]>(`${API_BASE}/api/users${qs ? `?${qs}` : ""}`);
}

export async function getUser(userId: number): Promise<UserProfile> {
  return fetchJSON<UserProfile>(`${API_BASE}/api/users/${userId}`);
}

export async function createUser(name: string): Promise<SampleUser> {
  return fetchJSON<SampleUser>(`${API_BASE}/api/users`, {
    method: "POST",
    body: JSON.stringify({ name }),
  });
}

// ── Recommendations ─────────────────────────────────────────────────────

export async function getRecommendations(userId: number): Promise<RecommendationResponse> {
  return fetchJSON<RecommendationResponse>(`${API_BASE}/api/recommendations/${userId}`);
}

export async function retrainModels(): Promise<RetrainResponse> {
  return fetchJSON<RetrainResponse>(`${API_BASE}/api/retrain`, { method: "POST" });
}

// ── Items ───────────────────────────────────────────────────────────────

export async function searchItems(query: string, limit: number = 20, userId?: number): Promise<SearchResponse> {
  const params = new URLSearchParams({ q: query, limit: String(limit) });
  if (userId) params.set("user_id", String(userId));
  return fetchJSON<SearchResponse>(`${API_BASE}/api/items/search?${params}`);
}

export async function getItem(itemIdx: number): Promise<ItemDetail> {
  return fetchJSON<ItemDetail>(`${API_BASE}/api/items/${itemIdx}`);
}

export async function getSimilarItems(itemIdx: number, limit: number = 10): Promise<SimilarResponse> {
  return fetchJSON<SimilarResponse>(`${API_BASE}/api/items/${itemIdx}/similar?limit=${limit}`);
}

// ── Ratings ─────────────────────────────────────────────────────────────

export async function submitRating(data: RatingRequest): Promise<RatingResponse> {
  return fetchJSON<RatingResponse>(`${API_BASE}/api/ratings`, {
    method: "POST",
    body: JSON.stringify(data),
  });
}

// ── Graph & Explanation ─────────────────────────────────────────────────

export async function getItemGraph(itemIdx: number, userId?: number): Promise<GraphResponse> {
  const params = new URLSearchParams();
  if (userId) params.set("user_id", String(userId));
  return fetchJSON<GraphResponse>(`${API_BASE}/api/items/${itemIdx}/graph?${params}`);
}

export async function getItemExplanation(itemIdx: number, userId?: number): Promise<ExplanationResponse> {
  const params = new URLSearchParams();
  if (userId) params.set("user_id", String(userId));
  return fetchJSON<ExplanationResponse>(`${API_BASE}/api/items/${itemIdx}/explanation?${params}`);
}
