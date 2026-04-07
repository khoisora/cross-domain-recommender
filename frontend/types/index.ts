/** Core TypeScript interfaces matching demo backend schemas. */

export type Domain = "movie" | "game";

/* ── Users ─────────────────────────────────────────────────── */

export interface SampleUser {
  id: number;
  external_id: string;
  name: string;
  avatar: string;
  taste_summary: string;
  total_ratings: number;
  avg_rating: number;
  is_sample?: boolean;
  group?: string;
}

export interface UserProfile extends SampleUser {
  ratings: UserRating[];
}

export interface UserRating {
  item_id: string;
  item_idx: number | null;
  title: string;
  domain: Domain;
  rating: number;
  image_url: string;
  genres: string;
}

/* ── Items ─────────────────────────────────────────────────── */

export interface ItemOut {
  idx: number;
  external_id: string;
  title: string;
  domain: Domain;
  genres: string;
  tags: string;
  image_url: string;
  description: string;
  avg_rating: number | null;
  rating_count: number;
  year: string;
  score: number;
  reason: string;
}

export interface ItemDetail {
  idx: number;
  external_id: string;
  title: string;
  domain: Domain;
  genres: string;
  tags: string;
  image_url: string;
  description: string;
  avg_rating: number | null;
  rating_count: number;
  year: string;
  user_rating?: number;
  similar_games?: SimilarItem[];
  similar_movies?: SimilarItem[];
}

export interface SimilarItem {
  external_id: string;
  title: string;
  domain: Domain;
  image_url: string;
  avg_rating: number | null;
  similarity: number;
}

export interface SearchResponse {
  items: ItemOut[];
  total: number;
}

export interface SimilarResponse {
  items: ItemOut[];
  cross_domain_items: ItemOut[];
  total: number;
}

/* ── Recommendations ───────────────────────────────────────── */

export interface RecommendationRow {
  key: string;
  title: string;
  subtitle: string;
  model_tag: string;
  items: ItemOut[];
}

export interface RecommendationResponse {
  user_id: number;
  user_name: string;
  rows: RecommendationRow[];
}

/* ── Retrain ───────────────────────────────────────────────── */

export interface RetrainResponse {
  status: string;
  models: string[];
  elapsed_s: number;
}

/* ── Ratings ───────────────────────────────────────────────── */

export interface RatingRequest {
  user_id: number;
  external_id: string;
  rating: number;
}

export interface RatingResponse {
  success: boolean;
  message: string;
}

/* ── Graph ─────────────────────────────────────────────────── */

export interface GraphNode {
  id: number;
  label: string;
  domain: Domain;
  type: "center" | "related" | "rated";
  genres: string;
  tags: string;
  image_url: string;
  avg_rating: number | null;
  user_rating?: number;
  rating_timestamp?: string;
}

export interface GraphEdge {
  source: number;
  target: number;
  weight: number;
  label: string;
}

export interface GraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  explanation_text: string;
}

/* ── Explanation ───────────────────────────────────────────── */

export interface ExplanationReason {
  type: string;
  text: string;
  weight: number;
  source_item_idx?: number;
}

export interface ExplanationResponse {
  reasons: ExplanationReason[];
  summary: string;
}
