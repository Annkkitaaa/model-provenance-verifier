export interface Run {
  id: number;
  model_a: string;
  model_b: string;
  model_a_revision: string | null;
  model_b_revision: string | null;
  probe_set_id: string | null;
  status: "pending" | "running" | "completed" | "failed";
  score: number | null;
  vocab_compatible: boolean | null;
  method: string | null;
  mean_cosine_similarity: number | null;
  mean_kl_divergence: number | null;
  mean_topk_jaccard: number | null;
  error: string | null;
  created_at: string;
  completed_at: string | null;
}

const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://127.0.0.1:8000";

async function parseOrThrow(response: Response): Promise<Run> {
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status} ${response.statusText}`);
  }
  return (await response.json()) as Run;
}

export async function listRuns(): Promise<Run[]> {
  const response = await fetch(`${API_BASE_URL}/runs`);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status} ${response.statusText}`);
  }
  return (await response.json()) as Run[];
}

export async function getRun(id: number): Promise<Run> {
  const response = await fetch(`${API_BASE_URL}/runs/${id}`);
  return parseOrThrow(response);
}

export async function createRun(modelA: string, modelB: string): Promise<Run> {
  const response = await fetch(`${API_BASE_URL}/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_a: modelA, model_b: modelB }),
  });
  return parseOrThrow(response);
}
