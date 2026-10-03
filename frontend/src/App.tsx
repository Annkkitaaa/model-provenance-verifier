import { useEffect, useRef, useState } from "react";
import "./App.css";
import { createRun, getRun, listRuns, type Run } from "./api";

const METHOD_LABELS: Record<string, string> = {
  teacher_forced_full_vocab: "cosine similarity (shared vocabulary)",
  topk_decoded_token_overlap: "top-k token overlap (different vocabularies)",
};

function formatScore(value: number | null): string {
  return value === null ? "-" : value.toFixed(3);
}

function formatRevision(value: string | null): string {
  return value === null ? "unresolved" : value.slice(0, 10);
}

export default function App() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [modelA, setModelA] = useState("gpt2");
  const [modelB, setModelB] = useState("distilgpt2");
  const [submitting, setSubmitting] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const pollTimers = useRef<Map<number, ReturnType<typeof setInterval>>>(new Map());

  useEffect(() => {
    listRuns()
      .then(setRuns)
      .catch((err: Error) => setFormError(err.message));
  }, []);

  useEffect(() => {
    return () => {
      for (const timer of pollTimers.current.values()) clearInterval(timer);
    };
  }, []);

  function pollUntilDone(runId: number) {
    const timer = setInterval(() => {
      getRun(runId)
        .then((run) => {
          setRuns((prev) => prev.map((r) => (r.id === run.id ? run : r)));
          if (run.status === "completed" || run.status === "failed") {
            clearInterval(timer);
            pollTimers.current.delete(runId);
          }
        })
        .catch(() => {
          clearInterval(timer);
          pollTimers.current.delete(runId);
        });
    }, 2000);
    pollTimers.current.set(runId, timer);
  }

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    setFormError(null);
    setSubmitting(true);
    try {
      const run = await createRun(modelA.trim(), modelB.trim());
      setRuns((prev) => [run, ...prev]);
      setSelectedRunId(run.id);
      if (run.status === "pending" || run.status === "running") {
        pollUntilDone(run.id);
      }
    } catch (err) {
      setFormError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }

  const selectedRun = runs.find((r) => r.id === selectedRunId) ?? null;

  return (
    <div className="page">
      <h1>Model Provenance Verifier</h1>
      <p className="subtitle">
        Submit two Hugging Face model ids to compare their output distributions on a fixed
        probe set.
      </p>

      <form className="run-form" onSubmit={handleSubmit}>
        <label>
          Base model
          <input value={modelA} onChange={(e) => setModelA(e.target.value)} required />
        </label>
        <label>
          Candidate model
          <input value={modelB} onChange={(e) => setModelB(e.target.value)} required />
        </label>
        <button type="submit" disabled={submitting}>
          {submitting ? "Submitting..." : "Compare"}
        </button>
      </form>
      {formError && <p className="error">{formError}</p>}

      <h2>Runs</h2>
      <table className="runs-table">
        <thead>
          <tr>
            <th>id</th>
            <th>base model</th>
            <th>candidate model</th>
            <th>status</th>
            <th>score</th>
            <th>created</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => (
            <tr
              key={run.id}
              className={run.id === selectedRunId ? "selected" : ""}
              onClick={() => setSelectedRunId(run.id)}
            >
              <td>{run.id}</td>
              <td>{run.model_a}</td>
              <td>{run.model_b}</td>
              <td className={`status status-${run.status}`}>{run.status}</td>
              <td>{formatScore(run.score)}</td>
              <td>{new Date(run.created_at).toLocaleString()}</td>
            </tr>
          ))}
          {runs.length === 0 && (
            <tr>
              <td colSpan={6}>No runs yet.</td>
            </tr>
          )}
        </tbody>
      </table>

      {selectedRun && (
        <div className="detail">
          <h2>Run #{selectedRun.id} evidence</h2>
          <dl>
            <dt>Status</dt>
            <dd className={`status status-${selectedRun.status}`}>{selectedRun.status}</dd>

            <dt>Score</dt>
            <dd>{formatScore(selectedRun.score)}</dd>

            <dt>Method</dt>
            <dd>{selectedRun.method ? METHOD_LABELS[selectedRun.method] ?? selectedRun.method : "-"}</dd>

            <dt>Shared vocabulary</dt>
            <dd>
              {selectedRun.vocab_compatible === null ? "-" : selectedRun.vocab_compatible ? "yes" : "no"}
            </dd>

            <dt>Mean cosine similarity</dt>
            <dd>{formatScore(selectedRun.mean_cosine_similarity)}</dd>

            <dt>Mean KL divergence</dt>
            <dd>{formatScore(selectedRun.mean_kl_divergence)}</dd>

            <dt>Mean top-k Jaccard overlap</dt>
            <dd>{formatScore(selectedRun.mean_topk_jaccard)}</dd>

            <dt>Probe set</dt>
            <dd>{selectedRun.probe_set_id ?? "-"}</dd>

            <dt>Base model revision</dt>
            <dd>{formatRevision(selectedRun.model_a_revision)}</dd>

            <dt>Candidate model revision</dt>
            <dd>{formatRevision(selectedRun.model_b_revision)}</dd>

            {selectedRun.error && (
              <>
                <dt>Error</dt>
                <dd className="error">{selectedRun.error}</dd>
              </>
            )}
          </dl>
        </div>
      )}
    </div>
  );
}
