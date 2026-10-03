# Frontend

Minimal React + TypeScript (Vite) UI for the Model Provenance Verifier API: a form to
submit a comparison, a table of past runs, and a detail view with the evidence breakdown.

## Setup

```bash
npm install
npm run dev
```

By default this talks to the API at `http://127.0.0.1:8000`. To point at a different
address, set `VITE_API_BASE_URL` (e.g. in a `.env.local` file):

```
VITE_API_BASE_URL=http://127.0.0.1:8000
```

The API must be running separately (see the root README) for the form to do anything.
