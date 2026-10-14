# DialReader frontend

A React + Vite page for the DialReader API: upload a gauge photo, enter the scale,
and see the reading, its error bar, a safe-range alert and the main pipeline stages.

## Run

Start the API from the repository root first:

```
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```

Then, in this folder:

```
npm install
npm run dev          # http://localhost:5173
```

The page calls `http://localhost:8000` by default. To point it elsewhere, set
`VITE_API_URL` (for example in `frontend/.env.local`). If the page is served from a
different origin than `localhost:5173`, add that origin to `DIALREADER_CORS_ORIGINS`
(comma-separated) before starting the API.
