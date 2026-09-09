# Deploying the Fault Detection Demo to a Live Server

## Architecture: one service, not two

`backend/main.py` does double duty — it serves the `/api/...` JSON
endpoints **and** mounts `frontend/` as static files at `/`
(`app.mount("/", StaticFiles(directory="frontend", html=True), ...)`).
So there is only **one** thing to deploy: the FastAPI app. Whatever URL it
ends up at (`https://fault-detection-demo.onrender.com`, your own domain,
etc.) serves both the page and the API from that same origin.

This is why `frontend/app.js` sets:
```js
const API_BASE = "";
```
Empty string means every `fetch()` call is a **relative** URL
(`/api/fault-types`, not `http://localhost:8000/api/fault-types`), resolved
against whatever origin actually served the page. `localhost` only means
"this machine" — hardcoding it would only ever work on the one computer
running `uvicorn` locally, never for anyone else opening the deployed link.
Don't hardcode a URL here for any deployment target; leave it as `""`.

> **Before deploying anywhere:** train the model locally first. The live
> server does **not** train anything — it only loads files you already
> generated:
> ```bash
> python ml/train_model.py
> ```
> This must have already produced `ml/model.joblib` and
> `ml/model_comparison.json`, and `data/raw/*.csv` must contain your 5 real
> CSVs. All of these need to be **committed to git** (don't `.gitignore`
> them) — the server just reads them, it can't create them.

---

## Option A — Render (recommended, and what's currently deployed)

Render runs your FastAPI app as a normal always-on Python process — no
serverless size limits to fight with `scikit-learn`/`pandas`/`scipy`, and
no split between frontend/backend to keep in sync.

### 1. Push the project to GitHub

```bash
cd fault-detection-demo
git init                      # skip if already a repo
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/<your-username>/fault-detection-demo.git
git push -u origin main
```
Add a `.gitignore` for `venv/`, `__pycache__/`, `*.pyc` — but **do not**
ignore `data/raw/`, `ml/model.joblib`, or `ml/model_comparison.json`.

### 2. Create the Web Service on Render

In the Render dashboard → **New** → **Web Service** → connect the repo:
- **Root Directory:** leave blank (repo root)
- **Runtime:** Python 3
- **Build Command:** `pip install -r backend/requirements.txt`
- **Start Command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- **Instance Type:** Free is fine for a demo

Render assigns a `$PORT` env var at runtime — the start command above must
bind to it (not a hardcoded `8000`), which is why it's `--port $PORT` and
not `--port 8000`.

### 3. Verify

Once deployed, visit `https://<your-app>.onrender.com/api/fault-types` —
you should get the JSON list of fault types back. Then load
`https://<your-app>.onrender.com/` itself and click **Run Simulation**.
No local `uvicorn` process needs to be running for this to work for anyone
— that was the bug: with `API_BASE` hardcoded to `localhost:8000`, it only
ever worked on your machine, with your local server up, because your
browser's `localhost` resolved to your own machine either way.

> **Free tier note:** Render's free web services spin down after ~15
> minutes of inactivity and take 30-60s to wake back up on the next
> request. That's normal — the first "Run Simulation" after a period of
> no traffic will just be slow, not broken.

---

## Option B — Vercel

Vercel can run this too, as a single Python serverless project (same
single-origin setup as Render — `backend/main.py` already serves the
frontend, so there's only one project to deploy here as well).

Add two small files to the project root:

**`api/index.py`**
```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))

from backend.main import app  # noqa: E402  (Vercel looks for `app`)
```

**`vercel.json`** (project root)
```json
{
  "builds": [{ "src": "api/index.py", "use": "@vercel/python" }],
  "routes": [{ "src": "/(.*)", "dest": "api/index.py" }]
}
```

`requirements.txt` must also exist at the project root (copy
`backend/requirements.txt`'s contents there — Vercel's Python builder looks
for it at the root specifically).

```bash
npm install -g vercel      # one-time, needs Node.js installed
vercel login
vercel --prod
```

> **Known limitation:** Vercel's serverless functions have a 250MB
> uncompressed size limit. `scikit-learn` + `pandas` + `numpy` + `scipy`
> together get close to it. If the deploy fails on bundle size, that's why
> — Render (Option A) doesn't have this problem, since it's a normal
> process rather than a packaged function.

---

## Option C — cPanel (shared hosting / your own domain)

Most modern cPanel hosts include **"Setup Python App"** (via CloudLinux's
Python Selector), running your FastAPI app under Passenger.

### 1. Create the Python app

cPanel → **Setup Python App** → **Create Application**:
- Python version: highest 3.10+ available
- Application root: e.g. `fault-detection-demo`
- Application URL: your domain/subdomain
- Application startup file: `passenger_wsgi.py`
- Application Entry point: `application`

### 2. Upload the project

Via cPanel's **Git™ Version Control** (point it at your GitHub repo), or
File Manager / FTP/SFTP.

### 3. Install dependencies

```bash
source /home/<user>/virtualenv/fault-detection-demo/3.10/bin/activate
cd /home/<user>/fault-detection-demo
pip install -r backend/requirements.txt
pip install a2wsgi   # bridges FastAPI's ASGI interface to Passenger's WSGI
```
Passenger expects a **WSGI** app; FastAPI is **ASGI**. `a2wsgi` bridges
that — no changes needed to `backend/main.py`.

### 4. Add `passenger_wsgi.py`

At the application root (next to `backend/`, `ml/`, `frontend/`):
```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))

from a2wsgi import ASGIMiddleware
from backend.main import app as fastapi_app

application = ASGIMiddleware(fastapi_app)
```
Since `backend/main.py` already mounts `frontend/` itself, this one app
serves everything — no separate static hosting step needed here either.

### 5. Restart

**Setup Python App** → **Restart**, any time you upload new files or
change code.

### 6. Verify

Visit `https://yourdomain.com/api/fault-types` for the JSON list, then load
`https://yourdomain.com/` and click **Run Simulation**.

---

## Retraining after the app is already live

If you swap in new/updated CSVs, or change `ml/train_model.py` or
`ml/feature_extraction.py`, you need to retrain and get the new
`ml/model.joblib` in front of the live app. **How depends on the host:**

### Render / Vercel — retrain locally, then redeploy

Both are stateless at request time (Vercel always; Render effectively so,
since a fresh deploy replaces the running instance). There's no "SSH in and
retrain" step:

```bash
python ml/train_model.py          # regenerates ml/model.joblib locally
git add ml/model.joblib ml/model_comparison.json data/raw/*.csv
git commit -m "Retrain model on updated data"
git push                          # auto-redeploys if the repo is linked
```

### cPanel — you have a real server, so you actually can run it there

```bash
source /home/<user>/virtualenv/fault-detection-demo/3.10/bin/activate
cd /home/<user>/fault-detection-demo
python ml/train_model.py
```
This overwrites `ml/model.joblib` in place. Passenger loads the model once
into memory at startup and won't notice the file changing underneath it —
**Restart** the app from cPanel's Setup Python App page afterward, the same
reason a local `uvicorn --reload` process needs a restart to pick up a
freshly-trained model (see "Clearing the cache" in `HOW_IT_WORKS.md`).

## Quick checklist before deploying

- [ ] `python ml/train_model.py` has been run locally and succeeded
- [ ] `ml/model.joblib` and `ml/model_comparison.json` exist and are committed
- [ ] `data/raw/*.csv` (all 5 files) exist and are committed
- [ ] `frontend/assets/simulink_model.png` exists and is committed
- [ ] `frontend/app.js`'s `API_BASE` is `""` (relative) — never a hardcoded
      `localhost` or a specific deployed domain
