# Deploying the Fault Detection Demo to a Live Server

This app has two parts that get deployed separately:

- **Frontend** — plain HTML/CSS/JS (`frontend/`). Trivial to host anywhere.
- **Backend** — a Python FastAPI app (`backend/` + `ml/`) that needs to run
  a Python process, not just serve static files.

The two options below cover **Vercel** (recommended — free, fast, easiest)
and **cPanel** (if you already have shared hosting/a domain there).

> **Before either option:** train the model locally first. The live server
> does **not** train anything — it only loads the files you already
> generated:
> ```bash
> python ml/train_model.py
> ```
> This must have already produced `ml/model.joblib` and
> `ml/model_comparison.json`, and `data/raw/*.csv` must contain your 5 real
> CSVs. All of these files need to be **committed to your repo** (do not
> gitignore them) — the server just reads them, it can't create them.

---

## Option A — Vercel (recommended)

Vercel works best here as **two separate projects**: one for the static
frontend, one for the Python API. This is simpler and more reliable than
trying to serve both from a single Vercel project.

### 1. Put the project on GitHub

```bash
cd fault-detection-demo
git init
git add .
git commit -m "Initial commit"
```
Add a `.gitignore` first if you don't have one:
```
venv/
__pycache__/
*.pyc
```
Do **not** ignore `data/raw/`, `ml/model.joblib`, or
`ml/model_comparison.json` — Vercel needs them at deploy time.

Create a new GitHub repo and push:
```bash
git remote add origin https://github.com/<your-username>/fault-detection-demo.git
git push -u origin main
```

### 2. Deploy the backend (FastAPI) as a Vercel project

Vercel runs Python code as serverless functions from an `api/` folder.
Add two small files to the project root (these don't change any of your
existing app logic — they just tell Vercel how to expose it):

**`api/index.py`**
```python
import sys
from pathlib import Path

# Make backend/ and ml/ importable from this serverless function
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

**`requirements.txt`** must exist at the project root too (Vercel's Python
builder looks for it there) — copy `backend/requirements.txt` to the root,
or add a root `requirements.txt` with the same contents.

Then:
```bash
npm install -g vercel      # one-time, needs Node.js installed
vercel login
vercel                     # deploy a preview
vercel --prod              # deploy to production
```
Vercel will give you a URL like `https://fault-detection-api.vercel.app`.
Test it: `https://fault-detection-api.vercel.app/api/fault-types` should
return the JSON list of fault types.

> **Known limitation:** Vercel's free-tier serverless functions have a
> 250MB uncompressed size limit. `scikit-learn` + `pandas` + `numpy` +
> `scipy` together are large and can get close to this. If your deploy
> fails with a size/bundle error, this is why. In that case, a small
> always-on host like **Render.com** or **Railway.app** (free tier,
> deploys a FastAPI app directly with `uvicorn`, no bundle-size games) is a
> much easier fallback for the backend specifically — same code, no
> changes needed, just point the frontend's `API_BASE` at whichever URL
> you end up with.

### 3. Deploy the frontend as a second Vercel project

```bash
vercel --cwd frontend
vercel --cwd frontend --prod
```
(Or via the Vercel dashboard: "Add New Project" → import the same GitHub
repo → set **Root Directory** to `frontend` → deploy. No build step
needed, it's plain static files.)

### 4. Point the frontend at the deployed backend

Edit `frontend/app.js`:
```js
const API_BASE = "https://fault-detection-api.vercel.app"; // your backend URL
```
Commit and redeploy the frontend project (`vercel --cwd frontend --prod`).

CORS is already wide open (`allow_origins=["*"]`) in `backend/main.py`, so
the frontend calling a different Vercel domain will work without changes.

---

## Option B — cPanel (shared hosting / your own domain)

Most modern cPanel hosts include **"Setup Python App"** (via CloudLinux's
Python Selector). This runs your FastAPI app under Passenger.

### 1. Create the Python app

In cPanel → **Setup Python App** → **Create Application**:
- Python version: highest 3.10+ available
- Application root: e.g. `fault-detection-demo`
- Application URL: pick a domain/subdomain, e.g. `api.yourdomain.com`
- Application startup file: `passenger_wsgi.py`
- Application Entry point: `application`

cPanel creates a virtualenv for you and shows an activation command like:
```bash
source /home/<user>/virtualenv/fault-detection-demo/3.10/bin/activate
```

### 2. Upload the project

Easiest via cPanel's **Git™ Version Control** (point it at your GitHub
repo and clone/pull into the application root), otherwise upload via File
Manager or FTP/SFTP.

### 3. Install dependencies

SSH in (or use the "Run Pip Install" box on the Setup Python App page):
```bash
source /home/<user>/virtualenv/fault-detection-demo/3.10/bin/activate
cd /home/<user>/fault-detection-demo
pip install -r backend/requirements.txt
pip install a2wsgi   # bridges FastAPI's ASGI interface to Passenger's WSGI
```
Passenger (cPanel's process manager) expects a **WSGI** app, but FastAPI is
**ASGI** — `a2wsgi` handles that translation, no changes to `backend/main.py`
needed.

### 4. Add `passenger_wsgi.py`

Create this file at the application root (next to `backend/`, `ml/`, etc.):
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

### 5. Serve the frontend

Two ways — pick whichever is simpler for your hosting setup:

- **Separate static hosting (simplest):** upload `frontend/` to your
  normal `public_html` (or a subdomain's document root), and edit
  `frontend/app.js`'s `API_BASE` to point at the Python app's URL
  (e.g. `https://api.yourdomain.com`). Mirrors the Vercel two-project
  approach above.
- **Single domain, one app serves both:** add this to the bottom of
  `backend/main.py`, **after** all the `@app.get(...)` routes are defined:
  ```python
  from fastapi.staticfiles import StaticFiles
  app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
  ```
  Then set `API_BASE = ""` (empty string, relative) in `frontend/app.js`,
  since the frontend and API are now on the same origin.

### 6. Restart

Go back to **Setup Python App** in cPanel and click **Restart** any time
you upload new files or change code.

### 7. Verify

Visit `https://api.yourdomain.com/api/fault-types` — you should get the
JSON list of fault types back. Then load the frontend URL and click **Run
Simulation**.

---

## Retraining after the app is already live

If you swap in new/updated CSVs, or change something in `ml/train_model.py`
or `ml/feature_extraction.py`, you need to retrain and get the new
`ml/model.joblib` in front of the live app. **How you do that depends on
which hosting option you used**, because Vercel and cPanel behave very
differently here.

### On Vercel — you can't run it "on" the server, and that's normal

Vercel functions are **serverless and stateless**: there's no persistent
machine to SSH into, and the filesystem a function runs against is
read-only at request time. So the workflow is always: **retrain locally,
then redeploy** —

```bash
python ml/train_model.py          # regenerates ml/model.joblib locally
git add ml/model.joblib ml/model_comparison.json data/raw/*.csv
git commit -m "Retrain model on updated data"
git push                          # auto-redeploys if the repo is linked to Vercel
# or, without git integration:
vercel --cwd . --prod             # (from the backend project's directory)
```
Every deploy is a fresh function build that loads whatever `model.joblib`
is in the commit/upload — there's no separate "restart" step to remember,
a new deploy always starts clean. (If you want this automated so you never
retrain by hand, a GitHub Action that runs `train_model.py` and commits the
result before Vercel's build step is the standard way — ask if you want
that set up.)

### On cPanel — you have a real server, so you actually can run it there

Unlike Vercel, the cPanel Python App gives you a persistent virtualenv you
can SSH/Terminal into directly:

```bash
source /home/<user>/virtualenv/fault-detection-demo/3.10/bin/activate
cd /home/<user>/fault-detection-demo
python ml/train_model.py
```
This overwrites `ml/model.joblib` in place on the server. Since Passenger
(the process manager running your app) loads the model once into memory at
startup and doesn't notice the file changing underneath it, you then need
to **restart the app** from cPanel's **Setup Python App** page (the
**Restart** button) for it to pick up the freshly-trained model — the exact
same reason a local `uvicorn --reload` process doesn't pick up a new model
file without restarting either (see "Clearing the cache" in
`HOW_IT_WORKS.md` if that's confusing).

## Quick checklist before either deployment

- [ ] `python ml/train_model.py` has been run locally and succeeded
- [ ] `ml/model.joblib` and `ml/model_comparison.json` exist and are committed
- [ ] `data/raw/*.csv` (all 5 files) exist and are committed
- [ ] `frontend/assets/simulink_model.png` exists and is committed
- [ ] `frontend/app.js`'s `API_BASE` points at wherever the backend actually ends up living
