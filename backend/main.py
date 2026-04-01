import uuid
import io
import tempfile
from pathlib import Path

import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from models import (UploadResponse, ScoreRequest, ScoreResponse,
                    HealthResponse)
from scorer import preprocess, score_ifn, get_score_summary

app = FastAPI(
    title="InterferonAtlas API",
    description="Continuous IFN signaling state classification for scRNA-seq and bulk RNA-seq",
    version="0.1.0"
)

# In-memory session store — replaced with Redis or disk cache in production
SESSIONS: dict[str, sc.AnnData] = {}
SESSION_DIR = Path(tempfile.gettempdir()) / "interferonAtlas_sessions"
SESSION_DIR.mkdir(exist_ok=True)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", version="0.1.0")


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    """Accept an .h5ad file and return a session_id."""
    if not file.filename.endswith(".h5ad"):
        raise HTTPException(status_code=400,
                            detail="Only .h5ad files are supported.")

    session_id = str(uuid.uuid4())[:8]
    tmp_path = SESSION_DIR / f"{session_id}.h5ad"

    contents = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(contents)

    try:
        adata = sc.read_h5ad(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=422,
                            detail=f"Could not read h5ad file: {e}")

    SESSIONS[session_id] = adata
    return UploadResponse(
        session_id=session_id,
        n_cells=adata.shape[0],
        n_genes=adata.shape[1],
        message=f"File uploaded. {adata.shape[0]} cells x {adata.shape[1]} genes."
    )


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    """Run IFN-SS scoring pipeline on uploaded dataset."""
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")

    adata = SESSIONS[req.session_id]

    try:
        adata = preprocess(adata)
        adata = score_ifn(adata)
        SESSIONS[req.session_id] = adata
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Scoring failed: {e}")

    summary = get_score_summary(adata)
    return ScoreResponse(
        session_id=req.session_id,
        message="Scoring complete.",
        **summary
    )


@app.get("/plot/umap")
def plot_umap(session_id: str, color: str = "IFN_SS"):
    """Return a UMAP PNG colored by IFN_SS or any obs column."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")

    adata = SESSIONS[session_id]
    if "X_umap" not in adata.obsm:
        raise HTTPException(status_code=400,
                            detail="UMAP not computed yet. Call /score first.")
    if color not in adata.obs.columns:
        raise HTTPException(status_code=400,
                            detail=f"Column '{color}' not in obs.")

    fig, ax = plt.subplots(figsize=(6, 5))
    sc.pl.umap(adata, color=color, color_map="RdBu_r",
               ax=ax, show=False, title=f"IFN-SS — {color}")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/plot/distributions")
def plot_distributions(session_id: str):
    """Return IFN-SS and bifurcation entropy histograms as PNG."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")

    adata = SESSIONS[session_id]
    if "IFN_SS" not in adata.obs.columns:
        raise HTTPException(status_code=400,
                            detail="Scores not computed yet. Call /score first.")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(adata.obs["IFN_SS"], bins=50,
                 color="#4e79a7", edgecolor="white")
    axes[0].set_xlabel("IFN Spectrum Score")
    axes[0].set_ylabel("Cells")
    axes[0].set_title("IFN-SS distribution")

    axes[1].hist(adata.obs["IFN_bifurcation"], bins=50,
                 color="#f28e2b", edgecolor="white")
    axes[1].set_xlabel("Bifurcation entropy")
    axes[1].set_ylabel("Cells")
    axes[1].set_title("Bifurcation entropy distribution")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/export/csv")
def export_csv(session_id: str):
    """Export per-cell scores as CSV."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")

    adata = SESSIONS[session_id]
    if "IFN_SS" not in adata.obs.columns:
        raise HTTPException(status_code=400,
                            detail="Scores not computed yet. Call /score first.")

    cols = ["IFN_SS", "IFN_bifurcation"] + \
           [c for c in adata.obs.columns if c.startswith("score_")]
    df = adata.obs[cols]
    buf = io.StringIO()
    df.to_csv(buf)
    buf.seek(0)
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition":
                 f"attachment; filename=ifn_scores_{session_id}.csv"}
    )