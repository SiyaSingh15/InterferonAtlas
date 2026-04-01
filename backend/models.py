from pydantic import BaseModel
from typing import Optional

class UploadResponse(BaseModel):
    session_id: str
    n_cells: int
    n_genes: int
    message: str

class ScoreRequest(BaseModel):
    session_id: str
    use_raw: bool = False

class ScoreResponse(BaseModel):
    session_id: str
    n_cells: int
    ifn_ss_mean: float
    ifn_ss_min: float
    ifn_ss_max: float
    bifurcation_mean: float
    module_coverage: dict
    message: str

class PlotRequest(BaseModel):
    session_id: str
    color: Optional[str] = "IFN_SS"

class HealthResponse(BaseModel):
    status: str
    version: str