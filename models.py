# models.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# =============================================================================
# 1. Request Models (Data coming INTO the API)
# =============================================================================

class GenerateRequest(BaseModel):
    """
    Defines the input for the primary paper generation endpoint (/generate_full).
    """
    board: str
    class_label: str
    subject: str
    chapters: Optional[List[str]] = None

    class Config:
        # Provides an example for the API documentation
        schema_extra = {
            "example": {
                "board": "CBSE",
                "class_label": "Class 10",
                "subject": "Science",
                "chapters": ["Chemical Reactions and Equations", "Acids, Bases and Salts"]
            }
        }


class InputData(BaseModel):
    """
    Defines the input for the legacy endpoints (/process, /generate).
    This model is quite broad, covering multiple use cases.
    """
    id: str
    Board: str
    Class: str
    Subject: str
    Chapter: str
    Prompt_Type: str
    hit_count: int
    is_logedIn: bool
    answer: bool = False
    question_paper: Optional[str] = None


class ResearchInput(BaseModel):
    """
    Defines the input for the research agent endpoint (/research).
    """
    query: str

    class Config:
        schema_extra = {
            "example": {
                "query": "What were the key advancements in battery technology in 2023?"
            }
        }


# =============================================================================
# 2. Response Models (Data going OUT of the API)
# =============================================================================

class ProcessResponse(BaseModel):
    """
    Defines the response structure for the legacy endpoints.
    """
    id: str
    result: str
    hit_count: int


class ResearchResponse(BaseModel):
    """
    Defines the response structure for the research endpoint.
    """
    response: str


# --- Nested Models for the Detailed Paper Response ---

class QuestionModel(BaseModel):
    """

    Represents a single question within the generated paper.
    'q_text' is highly flexible to accommodate simple text, choices, or case studies.
    """
    q_id: str
    q_text: Any  # Can be str, list[dict], or dict for case studies
    marks: float
    type: str
    is_choice: Optional[bool] = False
    _cleaning_note: Optional[str] = None


class SummaryModel(BaseModel):
    """Represents a summary of evidence retrieved for a section."""
    id: str
    summary: str


class RetrievalMetadataModel(BaseModel):
    """Represents the retrieved and summarized evidence for one section of the paper."""
    section_id: str
    summaries: List[SummaryModel]
    slot_meta: str


# models.py

from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any

# ... (other models are fine) ...

class PaperModel(BaseModel):
    """
    Represents the core generated paper object. This is the 'value' inside the final response.
    """
    paper_id: str
    board: str
    class_label: str
    subject: str
    total_marks: Optional[int]
    time_allowed_minutes: Optional[int]
    general_instructions: Optional[List[str]]
    questions: List[QuestionModel]
    retrieval_metadata: List[RetrievalMetadataModel]

    # --- ADD THIS VALIDATOR ---
    @validator("general_instructions", pre=True)
    def ensure_instructions_is_list(cls, v):
        """If general_instructions is a single string, wrap it in a list."""
        if isinstance(v, str):
            # Split by newline and filter out empty lines for cleaner output
            return [line.strip() for line in v.split('\n') if line.strip()]
        return v

class PaperResponse(BaseModel):
    """
    The final, top-level response model for the /generate_full endpoint.
    This "envelope" structure supports the versioned caching strategy.
    """
    version_id: str
    created_at: str
    value: PaperModel





# --- NEW DOCUMENT PROCESSING MODELS ---
class DocumentSummary(BaseModel):
    """Represents the summary for a single document."""
    filename: str
    summary: str

class TaskStatusResponse(BaseModel):
    """Defines the response for the task status endpoint."""
    task_id: str
    status: Any # Can be "processing", a dict with summaries, or an error dict




# models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

# ... (all existing models for Paper Generation, etc. remain unchanged)

# --- UPDATED RESUME ANALYSIS MODELS ---
# In models.py, replace the entire "RESUME ANALYSIS MODELS" section with this:

# --- UPDATED RESUME ANALYSIS MODELS (to match frontend contract) ---

class AnalysisType(str, Enum):
    """Enumeration for the different types of resume analysis."""
    GENERAL = "general"
    DETAILED = "detailed"
    SKILLS = "skills"
    EXPERIENCE = "experience"

# Nested models to represent the structured JSON data
class ScoreModel(BaseModel):
    overall: int = Field(..., description="Overall score from 0 to 100.")
    skills: int = Field(..., description="Relevance and presentation of skills score.")
    experience: int = Field(..., description="Impact and quality of work experience score.")
    education: int = Field(..., description="Clarity and relevance of education score.")

class PersonalInfoModel(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None # <-- ADDED

class ExperienceModel(BaseModel):
    position: str
    company: str
    duration: str
    description: str # <-- ADDED

class EducationModel(BaseModel):
    degree: str
    institution: str
    year: str

class RecommendationsModel(BaseModel):
    strengths: List[str] = Field(..., description="List of key strengths.")
    improvements: List[str] = Field(..., description="List of areas for improvement.")
    suggestions: List[str] = Field(..., description="Actionable suggestions for the candidate.")

class ResumeAnalysisData(BaseModel):
    """The main data structure expected by the frontend component."""
    score: ScoreModel
    personalInfo: PersonalInfoModel
    summary: str
    skills: List[str]
    experience: List[ExperienceModel]
    education: List[EducationModel]
    recommendations: RecommendationsModel

class ResumeAnalysisResponse(BaseModel):
    """The final top-level response for the API endpoint."""
    success: bool = True
    data: ResumeAnalysisData