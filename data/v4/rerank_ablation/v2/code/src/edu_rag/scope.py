"""One exact curriculum scope shared by catalog and retrieval queries."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class CurriculumScope(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    course_id: UUID | None = None
    grade: int | None = Field(default=None, ge=1, le=12)
    textbook_edition: str | None = Field(default=None, min_length=1, max_length=200)
    semester: Literal["first", "second", "full_year"] | None = None
    subject: Literal["math", "physics", "chemistry", "english"] | None = None
    curriculum: str | None = Field(default=None, min_length=1, max_length=200)

    def sql_params(self):
        return {key: str(value) if isinstance(value, UUID) else value
                for key, value in self.model_dump().items() if key in CurriculumScope.model_fields}


# Alias c is always the authoritative course table. Values are bound parameters.
SCOPE_SQL = """
    (CAST(:course_id AS uuid) IS NULL OR c.id=CAST(:course_id AS uuid))
    AND (CAST(:grade AS integer) IS NULL OR c.grade=:grade)
    AND (CAST(:textbook_edition AS text) IS NULL OR c.textbook_edition=:textbook_edition)
    AND (CAST(:semester AS text) IS NULL OR c.semester=:semester)
    AND (CAST(:subject AS text) IS NULL OR c.subject=:subject)
    AND (CAST(:curriculum AS text) IS NULL OR c.curriculum=:curriculum)
"""
