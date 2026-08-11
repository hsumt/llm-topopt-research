"""Deterministic data contracts for ingested research papers and chunks."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PaperMetadata(BaseModel):
    """Stable metadata recorded for one ingested PDF."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["1.0"] = "1.0"
    paper_id: str = Field(min_length=1)
    source_filename: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    title: str | None = Field(default=None, min_length=1)
    authors: list[str] = Field(default_factory=list)

    page_count: int = Field(ge=1)
    extracted_page_count: int = Field(ge=1)
    extracted_character_count: int = Field(ge=1)

    extractor: Literal["pypdf"] = "pypdf"

    @field_validator("paper_id", "source_filename")
    @classmethod
    def reject_blank_identifiers(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Identifier fields cannot be blank")
        return value

    @field_validator("authors")
    @classmethod
    def validate_authors(cls, authors: list[str]) -> list[str]:
        if any(not author.strip() for author in authors):
            raise ValueError("Author names cannot be blank")
        return authors

    @model_validator(mode="after")
    def validate_extracted_pages(self):
        if self.extracted_page_count > self.page_count:
            raise ValueError(
                "extracted_page_count cannot exceed page_count"
            )
        return self


class PaperChunk(BaseModel):
    """One deterministic, page-traceable unit of retrieved paper text."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["1.0"] = "1.0"
    paper_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    chunk_index: int = Field(
        ge=0,
        description="Zero-based position in the paper's ordered chunk list.",
    )
    page_start: int = Field(
        ge=1,
        description="One-based first PDF page represented by this chunk.",
    )
    page_end: int = Field(
        ge=1,
        description="One-based last PDF page represented by this chunk.",
    )
    text: str = Field(min_length=1)

    @field_validator("paper_id", "chunk_id", "text")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Chunk identifiers and text cannot be blank")
        return value

    @model_validator(mode="after")
    def validate_page_range(self):
        if self.page_end < self.page_start:
            raise ValueError("page_end cannot be less than page_start")
        return self