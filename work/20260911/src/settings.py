"""Experiment constants and paper provenance."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    T: int = 1000
    M: int = 20
    S: int = 100
    N: int = 20

    def __post_init__(self) -> None:
        if any(type(v) is not int or v <= 0 for v in (self.T, self.M, self.S, self.N)):
            raise ValueError("T, M, S, N must be positive integers")
        if self.T % self.M:
            raise ValueError("T must be divisible by M for equally spaced completed updates")

    @property
    def trajectories(self) -> int:
        return self.N * self.S


PAPER = {
    "url": "https://arxiv.org/abs/2608.14067",
    "version": "2608.14067v1",
    "title": "When Denoising Hurts: Rethinking the Terminal Step of Diffusion Time Series Forecasters -- Extended Version",
    "authors": ["Dat Nguyen-Cong", "Luong Tran", "Tung Kieu"],
    "implementation": "Paper-faithful Eq. 7, Eq. 8, Algorithm 1, Appendix D; scRNA grouping and completed-update snapshot adaptations",
    "official_code_status": "not found in public sources checked on 2026-09-12 JST; absence is not proven",
    "upstream_repository": None,
    "upstream_commit": None,
    "upstream_license": None,
    "reused_upstream_functions": [],
    "search_record": "work/20260911/PAPER_PROVENANCE.md",
}
