from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass

from pipeline.fetch.data_manager import MarketDataManager
from pipeline.llm_events.incremental_matcher import run_incremental_pipeline
from pipeline.llm_bets.verify_bets import main as run_bet_matching
from pipeline.postprocess.clean_confirmed_matches import clean_files


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLUSTER_DIR = os.path.join(ROOT_DIR, "pipeline", "clustering")


@dataclass
class CorePipelineService:
    root_dir: str = ROOT_DIR

    def _run_script(self, filename: str, *args: str) -> None:
        subprocess.run([sys.executable, os.path.join(CLUSTER_DIR, filename), *args], check=True, cwd=self.root_dir)

    def refresh(
        self,
        batch_size: int = 40,
        llm_mode: str | None = None,
        apply_matches: bool | None = None,
        run_contract_matching: bool = False,
    ) -> dict[str, object]:
        manager = MarketDataManager()
        _, kalshi_stats = manager.sync_data("kalshi")
        _, polymarket_stats = manager.sync_data("polymarket")

        self._run_script("event_keyword_indexer.py", "--full")
        self._run_script("group_events_cluster.py", "--full")
        self._run_script("outcome_candidate_generator.py")
        clean_files()

        llm_mode = (llm_mode or os.environ.get("LLM_EXECUTION_MODE", "dry-run")).lower()
        if apply_matches is None:
            apply_matches = os.environ.get("LLM_APPLY_RESULTS", "false").lower() in {"1", "true", "yes"}
        if llm_mode == "dry-run":
            matching_stats = run_incremental_pipeline(batch_size=batch_size, dry_run=True)
        elif llm_mode == "sync":
            matching_stats = run_incremental_pipeline(batch_size=batch_size, apply=apply_matches)
        elif llm_mode == "batch":
            from pipeline.llm_events.batch_workflow import prepare_consensus_pass

            matching_stats = prepare_consensus_pass("primary")
        else:
            raise ValueError("LLM_EXECUTION_MODE must be dry-run, sync, or batch")

        if run_contract_matching and apply_matches:
            run_bet_matching()

        return {
            "kalshi": kalshi_stats,
            "polymarket": polymarket_stats,
            "matching": matching_stats,
        }
