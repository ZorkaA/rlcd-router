# Milestone 1 Iteration 2 Explorer 1: StreamExtractor Synthetic Vocab Remediation

You are teamwork_preview_explorer_m1_it2_1.
Working directory: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_it2_1
Parent: orchestrator (ce5bc762-f633-465c-9133-7ec43d0b5719)

MANDATORY INPUTS:
- /Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md
- /Users/jack/Downloads/rlcd-router/PROJECT.md
- /Users/jack/Downloads/rlcd-router/.agents/orchestrator/GATE_STATUS.md
- Reviewer 1 report: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_reviewer_m1_1/handoff.md
- Code under review: `src/data/stream_extractor.py`

YOUR ROLE & OBJECTIVE:
Investigate the root cause and provide the exact fix specification for Reviewer 1's Finding 1:
In `src/data/stream_extractor.py:460`, `StreamExtractor.stream_synthetic()` hardcodes `vocab_size: int = 151936`. When used with `get_synthetic_model()` (`vocab_size = 1000`), generated token IDs exceed the embedding table bounds, causing `IndexError: index out of range in self`.
Design the exact fix so `stream_synthetic` dynamically detects `self.model.config.vocab_size` when `vocab_size is None`, while still allowing explicit overrides.
Write analysis.md and handoff.md in your working directory and notify parent.

## 2026-09-17T07:56:30Z
<USER_REQUEST>
You are teamwork_preview_explorer_m1_it2_1.
Your working directory is: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_it2_1
Your parent is orchestrator (conversation ID: ce5bc762-f633-465c-9133-7ec43d0b5719).

MANDATORY INPUTS:
- Read /Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md
- Read /Users/jack/Downloads/rlcd-router/PROJECT.md
- Read /Users/jack/Downloads/rlcd-router/.agents/orchestrator/GATE_STATUS.md
- Read /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_reviewer_m1_1/handoff.md
- Read /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_explorer_m1_it2_1/DISPATCH.md

YOUR OBJECTIVE:
Investigate and design the exact fix for Reviewer 1's Finding 1 in `src/data/stream_extractor.py:460`:
Make `stream_synthetic` dynamically detect `self.model.config.vocab_size` when `vocab_size is None`.
Write analysis.md and handoff.md in your working directory and notify parent.
</USER_REQUEST>
