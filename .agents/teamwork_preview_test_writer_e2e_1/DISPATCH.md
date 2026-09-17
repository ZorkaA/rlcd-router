# E2E Test Writer Task (Dual Track: E2E Testing)

## 2026-09-17T07:31:42Z

You are teamwork_preview_test_writer_e2e_1.
Your working directory is: /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_test_writer_e2e_1
Your parent is orchestrator (conversation ID: ce5bc762-f633-465c-9133-7ec43d0b5719).

MANDATORY INPUT:
- Read /Users/jack/Downloads/rlcd-router/ORIGINAL_REQUEST.md
- Read /Users/jack/Downloads/rlcd-router/PROJECT.md
- Read /Users/jack/Downloads/rlcd-router/.agents/teamwork_preview_test_writer_e2e_1/DISPATCH.md

YOUR ROLE & OBJECTIVE:
You are the E2E Testing lead. Implement the comprehensive opaque-box test suite for the Asynchronous MoE Router Phase 1 Calibration project.
Your tasks:
1. Create /Users/jack/Downloads/rlcd-router/TEST_INFRA.md following the specification in PROJECT.md and the Project Pattern.
2. Build the test suite under /Users/jack/Downloads/rlcd-router/tests/:
   - tests/__init__.py
   - tests/conftest.py: Fast synthetic model fixtures (Qwen2MoeConfig with d=64, 6 layers, 16 experts, top-k=4) and synthetic tensor fixtures.
   - tests/test_tier1_features.py: >=5 tests per feature covering all features in PROJECT.md § Feature Inventory.
   - tests/test_tier2_boundaries.py: >=5 tests per feature covering boundaries, edge cases, zero inputs, trailing token masks.
   - tests/test_tier3_pairwise.py: Pairwise feature interactions (layer buckets x horizons, CE + MMCE loss, NLL + L2 reg).
   - tests/test_tier4_workloads.py: Real-world workflows (end-to-end pipeline run, memory leak assertions, targeted ECE at 0.05 and 0.85).
3. Create /Users/jack/Downloads/rlcd-router/TEST_READY.md when test infrastructure and test suites are complete.
4. Verify by running `pytest -v tests/` (tests should pass against synthetic fixtures, or skip gracefully if implementation modules are not yet installed).
5. Document all findings and results in your handoff report and notify parent.
