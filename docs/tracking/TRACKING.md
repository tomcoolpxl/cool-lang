# Coolscript Implementation Tracking

## Recent Updates
- **2026-01-12:** Updated function keyword from `func` to `fn` across all documentation, examples, and test files to align with official grammar specification and modern language design (Rust/Kotlin style).

## Current Status
**Phase:** Milestone 1 (Week 4)
**Focus:** Runtime & Concurrency (C Runtime, Channels, Task Spawning).
**Completed:** Lexer, Parser (Full), AST, Semantic Analysis (Linear Types & No-Escape), Textual MLIR Codegen, LLVM Integration & Object File Generation, Basic Runtime (Alloc/Spawn/Print/Sleep), Function keyword standardization to `fn`.
**Active Tasks:** Exposing Channels to Coolscript.

## Dashboards
*   [Implementation Dashboard](DASHBOARD.md): detailed task breakdown and status for the 90-day sprint.
*   [Golden Tests](GOLDEN_TESTS.md): The "Gold Standard" test suite that defines Milestone 1 completion.

## Roadmaps & Plans
*   [Master Plan](../plans/MASTER_PLAN.md): High-level project vision.
*   [Milestone 1](../plans/MILESTONE_1.md): The Core Engine (Days 1-30).
*   [Milestone 2](../plans/MILESTONE_2.md): Scaling & Ergonomics (Days 31-60).
*   [Milestone 3](../plans/MILESTONE_3.md): Self-Hosting (Days 61-90).
*   [Build Orchestration](../plans/BUILD_ORCHESTRATION.md): How the build system fits together.

## Implementation Details
For technical deep-dives, see the [Internals](../internals) directory.
