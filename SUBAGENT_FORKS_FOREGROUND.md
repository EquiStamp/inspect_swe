# EquiStamp fork of inspect_swe: worker forks and foreground sub-agents

This branch (`subagent-forks-foreground`) stacks two Claude Code fixes on `subagent-attribution-111-114` (see [`SUBAGENT_ATTRIBUTION_111_114.md`](SUBAGENT_ATTRIBUTION_111_114.md)), so EquiStamp's internal control-tower bundle can pin one commit. It is not a product fork, and neither fix is filed upstream yet.

## What is on this branch

| Layer | Commit | What it is |
| --- | --- | --- |
| `subagent-attribution-111-114` at its head | `f24e38e` | Upstream #111 and #114, combined by hand. |
| EquiStamp | `7b6d467` | Worker forks are attributed to their own Agent call. A fork (`Agent` with `subagent_type: "fork"`) requests the main-thread model, so the subagent slug cannot identify it and every fork request was classified root. `classify()` now recognises a fork's request by how it ends: its own `Agent(fork)` call, that call's launch result, then a user turn starting `<fork-boilerplate>`. It returns `AgentBridgeContext("subagent", subagent_id=<that call's id>, subagent_type="fork")`, ahead of the slug checks and of the slug-drift warning. Upstream reports the misattribution in [#209](https://github.com/meridianlabs-ai/inspect_swe/issues/209). |
| EquiStamp | `f3b4765` | `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1` by default, overridable through `env`. Claude Code can launch an Agent call in the background without the model asking for it; under `--print` the main thread can then end its turn before any report returns, and the delegated work is lost. With the variable set, Agent calls, forks included, return their report as the tool result. |
| EquiStamp, this file | last commit | Documentation only. |

## Evidence

Check: one Claude Code 2.1.304 sample of a three-subsystem code audit whose system prompt asks for one sub-agent per subsystem, with `CLAUDE_CODE_SUBAGENT_MODEL_FORCE=1`, through a filter that logs `kind`, `subagent_id` and `subagent_type`. Each request's expected attribution is derived from its own messages, not from inspect_swe.

| Setup | Forks | Main thread gets every report | Task |
| --- | --- | --- | --- |
| `subagent-attribution-111-114`, Opus 4.6 | 48 of 48 fork requests classified root | no: 3 Agent calls returned `Async agent launched` | passed |
| `subagent-attribution-111-114`, Opus 4.6, forks off | none | no: the main thread ended its turn while 14 sub-agents ran; the deliverable was never written | failed |
| This branch, Opus 4.6 | 48 of 48 credited to their own fork (3 forks) | yes, 3 of 3 | passed |
| This branch, Sonnet 5 | 28 of 28 credited to their own fork (3 forks) | yes, 3 of 3 | passed |
| This branch, Sonnet 5, forks off | none | yes, 3 of 3 general-purpose sub-agents | passed |

Tests: `tests/test_claude_code_agent_context.py` (8 new fork cases) and `tests/test_claude_code_env.py` (2 new) pass; every `claude_code` test passes (144) against `inspect-ai==0.3.287`. This branch, like its base, does not import with the `inspect-ai` 0.3.277 in `uv.lock`.

## Pinning

As for the base branch: pin the commit, keep Claude Code on 2.1.304, and set `CLAUDE_CODE_SUBAGENT_MODEL_FORCE=1`. The fork check depends on the `<fork-boilerplate>` marker; a Claude Code version that changes it sends forks back to root without an error.

## Exit condition

Drop the fork check when upstream attributes forks (#209); drop the background default when upstream sets it or Claude Code stops backgrounding unasked under `--print`. The base branch's exit condition still applies.
