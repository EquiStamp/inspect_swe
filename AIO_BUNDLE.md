# EquiStamp fork of inspect_swe for the AIO bundle

This branch (`aio-bundle`) carries two unmerged upstream pull requests, stacked, so the AIO bundle can pin one commit. It is not a product fork: use upstream [`meridianlabs-ai/inspect_swe`](https://github.com/meridianlabs-ai/inspect_swe) once both PRs are merged and released.

## What is on this branch

| Layer | Commit | What it is |
| --- | --- | --- |
| Upstream PR [#111](https://github.com/meridianlabs-ai/inspect_swe/pull/111) at its head | `2f464e8` | Filters see which sub-agent a request came from (`subagent_id`, `subagent_type`) through `AgentBridgeContext`. |
| Upstream draft PR [#114](https://github.com/meridianlabs-ai/inspect_swe/pull/114), two commits | `9de3d8c`, `b6150d3` | Keeps an async Task pending until it completes, so every request of a sub-agent launched with `Async agent launched` keeps its id. Cherry-picked onto #111's head, with the conflict in `src/inspect_swe/_claude_code/_events/live_consumer.py` resolved by hand. |
| EquiStamp, tests only | the commit after `b6150d3` | Restores #111's `tests/test_claude_code_agent_context.py` and carries #114's tests in `tests/test_pr114_async_tasks.py`, with four assertions updated to #111's semantics. |
| EquiStamp, this file | last commit | Documentation only. |

## Why the two PRs need combining by hand

#114 is stacked on #111, but on an 8 August revision of it (`2fc180c`). #111's head is 74 commits later and includes the commit that adds `subagent_id` and `subagent_type` to filters (`e295052`) and the commits that detach a Task when its launch reply arrives (`414cf7e`, `2ebcbec`). #114 changes the same code in `live_consumer.py`, so it does not apply to #111's head. No #111 commit has both the id feature and a clean merge with #114.

What the hand resolution does: on an async launch reply, the Task stays pending (`launched_async = True`) instead of being detached; a Task is detached only when it completes or when a completion notification arrives. The resolution is unreviewed upstream.

## Evidence

Check: one Claude Code 2.1.304 sample that launches a foreground general-purpose, Explore and Plan sub-agent and one background sub-agent, with `CLAUDE_CODE_SUBAGENT_MODEL_FORCE=1`, on Haiku 4.5, through a filter that logs `kind`, `subagent_id` and `subagent_type` ([EquiStamp/EU3_Control_Issues#215](https://github.com/EquiStamp/EU3_Control_Issues/issues/215)).

| Setup | Requests with an id (foreground sub-agents) | Kinds and types |
| --- | --- | --- |
| #111 at `2f464e8` alone | 7 of 19 in the runs on this setting | correct for every request |
| This branch | all, in 2 of 2 runs (background agent included) | correct for every request |

Tests: `tests/test_claude_code_agent_context.py` and `tests/test_pr114_async_tasks.py` pass (55). The rest of the suite: 656 passed; `test_bundled_catalog_tracks_live_latest` needs the network and fails offline; `test_home_isolation.py` and `test_trajectory_parsing.py` need `minisweagent`, which was not installed.

## Pinning

Pin the commit, not the branch:

```toml
"inspect-swe @ git+https://github.com/EquiStamp/inspect_swe@<commit sha of this branch's head>"
```

Claude Code must stay on 2.1.304 and `CLAUDE_CODE_SUBAGENT_MODEL_FORCE=1` must be set (since 2.1.251 Explore and Plan sub-agents ignore `CLAUDE_CODE_SUBAGENT_MODEL` without it); #111 does not set it.

## Exit condition

Drop this fork when #111 and #114 are both merged and released upstream, or when #114 is rebased onto #111's head. Then pin the upstream release.
