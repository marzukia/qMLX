# qMLX #93 SCRUB — census (run 2026-10-29, pre-rewrite)

Live dev tip at census: `afac2ea884e2ed267d905907a3b7f03ff41f25d5` (verified via `git ls-remote origin dev`).
Repo: marzukia/qMLX. Open PRs at census: **0** (verified `gh pr list --state open` = []).
Refs in play: ~120 local/remote branch refs, 5 tags (v0.13.0, v0.14.0, v0.15.0, v0.15.1, v0.16.0), 1427 commits total, 19 unique `salvage.tar` blobs.

## Pattern → hits

| Pattern | Tree (current, per-branch) | History (git log --all -S / grep) | PR/issue (title+body+comments) |
|---|---|---|---|
| marzuki-helium | 1 file × ~50 branches: `docs/tape-rollback-status.md:156` (dev tip incl.), `reap-models.html:241` (main + ~20 branches), `bench/bench_mtp_k3.py:12` (~15 branches) | 28 commits | PR #40, #49 body (closed); PR #55 body (closed) |
| marzuki-carbon *(family hostname, found by census)* | `docs/tape-rollback-status.md:155` (~30 branches incl. dev tip) | in ~14 non-tar + all 11 tar commits (23 occurrences in tar) | issue #52 body ×4; PR #55 body ("Marzuki-carbon", case variant) |
| marzuki-hydrogen | none (absent from all live trees) | 1 commit (f1e539a, in salvage tar) | PR #55 body; issue #93 |
| marzuki-neon / -lithium / -oxygen / -helium-1 | none | 0 | issue #93 only |
| 100.95.144.25 | none | 0 | issue #93 only |
| 100.94.200.100 | none (non-tar) | 11 commits (salvage.tar blobs), 23 occurrences | issue #93 only |
| 100.73.242.40 / 100.91.94.56 | none | 0 | issue #93 only |
| other 100.64.0.0/10 tailnet IPs | none found (regex sweep of trees + all 19 tar extractions) | none found | none |
| andryo@ (email, any domain) | none | 2 commits (tar): `andryo@mrzk.io`, `andryo@nestfinder.au` | issue #93 only |
| commit **author/committer** `Andryo Marzuki <andryo@mrzk.io>` | n/a | **124 commits** (author+committer header) | n/a |
| /home/andryo | none | 0 | issue #93 only |
| "Andryo" as maintainer credit | `CODE_OF_CONDUCT.md:63`, `CONTRIBUTORS.md:5`, `NOTICE:2`, `pyproject.toml:13` (dev tip) | commit 0b29076 msg | PR #54/#60/#75/#84 bodies ("Operator order (Andryo)") — judgment: kept |
| neonvm@ | none | 0 | issue #93 only |
| 192.168.122.x | none | 0 | issue #93 only |
| /mnt/models | none | 0 | none |
| junkyard.sh / *.junkyard.sh | `gh-pages:index.html:178` (umami.junkyard.sh tracker script); `fleetwork-salvage-commit/18:REPORT.md:5` (drop.junkyard.sh) | 5 commits (97e08a5 gh-pages, 0d67724 + tar blobs) | issues #61/#62/#63 bodies (https://junkyard.sh/h/rep/… links); issue #93 |
| nestfinder / nestfinderau (fleet repo refs, in tar transcripts) | none (non-tar) | 3 commits (tar) | none |

## Committed session transcripts (path-drop candidates)

- No direct `.fleetwork/` paths committed in any tree. Session transcripts exist **inside** `salvage.tar` blobs (19 unique) on the 11 `fleetwork-salvage/*` branches + `fleetwork-salvage-commit/18`: `.fleetwork/session.jsonl` (full LLM session log, incl. 100.94.200.100 endpoint, nestfinderau, andryo@ emails), `.fleetwork/llm-stream.json`, `REPORT.md` (drop.junkyard.sh link). Since they are embedded in a single tar blob, they cannot be path-dropped individually → judgment: scrub in place via `--replace-text` (keeps salvaged work intact). Reported in scrub-report.md.

## PR/issue census detail

- Open PRs: 0 (expect 0 — verified).
- Bodies needing API edit: PR #40 (marzuki-helium), PR #49 (marzuki-helium), PR #55 (Marzuki-carbon + marzuki-helium), issue #52 (marzuki-carbon ×4), issues #61/#62/#63 (junkyard.sh link ×1 each).
- Issue #93 (open, the tracking issue) contains the pattern set by definition → left untouched.
- All PR/issue comments checked (all 12 numbered items with body hits + #93): **0 comment hits**.
- `mrzk.io` (project site domain in gh-pages index.html/CNAME; `ag.mrzk.io`/`marzuki.local` author emails on agent commits): outside the pattern set → left, reported as residual observation.
