# Rig status / handoff (read this first in a new session)

*Updated 2026-10-05 16:35. Full report: [REPORT.md](REPORT.md) (results, equations, change log 13a).*

## What is running (unattended)
`tools/run_plan.py --from ppo_extension` (detached, log `data/run_plan.log`):
1. Done: **session 2c** (SAC 600 rig min 30/30; ODIL 3 rounds 28/24/25 of 30) and the **5 mm retest of
   session 2** (SAC 7/18/28/29 of 30 at 60/120/300/600 min; ODIL 17/25/25).
2. **PPO extension** running: `ppo1` to 1440 rig min (~1.4 wall min per rig min). Rolled back to
   `ppo_720min.zip` on 10-05 18:50 (720-851.7 aborted: ball pinned on a corner marker, see REPORT 13a).
   ETA Tue ~11:45. Restart after a recovery with `--from ppo_extension` (it resumes from `ppo_latest`).
   A ball 60 s at a marker now stops the session (recoverable: run_plan levels, rolls it off, resumes).
3. Queued after PPO (run_plan steps): `test_fitted` + `retest_fitted` (model-based baseline, 12 / 5 mm),
   `retest_ppo` (5 mm at 780/1080/1440), repeats `s3_odil`, `s3_sac`, `s4_sac`, `s4_odil`, `retest_s34`,
   `s3_ppo` (600 rig min). Roughly Tue 11:00 -> Wed ~18:00 (~30 h).
4. Off the rig: `data/fitted_queue.sh` (log `data/fitted_queue.log`) trains the baseline in ODIL's fitted
   physics (`rl_sim/train_fitted.py`, niced, one core): 2x SAC 500 k, 2x PPO 5 M steps, ETA Tue ~05:30.
   Must be done before `test_fitted` starts.

Recoverable stops (re-level failed, 3000-tick limit) are handled by run_plan itself (fresh camera
levelling + resume, event `auto-resume`). Anything else ends the plan with the plate held
(event `plan stopped`).

## How to check (cheap)
```
tail -5 phase3_logs/rig_learn.jsonl            # last events (tests, auto-resume, plan stopped/finished)
pgrep -fl "rig_learn|run_plan"                 # still running?
```
Live view: http://100.67.4.122:8001/?k=5kJvVEtywQeB

## If the plan stopped
1. Read the `why` of the last `shutdown` / `plan stopped` event.
2. Level: `cd physical_training && ../.venv-rl/bin/python3 -u tools/level_plate.py` (camera; never
   drive motors to stored positions; torque stays on).
3. Resume the run: `rig_learn.py ... --resume <rl_sim/runs/<run>_rig/<algo>_<N>min.zip> <N>` (SAC loads
   `sac_buffer_latest.pkl` from the same folder automatically). Use a new `--tag` for anything that is
   not a resume (never overwrite).
4. Log what happened in REPORT.md section 13a.

## Results so far (12 mm test, best per method)
- ODIL rig-only: 28/30 after 60 rig min, jerk ~0.0045 (4x smoother).
- SAC: 28/30 after ~120 min (session 1); session 2c 29/30 at 120 min.
- PPO: 23/30 at 60 min, 29/30 at 600 min; strong test-to-test swings.
- 5 mm retest: ODIL closest (4.3-4.9 mm); SAC hits most (28-29/30) but overshoots; PPO 24/30 at 600.
- Shape test (60 runs): ODIL tracker trained only from rig data (0 extra rig min) most accurate, 3.2 mm.

## After the plan
5 mm test of the best session-2 controllers; paper figures (learning curves vs rig time, precision,
shapes); update REPORT.md, the PDF guide and the one-page overview. Optional: a second PPO run on the
final software; PPO trained in the fitted model (M1 Max).

## User rules (also in CLAUDE.md)
Level the plate yourself with the camera; never move motors to stored positions; paper first (log every
change); ask before pushing to GitHub only if unsure (pushes have been authorised); phone alerts only
when the user is needed.
