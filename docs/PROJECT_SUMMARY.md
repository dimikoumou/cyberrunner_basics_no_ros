# CyberRunner ball-on-plate and maze — project summary (for Claude chat)

*Compact, self-contained summary of everything done from 2026-09-26 to 2026-10-07. Add this file to a
Claude Project (claude.ai → Projects → project knowledge) to use it in chat. Full detail: the advisor
report https://claude.ai/code/artifact/bd187982-2e14-4986-b778-577de74b7b0f and the GitHub repo
https://github.com/dimikoumou/cyberrunner_basics_no_ros (branch `ball-on-plate`).*

## The rig
CyberRunner plate (~26 × 21.6 cm, white lined paper, a taped-over hole near the centre), 1.3 cm steel
ball, two Dynamixel motors (IDs 1 and 3) tilt it, motor 2 = ball-reload elevator. One overhead camera:
ball position + plate tilt from 8 frame markers, 29 Hz. Commands ±4° through a camera-closed tilt servo.
A 4-core Intel Mac runs it; an M1 Max for offline work.

## Part 1 — ball on plate, simulation → rig (Sep 26–27)
- Vision fixed (half-hidden outer markers cached, colour ranges retuned), classic controller, web UI,
  hole + elevator auto-reload.
- Classic: first 10 s hold; ball rests 2.6 mm from centre for 260 s; line following 8.7 laps/min.
- Calibrated simulator (1-s replay error 4.3 mm vs 20.5 mm baseline). PPO v3 in sim: 92 % vs 65 % (PD);
  ran on the rig zero-shot.
- ODIL developed from 56 % to 97 % in sim (stiction compensation, delay-robust model). Rig, 120 targets:
  as accurate as PPO, ~2.5× smoother. Rig-fitted ODIL v11 from 30 min of logs.

## Part 2 — path tracking and the real maze (Sep 27–29)
- ODIL tracker + closed-loop refinement: drawing error 3.6 mm vs 4.2 classic, worst deviation halved.
- Maze on paper (virtual holes): 1259 runs, best 98 %, mean ~33 %.
- **Real labyrinth completed 3 times out of 167 runs** (first 2026-09-28: 1.93 m in 172 s). Typical run
  ~25 % of the route; falls come from the ball being 2–3× too fast near certain holes.
- Learning loop (drive → world model → refine) has not improved maze runs yet (open question).
- Safety lesson: two motor runaways (Sep 29) from driving motors to stored positions → never again.

## Part 3 — learning only on the rig: ODIL vs SAC vs PPO (Oct 2–7) — for a paper
Test: 30 frozen targets; reached = within R in 8 s; 12 mm test + 5 mm retest; every frame recorded.
Runs on final software: s2c, s3, s4 (ODIL 3 × 30 min rounds; SAC 240–600 min); PPO ppo1 (1440 min)
and s3_ppo1 (600 min). Model-based baseline: SAC/PPO with rig settings trained only inside ODIL's
fitted physics model (from 30 / 90 rig min of data).

| Controller | Rig min | 12 mm (of 30) | 5 mm (of 30) | Jerk |
|---|---|---|---|---|
| ODIL | 30 | 26.0 (24–28) | 23.0 (17–27) | 0.003–0.005 |
| ODIL | 90 | 25.0 (23–27) | 26.0 (25–27) | ~0.003 |
| SAC | 60 | 27.0 (25–29) | 13.3 (7–20) | ~0.017 |
| SAC | 120 | 29.3 (29–30) | 23.0 (18–28) | ~0.017 |
| SAC | 240–300 | 30.0 | 28.3 (27–30) | ~0.018 |
| PPO | 600 | 26.0 (23–29) | 24 | ~0.019 |
| PPO | 1440 | 27 | 19 | ~0.019 |
| PPO in ODIL's model | 30 / 90 data | 28 / 30 | 17 / 18 | ~0.018 |
| SAC in ODIL's model | 30 / 90 data | 16 / 14 | 13 / 18 | ~0.017 |

Shape test (session 1, 60 runs): ODIL tracker from the same rig data, 0 extra rig minutes, most accurate
(3.2 ± 0.4 mm); SAC 4.2, PPO 4.0 (13/15 completed).

**Interpretation (honest paper claim):** a small physics model fitted from ~30 rig minutes is the main
win — model-based controllers reach 26–28/30 in 30 min, SAC needs ~120, PPO 600+ and never settles.
PPO inside ODIL's model matches ODIL at 12 mm, so the claim is not "ODIL beats RL" but "fit a model
from minutes of data; ODIL is a fast, very smooth and more precise way to use it". ODIL does not improve
after its first round (open). One ODIL test (run s4, 60 min, 9/30: ball stuck in the taped hole) is left
out of the means and disclosed as a footnote.

Unattended operation needed: marker guard, corner shield, dip twitch for the hole, a 60-s marker stop,
plausibility check, jump filter, re-level pauses + watchdog, an unattended runner that levels by camera
and resumes (SAC with its replay memory), never-overwrite files. Every change and incident is logged
(REPORT.md 13a).

## Publication status and next steps
Now: workshop / short conference paper. Proposed to strengthen (not started): (1) ablation of ODIL's
boost, smoothness term and delay model; (2) fitting the physics + a learned plate slope map and the
controller in one ODIL optimisation; (3) the real maze as showcase (vs CyberRunner, Bi & D'Andrea 2023).
Advisors to decide: workshop now, or full paper with the maze.

## User rules for the rig
Level the plate only by camera from where the motors are; never drive motors to stored positions or
loosen caps; never blame hardware — solve in software; log every protocol change; ask before writing
Dynamixel registers or deleting files; phone alerts only when needed.

## References
Karnakov, Litvinov, Koumoutsakos (2024) ODIL, PNAS Nexus 3(1) pgae005 · Karnakov, Amoudruz,
Koumoutsakos (2025) arXiv:2506.15902 · Haarnoja et al. (2018) SAC · Schulman et al. (2017) PPO ·
Bi, D'Andrea (2023) CyberRunner · Raffin et al. (2021) Stable-Baselines3.
