# Is the CyberRunner ball-on-plate work publishable? — assessment and plan

*Written 2026-09-27. Research only: nothing was run on the rig and nothing was committed. Numbers
about our own work come from [`BALL_ON_PLATE_REPORT.md`](BALL_ON_PLATE_REPORT.md) and
[`../REPORT.md`](../REPORT.md). Every outside claim cites a source (title + URL). Deadlines were
checked on the web on 2026-09-27; confirm them again before you rely on them.*

---

## 0. Short answer

**Yes, it can be published, but not in its current form.** The engineering is solid and well
documented, but the evidence is anecdotal: 3 RL trips, 8 paired trips and single long holds.
Ball-on-plate by itself is a textbook system with decades of papers, so "we balanced a ball" or
"PPO works zero-shot on a ball-balancer" won't get accepted on its own.

The **publishable core** is narrower and more interesting:

> **Rest-to-rest positioning of a rolling ball on a stiction-dominated (paper) surface, with
> vision-in-the-loop delay: a controlled real-hardware comparison of (i) a friction-aware classical
> controller, (ii) zero-shot sim-to-real RL from a log-calibrated simulator, and (iii) ODIL-style
> discrete-loss policy optimisation, on the open CyberRunner platform.**

The strongest single contribution, **if it works on hardware**, is **(iii): ODIL control on a
real robot**. I found no report of ODIL closed-loop control on physical hardware (see §1.3).
Without ODIL, the paper is the classic-vs-RL comparison plus the stiction handling. That is a
decent **workshop, ECC or L4DC** paper but not a strong ICRA/IROS paper.

Best-fit venues, given the dates:
- **L4DC 2027**, deadline **13 Nov 2026**. Learning plus control, and a real-hardware comparison fits.
- **ECC 2027**, deadline **31 Oct 2026**. Control-oriented; the classic friction-aware controller fits.
- **IROS 2027 or RA-L (with the IROS option)**, deadline **1 Mar 2027**. The full, polished version
  with all three methods and statistics.
- ICRA 2027 (16 Sep 2026) and the ACC 2027 joint L-CSS deadline (11 Sep 2026) have already passed.
  The regular ACC deadline (2 Oct 2026) is 5 days away, which is not realistic.

**Talk to the ETH/IDSC side before you submit anything, arXiv included** (see §3).

---

## 1. Novelty assessment

### 1.1 What already exists (so it is *not* new)

| Area | Prior work | Consequence for us |
|---|---|---|
| Ball-and-plate control in general | Large literature: PID, LQR, sliding-mode, fuzzy, state feedback. E.g. *Commparison Between Different Methods of Control of Ball and Plate System with 6DOF Stewart Platform* (https://www.researchgate.net/publication/282268773); *Control of a ball-and-plate system using a State-feedback controller* (https://www.scielo.cl/pdf/ingeniare/v28n1/0718-3305-ingeniare-28-01-6.pdf) | Classic PID/PD on a plate is not a contribution |
| Friction compensation on ball-and-plate | *A novel disturbance-observer based friction compensation scheme for ball and plate system*, ISA Transactions (https://www.sciencedirect.com/science/article/abs/pii/S0019057813002012) | Friction on ball-and-plate has been studied. Our angle has to be the **strong breakaway/rolling ratio of a compliant surface** (1.5–2.4° vs ~0.4°) and **spatially varying stiction** |
| Stiction and stick-slip control theory | Yang & Tomizuka 1988 (adaptive pulse width); Armstrong-Hélouvry et al. 1994; *Stick-slip and convergence of feedback-controlled systems with Coulomb friction* (https://arxiv.org/pdf/2006.08977); *Analysis of relay-based feedback compensation of Coulomb friction* (https://arxiv.org/pdf/2205.09352) | Our pulse+brake and ramp-kick are **applications** of known ideas. Novelty only in the combination and the setting |
| Learning-based control on real ball-and-plate | *Learning Control from Raw Position Measurements*, Amadio et al., ACC 2023, with a real ball-and-plate rig (https://arxiv.org/abs/2301.13183); *Adaptive Optimal Trajectory Tracking Control Applied to a Large-Scale Ball-on-Plate System*, Köpf et al. (https://arxiv.org/abs/2010.13486) | "Learning on a real ball-and-plate" is done |
| Sim-to-real RL with domain randomisation on ball balancers | Muratore et al.: *Assessing Transferability from Simulation to Reality for RL* (https://arxiv.org/abs/1907.04685); *Bayesian Domain Randomization for Sim-to-Real Transfer* (https://www.researchgate.net/publication/339737466); *Benchmarking Sim-2-Real Algorithms on Real-World Platforms* (https://www.ias.informatik.tu-darmstadt.de/uploads/Team/FabioMuratore/Menzenbach--BenchmarkingSim2RealAlgorithmsOnRealWorldPlatforms.pdf). These use the Quanser Ball Balancer among their platforms; please check each paper for its exact set. Microsoft Project Moab: RL with domain randomisation, deployed on a real ball-balancing robot (https://microsoft.github.io/moab/tutorials/1-balance/index.html) | **Zero-shot PPO + domain randomisation on a ball balancer is known.** Our variant: centre-stabilisation is replaced by *goal-conditioned rest-to-rest* moves with *heavy stiction* and a *log-calibrated* simulator |
| CyberRunner platform | Bi & D'Andrea, *Sample-Efficient Learning to Solve a Real-World Labyrinth Game Using Data-Augmented Model-Based RL*, ICRA 2024 (https://arxiv.org/abs/2312.09906); Gaber, Bi & D'Andrea, *Adaptive Nonlinear MPC for a Real-World Labyrinth Game*, CDC 2024 (https://arxiv.org/abs/2406.08650) | The platform, model-based RL on it and MPC on it are all published. **Flat-plate positioning with a compliant surface on CyberRunner has not been, as far as I found** |
| Zero-shot RL from a simple analytic model vs tuned controller (same group) | Jiang, Bi, D'Andrea & Ramachandran, *Cross-Platform Control for Autonomous Surface Vehicles via Adaptive RL*, arXiv 2607.02037, Jul 2026 (https://arxiv.org/abs/2607.02037) | The **same group** already runs this type of study (simple analytic sim → zero-shot RL → compared with a tuned controller). This is a good template, and it shows what they expect: an adaptive/teacher-student baseline, % error improvements and multiple platforms |
| ODIL | Karnakov, Amoudruz & Koumoutsakos, *Optimal Navigation in Microfluidics via the Optimization of a Discrete Loss*, PRL 134, 044001 (2025) (https://arxiv.org/abs/2506.15902; https://doi.org/10.1103/PhysRevLett.134.044001). Framework: https://github.com/cselab/odil; original ODIL for PDEs: https://arxiv.org/abs/2205.04611 | The method is published and demonstrated **in simulation** (microswimmers and flows). Reported: more robust than RL and up to ~3 orders of magnitude faster |
| Differentiable-simulation policy learning on real robots | *Learning on the Fly: Rapid Policy Adaptation via Differentiable Simulation* (https://arxiv.org/abs/2508.21065); *Real-Sim-Real loop with differentiable simulation* (https://arxiv.org/abs/2503.10118); *A Review of Differentiable Simulators* (https://arxiv.org/pdf/2407.05560) | "Gradient-through-a-model policy vs PPO on hardware" has been done in other domains. ODIL differs from these because it optimises **trajectories and policy jointly with a physics residual** instead of rolling out the simulator. Frame it precisely that way |
| Friction-aware sim-to-real RL for rolling/spherical systems | *asRoBallet: Closing the Sim2Real Gap via Friction-Aware RL for Underactuated Spherical Dynamics* (https://arxiv.org/html/2604.24916v2) | Recent and adjacent: friction modelling for sim-to-real. Cite it and state the difference |

### 1.2 What is plausibly new (defensible claims)

1. **Stiction-dominated ball-on-plate as a studied regime.** Breakaway ≈ 1.5–2.4° against rolling
   ≈ 0.3–0.6° (4–6×). Stiction varies with position (0.7–3.7° across the map, 3.4–3.9° near the frame),
   and the ball sinks into a compliant surface over time. Most ball-and-plate papers use hard plates
   and treat friction as a nuisance. A clean **characterisation** would be a useful contribution and
   is cheap to produce: breakaway vs. dwell time, a spatial map, repeatability across days.
2. **A friction-aware classical baseline that actually works in that regime.** It combines a
   Yang–Tomizuka adaptive pulse width per cell, a learned local-slope map, planned trapezoid
   rest-to-rest moves with friction feed-forward, and a delay-aligned closed-loop tilt servo.
   Results: 2.6 mm median rest error, 260 s holds, one-motion moves. Each piece is known; the
   integrated, ablated system in this regime is a reasonable engineering contribution.
3. **Zero-shot RL from a simulator calibrated on rig logs**, with a hard action-rate limit and a
   measured 1-s replay error (3.4/4.3 mm vs 12.9/20.5 mm for the "stays put" baseline). There is also
   an honest failure analysis: RL is faster, but loses the last millimetres and has no slope memory.
   That result, and the hybrid that follows from it, is instructive.
4. **ODIL on real hardware.** A search for ODIL follow-ups (arXiv, Google Scholar-style queries)
   found only the simulation work of Karnakov et al. and the PDE framework. **If** an ODIL-trained
   policy runs on the plate and is compared fairly with PPO (wall-clock, samples, success,
   precision), that is the most novel claim available: *"first hardware demonstration of ODIL
   closed-loop control; comparison with RL and a friction-aware classical controller on a
   stiction-dominated system"*. Caveat: our differentiable model has **no stiction**, so the result
   depends on how the gap is handled. A hybrid near-field settle or a smooth stiction surrogate
   would make a good ablation.
5. **An open, cheap testbed/benchmark extension of CyberRunner** (flat plate, paper sheets with
   printed/drawn goals, logs and simulator). This is useful to the community but, on its own, only
   a workshop/dataset-track contribution.

### 1.3 What is incremental or weak right now

- **Sample sizes.** RL on the rig: 3 trips. Comparison: 8 trips, one seed, 2 target sizes. The
  holds are single runs. Reviewers will ask for ≥ 30 trials per condition with CIs.
- **RL baseline strength.** PPO without history adaptation, without an integral/slope-memory input,
  and without fine-tuning on the rig. The same group's recent paper uses an adaptive teacher-student
  RL baseline (arXiv 2607.02037). Expect reviewers from that community to ask for something similar.
- **Classical baseline fairness.** The classical controller was hand-tuned over two days with many
  iterations. The RL controller got three training versions. You must state the tuning budget or
  equalise it.
- **Scope limits.** Targets must be ≥ 5 cm from the frame, and the camera sometimes freezes. That is
  fine to report, but state it plainly.
- **"Multi-agent workflows (analysts + skeptic)" were used to pick changes.** This is AI-assisted
  engineering. IEEE requires disclosure of AI-generated content (text, figures, code) in the
  acknowledgments, and AI cannot be an author (IEEE RAS generative-AI guidelines:
  https://www.ieee-ras.org/publications/guidelines-for-generative-ai-usage/; IEEE author guidance:
  https://open.ieee.org/author-guidelines-for-artificial-intelligence-ai-generated-text/).
- **ODIL is not done yet.** `rl_sim/odil_plate.py` exists, but `rl_sim/runs_odil_v1.log` is empty.
  Nothing about ODIL is claimable until it runs in sim, and then on the rig.

### 1.4 Candidate framings, ranked

| Rank | Framing | Novelty | Work still needed | Venue fit |
|---|---|---|---|---|
| **A** | *ODIL vs RL vs friction-aware classical for rest-to-rest ball positioning on a stiction-dominated surface (real hardware)* | Highest (first hardware ODIL) | ODIL working in sim + rig; ≥ 30 trials × 3 methods × 2–3 target sizes; compute/sample budgets | L4DC 2027 (tight), IROS/RA-L 2027 (comfortable) |
| B | *Zero-shot sim-to-real on a stiction-dominated ball-on-plate: log-calibrated simulator, action-rate limits, and where learned control loses to a classical controller* | Medium | Statistics; ablations (no calibration / no action history / no rate limit); stronger RL baseline | ECC 2027, L4DC 2027 |
| C | *Friction-aware classical control of a rolling ball on a compliant surface* (characterisation + pulse/brake + slope map + planned moves) | Medium-low (mostly known pieces) | Characterisation experiments; ablation table; statistics | ECC 2027, a workshop |
| D | *Open flat-plate benchmark on CyberRunner* (logs, simulator, UI, tasks) | Low-medium | Clean code + data release, licence clarity (§3.4) | ICRA/IROS workshop, arXiv |

**Recommendation:** aim for **A** at **IROS 2027 / RA-L (1 Mar 2027)**. If the supervisors want an
earlier milestone, cut a **B (+C)** paper for **L4DC (13 Nov 2026)** or **ECC (31 Oct 2026)**. Do
**not** publish B/C in a way that uses up the novelty of A. For example, keep the ODIL hardware
results out of an early workshop paper, or treat the early paper as a short workshop version the
group is fine with.

---

## 2. Venues and dates (checked 2026-09-27)

| Venue | Deadline | Event | Fit | Source |
|---|---|---|---|---|
| ICRA 2027 (Seoul) | **16 Sep 2026: passed** | 24–28 May 2027 | Would have fit A | https://2027.ieee-icra.org/announcements/call-for-technical-papers/ |
| ACC 2027 (Philadelphia) | Joint ACC+L-CSS 11 Sep 2026 (passed); ACC-only **2 Oct 2026** (firm) | 7–9 Jul 2027 | B/C, but 5 days away: not realistic | https://acc2027.a2c2.org/ |
| **ECC 2027** (Brussels) | **31 Oct 2026** (notification 5 Mar 2027) | 13–16 Jul 2027 | **B/C**, control audience, likes real experiments | https://ecc27.euca-ecc.org/ |
| **L4DC 2027** (KTH Stockholm) | **13 Nov 2026** (OpenReview; decisions 25 Jan 2027; late-breaking results 5 Apr 2027) | 16–18 Jun 2027 | **A or B**: learning plus control, values honest hardware comparisons | https://l4dc2027.control.ee.ethz.ch/ |
| **IROS 2027** (Florence), or **RA-L with IROS option** | **1 Mar 2027** | 26 Sep–1 Oct 2027 | **A (full version)** | https://mldeadlines.com/conference/iros-2027/ ; RA-L: https://www.ieee-ras.org/publications/ra-l/ |
| CDC 2027 / L-CSS + CDC | Not yet announced. Past pattern: L-CSS+CDC ≈ 17 March (2025, 2026) | Dec 2027 | A/C for a control audience; the group published CyberRunner MPC at CDC 2024 | https://cdc2026.ieeecss.org/authors/author-information (2026 pattern) |
| IFAC World Congress | Last one was 2026. Next is 2029 | — | Not relevant now | — |
| ICRA 2027 workshops / late-breaking | Usually announced winter/spring. Check the ICRA site | May 2027 | D, or a short version of B | https://2027.ieee-icra.org/ |
| arXiv | Any time, **after supervisor approval** | — | Pre-print of whichever paper | — |

Student-led fit: ECC and L4DC accept 6–8-page-class papers that a student can realistically finish.
Check each CFP for exact page limits. IROS/RA-L is the natural target for the full ETH-group paper.
D'Andrea's group has published CyberRunner work at ICRA and CDC, and the recent ASV work is on arXiv
(cs.RO).

---

## 3. Process with ETH Zurich / IDSC (D'Andrea group)

### 3.1 Who is who (verified), and "Aswin" (unconfirmed)

- **Prof. Raffaello D'Andrea**: Professor of Dynamic Systems and Control, IDSC, ETH Zurich
  (https://en.wikipedia.org/wiki/Raffaello_D%27Andrea).
- **Thomas Bi**: creator and lead author of CyberRunner (ICRA 2024). The CyberRunner site lists
  contact **bit@ethz.ch** (https://www.cyberrunner.ai/).
- **"Aswin" (UNCONFIRMED, possible match).** *Dr. Aswin Karthik Ramachandran Venkatapathy* ("Aswin
  Ramachandran") is listed as staff of D'Andrea's group at IDSC
  (https://idsc.ethz.ch/research-dandrea/people/person-detail.MzQwMjcz.TGlzdC8zMzc0LDE3MzMwODk3OQ==.html).
  He is named as a **contributor to the CyberRunner GitHub repo** (https://github.com/thomasbi1/cyberrunner)
  and is the last author with Bi and D'Andrea on arXiv 2607.02037 (https://arxiv.org/abs/2607.02037).
  In **this repo**, `camera_calibration_realtime.py` has the header `# Author: Aswin Ramachandran`,
  and `todo.txt` mentions "Aswin's clean sound-based calibration file". **I have not confirmed that
  this is the "Aswin" the user means or what his role would be. Please confirm with the user.**
- Also, `todo.txt` mentions "Thomas's state estimation", which suggests the state-estimation code
  derives from Thomas Bi's CyberRunner code. That matters for licensing (§3.4).
- The fork chain is `kro0l1k/cyberrunner_basics_no_ros` → `dimikoumou/cyberrunner_basics_no_ros`.
  The upstream owner's identity and role are unknown.

### 3.2 How student research typically gets published there

- ETH Master's students do one **semester project** (~14 weeks, ~300–400 h) and a **Master's thesis**.
  IDSC supervisors define projects and students email the supervisor they want
  (https://idsc.ethz.ch/education/theses-semester-projects.html;
  https://ee.ethz.ch/studies/master-s-programmes/main-master/projects-and-master-thesis.html).
  If the user is **not** formally enrolled in such a project, the easiest path is to turn this work
  into one (or a research-assistant arrangement). Then supervision, credit and authorship are
  clear from the start.
- **Typical pattern in this group:** the student is first author, the doctoral/postdoc supervisor
  is middle author, and the professor is last or near-last. Example: Gaber, Bi & D'Andrea, CDC 2024
  (https://arxiv.org/abs/2406.08650). The recent ASV paper lists Jiang, Bi, D'Andrea, Ramachandran
  (https://arxiv.org/abs/2607.02037). Author order varies by project, so **don't assume**.
- **ETH authorship rules:** authorship and author order must be discussed *as early as possible*
  with everyone involved, revisited when roles change, and contributions declared transparently
  (ETH Integrity Guidelines, RSETHZ 414: https://ethz.ch/content/dam/ethz/main/eth-zurich/organisation/rechtssammlung/414en.pdf).
- **Thesis in the ETH Research Collection** needs a declaration of consent signed by the student
  **and** the supervising professor
  (https://unlimited.ethz.ch/spaces/RC/pages/194119870/Master+Bachelor+and+Semester+Theses+Student+Papers).
- **What the group will likely expect:** real-hardware results with repeated trials and clear metrics,
  a video, honest baselines, and a link to their line of work (CyberRunner; zero-shot RL from simple
  models, as in the ASV paper). This is an inference from their papers, not a stated policy.

### 3.3 How to approach them

1. One short email (to the supervisor you already have; otherwise Thomas Bi, and Aswin if confirmed)
   with a **one-page summary**, **one 60–90 s video** (click-to-target, hold, RL vs classic,
   draw-path), and a link to the report. Ask directly: *"Is this worth writing up with you, and if
   so, as a semester project/thesis or a paper, and at which venue?"*
2. Propose framing A with a timeline (§4), and say you're open to their choice of venue and scope.
3. Ask about **authorship and order** at the first meeting, as the ETH guidelines require.
4. Ask about **licensing** of the code derived from CyberRunner and the fork (§3.4), and whether
   the group wants to own or host the release.
5. Don't post on arXiv, social media or YouTube before they agree.

### 3.4 IP and licensing

- **CyberRunner software is AGPL-3.0**, and the repo contains the code and documentation to build
  the robot (https://github.com/thomasbi1/cyberrunner). The CyberRunner website points there for
  "Software/Hardware" (https://www.cyberrunner.ai/). I found **no separate hardware licence** stated.
  Ask Thomas Bi whether the hardware files come under the same terms.
- **AGPL implication:** if our code is a derivative of CyberRunner code (e.g. the state
  estimation), a public release must be under AGPL-3.0-compatible terms with source. Publishing a
  paper does not trigger this; releasing or deploying the code does.
- **The immediate upstream fork `kro0l1k/cyberrunner_basics_no_ros` shows no licence** (checked on
  GitHub). Without a licence, default copyright applies. **Get the owner's permission or a licence
  before releasing the code publicly.** This repo has no LICENSE file either.
- **Student IP at ETH:** IP created by unemployed Bachelor's/Master's students alone belongs to the
  students. If ETH staff (e.g. supervisors) are involved, ETH is entitled to joint ownership of an
  invention (https://ethz.ch/en/industry/researchers/contracts/aspects/ip.html;
  Exploitation Guidelines: https://ethz.ch/content/dam/ethz/main/eth-zurich/organisation/rechtssammlung/440.4en.pdf).
  There's nothing patent-worthy here, but agree on code/data ownership and release early.
- **Citation:** the CyberRunner repo asks for Bi & D'Andrea, ICRA 2024 to be cited. Cite it as the platform.

### 3.5 Code and data release expectations

A good default, subject to the group's decision: release (a) the simulator plus the calibration
script and fitted parameters, (b) anonymised rig logs used for calibration and evaluation, (c) the
trained policies (`policy.npz`) and ODIL code, (d) evaluation scripts that reproduce every
table/figure, (e) a video. Remove or ignore local learned files and tidy the dead code listed in
the report §8.4. Choose the licence together with the group, given AGPL and the unlicensed fork.

---

## 4. Concrete plan

### 4.1 Evidence to add before submitting

**Protocol (fix it before collecting data and write it down: this is your pre-registration):**
- Target sets: **3 radii** (8, 12, 20 mm) × **fixed random start/target pairs** (e.g. 30 pairs per
  radius, same seeds for all methods), all ≥ 5 cm from the frame. Optionally report a separate
  near-frame set as a known failure regime.
- Methods: **Classic**, **RL v3 (or v4)**, **Hybrid**, **ODIL**, plus a **plain PD/PID baseline**
  (what a textbook ball-and-plate would do). Optionally add an **adaptive/history RL** baseline.
- **Randomise the method order** within each session. Note time of day and paper sheet, and let
  stiction "settle" in a consistent way (the ball sinks in over time: control the dwell before each
  trip, e.g. 5 s).
- Log everything: the raw vision state, the commanded and measured tilt, and timestamps.

**Metrics (per trip):** success (reach within r and stay ≥ T s); time-to-arrival; time-to-settle;
final and median rest error (mm); % time in target after arrival; number of "moves"/hops (speed
zero-crossings); overshoot (mm); action jerk/total variation; exits per minute; failures (never
arrived, wall hit). For holds: duration distribution over **≥ 10 holds** per method.

**Statistics:** report means/medians with **95 % bootstrap CIs**. Paired comparisons on the same
start/target pairs: **Wilcoxon signed-rank** for continuous metrics, **McNemar** for success.
Apply a Holm correction across metrics. N ≈ 30 per cell detects medium effects. Report the effect
sizes as well.

**Ablations (the ones reviewers will ask for):**
- Classic: without delay-aligned servo / without pulse+brake / without slope map / without planned
  trapezoid / without friction feed-forward.
- RL: uncalibrated vs log-calibrated simulator; no domain randomisation; no action history; no
  rate limit (show the bang-bang/dither failure); optionally with vs without stiction in the
  simulator.
- ODIL: with vs without the smoothness term; model with vs without a stiction surrogate; ODIL
  alone vs ODIL + classic settle.
- **Compute/sample budget:** wall-clock and environment steps for PPO vs ODIL. This is the headline
  comparison of the ODIL paper, so reproduce it here.

**Characterisation (cheap, and it makes the paper):**
- Breakaway angle vs dwell time (0.5 / 2 / 10 / 60 s) at several locations; rolling friction vs
  speed; a spatial breakaway/slope map with its repeatability across days and sheets; paper vs a
  hard surface (glass or plastic sheet) as a control condition.
- Simulator fidelity: open-loop replay error vs horizon, on held-out logs, for each model variant.

**Video:** a 2–3 min supplementary video: the task, the stiction phenomenon (slow motion), each
method side by side on the same trip, the UI demos, and failure cases.

### 4.2 Paper outline (8 pages, framing A)

1. **Introduction.** Rest-to-rest positioning under stiction and delay; why a compliant surface is
   hard; contributions (characterisation, friction-aware classical controller, log-calibrated
   zero-shot RL, first hardware ODIL, head-to-head comparison, open release).
2. **Related work.** Ball-and-plate control; friction/stiction compensation; sim-to-real on ball
   balancers (Muratore et al., Moab); learning control on ball-and-plate (Amadio et al., Köpf et al.);
   differentiable-simulation policies; ODIL; CyberRunner (Bi & D'Andrea; Gaber et al.).
3. **Platform and problem.** CyberRunner hardware, the paper surface, vision pipeline, delays
   (~0.15 s at 30 fps), the closed-loop tilt servo; the task definition and metrics.
4. **Characterising the stiction-dominated regime.** Breakaway vs rolling, the spatial map, dwell dependence.
5. **Methods.** 5.1 Friction-aware classical control. 5.2 Log-calibrated simulator + goal-conditioned
   PPO (action history, rate limit, domain randomisation). 5.3 ODIL policy through a differentiable
   rig-fitted model. 5.4 Hybrid.
6. **Experiments.** Protocol, results table with CIs, ablations, compute/sample budget, simulator fidelity.
7. **Discussion.** Where each method wins (RL/ODIL: fast one-shot approach; classic: last-millimetre
   precision via slope memory and pulses); limits (near-frame zone, camera).
8. **Conclusion and release.**

Acknowledgments should include the **AI-assistance disclosure** required by IEEE policy.

### 4.3 Timeline (from 2026-09-27)

| When | What |
|---|---|
| **Week 1 (by 4 Oct)** | Email the supervisors/group (§3.3) with the one-pager and video. Confirm who "Aswin" is. Ask about authorship, venue and licence. Get ODIL running in sim (the log is currently empty). |
| Weeks 2–3 | Fix the protocol. Run the characterisation experiments. Build the evaluation harness (fixed seeds, automatic metrics, bootstrap/Wilcoxon script). |
| Weeks 3–5 | Collect the main comparison: classic / RL / hybrid / PD (~30 trips × 3 radii each). Run the RL and classic ablations. |
| **Decision point, ~24 Oct** | If the group wants an early paper: write B(+C) and submit to **ECC (31 Oct)** or, better, **L4DC (13 Nov)**. Otherwise keep going. |
| Nov–Dec | ODIL on the rig; ODIL ablations; compute/sample budget comparison; stiction-surrogate variant. |
| Jan 2027 | Full re-run of all methods on the same day(s) for the final table; film the video; release the code and data (licence cleared). |
| Feb 2027 | Write the paper; internal review with the group; post to arXiv only once they agree. |
| **1 Mar 2027** | Submit framing A to **IROS 2027** or **RA-L with the IROS option**. Alternative: CDC/L-CSS (~mid-March if it follows the 2025–26 pattern). |

### 4.4 Next five steps

1. **Write to the group first:** one-page summary + 90 s video. Confirm "Aswin" and each person's
   role. Ask about authorship, venue, and whether this becomes a semester project/thesis.
2. **Get ODIL working end-to-end in simulation** (the log is empty), compare it with PPO on
   wall-clock and success, and then test it on the rig. This decides between framing A and B.
3. **Freeze an evaluation protocol and harness:** fixed seeded start/target sets, 3 radii, automatic
   metrics and stats script (bootstrap CIs, Wilcoxon, McNemar).
4. **Run the stiction characterisation** (breakaway vs dwell, spatial map, paper vs hard surface)
   and the ≥ 30-trial head-to-head for classic / RL / hybrid / PD.
5. **Clear the licensing** (AGPL CyberRunner, the unlicensed upstream fork) and prepare the release:
   code, logs, simulator, policies, video.
