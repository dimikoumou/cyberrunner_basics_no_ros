#!/usr/bin/env python3
"""Assemble the visual guide PDF from fig/ and eq/ (run make_figures.py and make_equations.py first)."""
import json
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)
from PIL import Image as PILImage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG, EQ = os.path.join(HERE, "fig"), os.path.join(HERE, "eq")
OUT = os.path.join(HERE, "CyberRunner_ODIL_RL_visual_guide.pdf")
W = A4[0] - 36 * mm

ss = getSampleStyleSheet()
INK = colors.HexColor("#1f2328")
MUTED = colors.HexColor("#6b7280")
BLUE = colors.HexColor("#2563eb")
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=19, textColor=INK, spaceBefore=4, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=13.5, textColor=BLUE, spaceBefore=10, spaceAfter=5)
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontSize=11, textColor=INK, spaceBefore=6, spaceAfter=3)
P = ParagraphStyle("P", parent=ss["BodyText"], fontSize=10, leading=14.2, textColor=INK, spaceAfter=5, alignment=TA_LEFT)
CAP = ParagraphStyle("CAP", parent=P, fontSize=8.6, leading=11.5, textColor=MUTED, spaceAfter=10)
BUL = ParagraphStyle("BUL", parent=P, leftIndent=12, bulletIndent=2, spaceAfter=2)
LEAD = ParagraphStyle("LEAD", parent=P, fontSize=11.5, leading=16)


def img(path, width=W, max_h=150 * mm):
    w, h = PILImage.open(path).size
    width = min(width, max_h * w / h)
    return Image(path, width=width, height=width * h / w)


def fig(name, caption, width=W):
    return KeepTogether([img(os.path.join(FIG, name), width), Paragraph(caption, CAP)])


def eq(name, explain, scale=1.0):
    path = os.path.join(EQ, name + ".png")
    w, h = PILImage.open(path).size
    width = min(W, w / 220 * 72 * 0.92 * scale)          # ~natural size, never wider than the column
    return KeepTogether([Image(path, width=width, height=width * h / w), Paragraph(explain, CAP)])


def bullets(items):
    return [Paragraph(t, BUL, bulletText="•") for t in items]


CELL = ParagraphStyle("CELL", fontName="Helvetica", fontSize=8.6, leading=10.5, textColor=INK)
CELLB = ParagraphStyle("CELLB", parent=CELL, fontName="Helvetica-Bold")


def table(rows, widths, head=True):
    rows = [[Paragraph(str(c), CELLB if (head and i == 0) else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(rows, colWidths=widths)
    spans = [i for i, r in enumerate(rows) if i > 0 and all(not c.getPlainText() for c in r[1:])]
    st = [("SPAN", (0, i), (-1, i)) for i in spans] + [("FONT", (0, 0), (-1, -1), "Helvetica", 8.8), ("TEXTCOLOR", (0, 0), (-1, -1), INK),
          ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d1d5db")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if head:
        st += [("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.8), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6"))]
    t.setStyle(TableStyle(st))
    return t


def dry_results():
    fn = os.path.join(ROOT, "phase3_logs", "rig_learn.jsonl")
    rows = [["Run (dry test)", "Rig data", "Reached", "Time to reach", "Inside after", "Final dist.", "Jerk"]]
    if os.path.exists(fn):
        recs = [json.loads(l) for l in open(fn)]
        starts = [i for i, r in enumerate(recs) if r.get("event") == "start" and r.get("dry")]
        last = recs[starts[-1]:] if starts else []
        for r in last:
            if r.get("event") == "shutdown" and "motor" in r.get("why", ""):
                rows.append([f"PPO: the motor watchdog stopped the run ({r['why']}) -- the safety stop worked; cap tightened to 1000 ticks for the real runs", "", "", "", "", "", ""])
            if r.get("event") == "test":
                rows.append([r["method"], f"{r['rig_minutes']:.0f} min", f"{r['reached']}/{r['n']}",
                             f"{r['t_reach_med']:.1f} s" if r.get("t_reach_med") else "-",
                             f"{100 * r['inside_after_mean']:.0f} %", f"{r['final_mm_med']:.0f} mm" if r.get("final_mm_med") else "-",
                             f"{r['jerk_med']:.4f}" if r.get("jerk_med") is not None else "-"])
    return rows


def session_results():
    """the tests of the last real (non-dry) session at 12 mm"""
    fn = os.path.join(ROOT, "phase3_logs", "rig_learn.jsonl")
    recs = [json.loads(l) for l in open(fn)]
    starts = [i for i, r in enumerate(recs) if r.get("event") == "start" and not r.get("dry")]
    refs = [r for r in recs[:starts[-1]] if r.get("event") == "test" and r.get("run") == "ref"][-2:]
    last = refs + recs[starts[-1]:]       # session 1e did not repeat the references of the start just before
    names = {"ref_odil_v11_sim": "ODIL v11, sim-trained (ref.)", "ref_ppo_v3_sim": "PPO v3, sim-trained (ref.)",
             "odil": "ODIL, rig only", "sac": "SAC, rig only", "ppo": "PPO, rig only"}
    rows = [["Controller", "Rig data", "Reached", "Time to reach", "Inside after", "Final dist.", "Jerk"]]
    for r in last:
        if r.get("event") == "test" and r.get("radius_mm", 12.0) == 12.0:
            rows.append([names.get(r["method"], r["method"]), f"{r['rig_minutes']:.0f} min", f"{r['reached']}/{r['n']}",
                         f"{r['t_reach_med']:.1f} s" if r.get("t_reach_med") else "-",
                         f"{100 * r['inside_after_mean']:.0f} %", f"{r['final_mm_med']:.1f} mm",
                         f"{r['jerk_med']:.4f}"])
    return rows


def page_deco(c, doc):
    c.saveState()
    c.setFont("Helvetica", 7.5)
    c.setFillColor(MUTED)
    c.drawString(18 * mm, 10 * mm, "CyberRunner -- ODIL vs RL, a visual guide")
    c.drawRightString(A4[0] - 18 * mm, 10 * mm, f"{doc.page}")
    c.restoreState()


def build():
    s = []
    # ---- title ----
    s += [Spacer(1, 30 * mm), Paragraph("CyberRunner: ODIL vs reinforcement learning", ParagraphStyle(
        "T", parent=H1, fontSize=26, leading=31)),
        Paragraph("A visual guide to the methods, the equations and the experiments", ParagraphStyle(
            "ST", parent=P, fontSize=14, textColor=MUTED, leading=18)), Spacer(1, 8 * mm),
        Paragraph("Ball-on-plate and the labyrinth on a real tilting-plate robot. What each method is, exactly "
                  "how it learns, what it has achieved so far, and the experiment now running: training ODIL, "
                  "SAC and PPO from real rig data only.", LEAD), Spacer(1, 6 * mm),
        table([["Part", "Question", "Status"],
               ["1. Ball on plate, sim -> rig", "Train in a simulator, run on the real plate", "done: ODIL = RL accuracy, 2.5x smoother"],
               ["2. Maze", "Follow the printed route of the real board", "done: real maze solved 3 times"],
               ["3. Ball on plate, rig only", "How much real rig time does each method need?", "session 1 done for ODIL and SAC; PPO next"]],
              [45 * mm, 70 * mm, W - 115 * mm]),
        Spacer(1, 10 * mm), img(os.path.join(ROOT, "maze", "report_media", "first_finish_070pct_of_video.jpg"), W * 0.8),
        Paragraph("The rig's live view during the first complete run of the real labyrinth (route already done in green).", CAP),
        PageBreak()]

    # ---- 1 the rig ----
    s += [Paragraph("1. The rig and the control loop", H1),
          Paragraph("A camera looks down on a plate that two motors tilt. Thirty times a second the software finds the "
                    "plate and the ball, decides how to tilt, and moves the motors. Every method in this guide only "
                    "changes step 4: the controller. Everything else is shared.", P),
          fig("01_loop.png", "Figure 1. The control loop. Steps 1-3 and 5-6 are identical for all methods."),
          Paragraph("Measuring the state", H2),
          Paragraph("Four markers on the plate give its tilt; the ball's colour gives its pixel, which is back-projected "
                    "onto the plate plane in millimetres. Speed comes from two consecutive photos:", P),
          eq("velocity", "Speed from the camera. It is noisy: a resting ball seems to jiggle by up to 18 mm/s."),
          Paragraph("The tilt servo", H2),
          Paragraph("The motors do not hold an angle reliably (the linkage slips), so the camera closes the loop: each "
                    "frame the motor target moves by the change in the wanted tilt plus a correction on the measured "
                    "error, compared with the target from when the frame was taken (D = 4-5 frames ago):", P),
          eq("servo", "m = motor position (ticks), c = ticks per degree, theta* = wanted tilt, theta-hat = camera-measured "
                      "tilt. Safety limits: no push beyond 7 deg, and the motor stays inside a fixed travel cap."),
          CondPageBreak(70 * mm)]

    # ---- 2 physics ----
    s += [CondPageBreak(110 * mm),
          Paragraph("2. The physics every method lives with", H1),
          fig("02_physics.png", "Figure 2. Left: a rolling ball accelerates down the slope. Right: real rig data (dry "
                                "run) -- the camera-measured tilt (blue) follows the command (grey) with a delay and a lag."),
          Paragraph("Rolling", H3),
          eq("roll", "A solid ball rolling without slipping accelerates at 5/7 of the free-sliding value. "
                     "b is a small level error of the plate."),
          Paragraph("Friction and stiction", H3),
          Paragraph("Two friction effects matter at these small angles: rolling friction a<sub>r</sub>, and stiction -- a "
                    "resting ball does not move until the tilt exceeds a breakaway level a<sub>s</sub> (about 1.8 deg). "
                    "Both are written smoothly so that gradients exist:", P),
          eq("friction", "w blends from 'resting' to 'rolling' as the speed grows; h is a soft minimum, so a drive below "
                         "the breakaway is cancelled."),
          eq("friction2", "v<sub>s</sub> = 1 cm/s, kappa sets how sharp the soft minimum is."),
          Paragraph("The plate is late", H3),
          eq("lag", "The plate follows the command after a pure delay of d frames and a first-order lag tau. This delay is "
                    "the main reason fast controllers overshoot."),
          Paragraph("Fitting the physics to rig data (system identification)", H3),
          Paragraph("ODIL needs these numbers for this particular rig. They are measured from recordings by least squares: "
                    "the gain k and rolling friction from measured accelerations, the delay and lag from how the measured "
                    "tilt follows the command:", P),
          eq("sysid", "Fitted on the rig: k ~ 0.08-0.11 m/s<super>2</super> per degree, a<sub>r</sub> ~ 0.03-0.05 m/s<super>2</super>, "
                      "delay 3-4 frames, lag 0.03-0.1 s."),
          CondPageBreak(70 * mm)]

    # ---- 3 ODIL ----
    s += [Paragraph("3. ODIL -- Optimizing a Discrete Loss", H1),
          Paragraph("<b>Idea in one sentence:</b> write the physics as a penalty, make the whole trajectory and every "
                    "command unknowns, and solve for all of them at once by gradient descent; then fit a small network "
                    "to reproduce the commands from the ball's state.", LEAD),
          fig("05_odil_demo.png", "Figure 3. A worked ODIL example (1-D). Start from a wrong guess (grey): every position "
                                  "and every tilt is adjusted together until each step obeys the physics (right: the "
                                  "physics error falls by six orders of magnitude) while the ball goes from rest at 0 to "
                                  "rest at 1."),
          Paragraph("The loss", H2),
          eq("odil", "Term 1: physics residual at every step (midpoint rule). Term 2: tracking the target or reference r. "
                     "Term 3: smooth commands. Term 4: the commands must be reproducible by the policy network pi."),
          eq("odil_gd", "One gradient step updates everything. Thousands of trajectories (random starts, targets, physics "
                        "parameters) are optimised together, so the network learns a general controller."),
          Paragraph("Closed-loop refinement", H2),
          Paragraph("A policy fitted to optimised trajectories never saw its own compounding errors. It is therefore "
                    "refined by back-propagating through closed-loop rollouts of a differentiable model with the rig's "
                    "delay, lag, camera noise and friction:", P),
          eq("refine", "Tracking error (in units of 5 mm), smoothness, and tilt beyond what a perfect follower would need. "
                       "The gradient flows exactly through the physics -- this is what RL has to estimate from samples."),
          Paragraph("The version used on the rig", H2),
          Paragraph("In the rig-only experiment (<font face='Courier'>odil_plate_v9.py</font>, 3 delay stages) the "
                    "commands are not separate unknowns: the policy produces them inside the physics term (so term 4 is "
                    "exact). 768 start-target pairs x 41 time points, durations up to 3 s; the end must be a true rest "
                    "at the target centre; the physics (gain, stiction, level bias, delay) is randomised per trajectory "
                    "around the values fitted from the rig. About 12 minutes on a 4-core laptop, no GPU.", P),
          Paragraph("Stiction compensation (not part of ODIL)", H2),
          Paragraph("A gentle controller cannot always free a resting ball. On the rig a classic add-on helps: if the ball "
                    "has moved less than 1.5 mm in 0.4 s <b>and is farther than the target radius R</b> from the target, "
                    "an extra tilt towards the target ramps up (6 deg/s, at most 2 deg) until the ball moves. It is "
                    "reported separately, and an ablation without it is planned.", P),
          fig("06_method_odil.png", "Figure 4. ODIL trained only from rig data: record, fit the physics, optimise, test; "
                                    "the next round records with the new controller."),
          CondPageBreak(70 * mm)]

    # ---- 4 RL basics + PPO ----
    s += [Paragraph("4. Reinforcement learning: what PPO and SAC share", H1),
          Paragraph("RL never sees the physics. It tries tilts, receives a reward, and gradually prefers actions that led "
                    "to more reward. Both RL methods here use the same observation (target error, velocity, position, "
                    "tilt, target radius, the last 4 commands), the same rate limit and the same reward:", P),
          eq("reward", "Closer is better; a bonus inside the target and more when also still; small costs for jerky and "
                       "large tilts and for touching the frame."),
          eq("return", "RL maximises the discounted sum of future rewards -- it has to discover which early actions "
                       "lead to good outcomes seconds later."),
          Paragraph("PPO -- Proximal Policy Optimization (on-policy)", H2),
          fig("07_method_ppo.png", "Figure 5. PPO drives a batch with the current controller, learns from it once, then "
                                   "discards it."),
          eq("ppo", "rho compares the new and old policy; clipping keeps each update small and stable."),
          eq("gae", "The advantage A-hat says how much better an action was than expected; V is a learned value function."),
          fig("09_ppo_curve.png", "Figure 6. PPO's real learning curve in simulation (v3). Its best version needed 2.5 "
                                  "million steps -- 24 hours of driving if it had been learned on the rig."),
          CondPageBreak(70 * mm)]

    # ---- 5 SAC ----
    s += [Paragraph("5. SAC -- Soft Actor-Critic (off-policy)", H1),
          Paragraph("SAC keeps every step it has ever driven in a memory and learns from it again and again. A critic "
                    "learns how good each tilt is in each situation; the actor (the controller) moves towards what the "
                    "critic rates highly. 'Soft' means it is rewarded for staying a little random, which keeps it "
                    "exploring.", P),
          fig("08_method_sac.png", "Figure 7. SAC: drive, store, and replay old steps many times. Much less driving than "
                                   "PPO for the same learning."),
          eq("sac_obj", "The objective: reward plus an entropy bonus alpha * H (how random the policy still is)."),
          eq("sac_q", "Critic: match the reward plus the discounted value of the next state; two critics, take the "
                      "smaller (against over-optimism); Q-bar are slowly updated target copies."),
          eq("sac_pi", "Actor: choose actions the critic rates highly while staying somewhat random."),
          Paragraph("On the rig, SAC's network updates happen between 15-second episodes with the plate held level, so "
                    "learning never slows the 29 Hz control loop.", P),
          Paragraph("The three methods in one line each", H2),
          table([["", "Learns from", "Needs a model", "Experience reuse", "Rig time (expected)"],
                 ["ODIL", "physics fitted to rig data", "yes (fitted)", "-- (optimises through it)", "~30 min per round (measured: 60 min)"],
                 ["SAC", "rewards on real attempts", "no", "replays everything", "~2-5 h (measured: ~2 h)"],
                 ["PPO", "rewards on real attempts", "no", "uses each batch once", "~10-30 h"]],
                [16 * mm, 45 * mm, 25 * mm, 40 * mm, W - 126 * mm]),
          CondPageBreak(70 * mm)]

    # ---- 6 trajectories + results ----
    s += [Paragraph("6. What the controllers actually do", H1),
          fig("03_trajectories.png", "Figure 8. Simulated paths of the trained controllers from the same four starts to "
                                     "the same targets. Classic hesitates and stops short; PPO arrives fast but loops "
                                     "and corrects in bursts; ODIL arrives smoothly but can overshoot once (lower "
                                     "path)."),
          fig("04_distance_tilt.png", "Figure 9. One start in detail. Left: distance to the target (dashed = target edge); "
                                      "in this run ODIL later drifts out again -- the stiction problem that the friction "
                                      "compensation addresses. Right: the tilt commands -- PPO's are visibly jerky, "
                                      "ODIL's smooth."),
          CondPageBreak(95 * mm),
          Paragraph("Results so far (part 1)", H2),
          fig("10_experience_rig.png", "Figure 10. Left: experience each method learned from in simulation. Right: on the "
                                       "real rig (120 random targets, 30 per setup) ODIL is as accurate as RL and about "
                                       "2.5 times smoother."),
          eq("jerk", "The two headline metrics. 'Inside' = share of the 4 s after arrival spent inside the target."),
          CondPageBreak(70 * mm)]

    # ---- 7 the experiment ----
    s += [Paragraph("7. The experiment now: training from real rig data only", H1),
          Paragraph("No hand-made simulator: every method learns only from what the real rig shows it. One process holds "
                    "the rig for a whole session (restarts froze the camera), and every controller is tested on the same "
                    "30 fixed targets, logged with the rig minutes it has used so far. Result: learning curves of "
                    "quality against real rig time.", P),
          fig("11_session.png", "Figure 11. One session: references, three ODIL rounds (drive / fit + train / test), then "
                                "10 hours of SAC with a test every 30 minutes."),
          fig("12_random_tilts.png", "Figure 12. ODIL's round-0 data from the dry test: random tilts, no controller. The "
                                     "ball covered the whole plate -- too fast (it was lost ~10 % of frames); the real run "
                                     "uses gentler tilts (max 1.75 deg, braking above 15 cm/s).", W * 0.62),
          CondPageBreak(120 * mm),
          Paragraph("Session 1 results (2026-10-03)", H2),
          fig("13_learning_curves_session1.png", "Figure 13. Learning on the physical rig only, same 30 targets (12 mm). "
                                                 "Dashed/dotted: the simulation-trained references."),
          table(session_results(), [46 * mm, 17 * mm, 19 * mm, 22 * mm, 20 * mm, 20 * mm, W - 144 * mm]),
          Spacer(1, 3 * mm)]
    s += bullets(["<b>Rig time to a good controller:</b> ODIL 60 min, SAC about 120 min (28/30 each). Both rig-only "
                  "ODIL rounds after the first match or beat the simulation-trained ODIL.",
                  "<b>Smoothness:</b> ODIL's jerk is about 4x lower at every point -- its clearest advantage.",
                  "<b>Speed and precision:</b> SAC reaches targets about twice as fast and ends closer to the centre "
                  "(below 5 mm in 17-40 % of trips from 120 min, ODIL 3-23 %).",
                  "<b>SAC over a long run:</b> best at 120-210 min; since then holding has slowly slipped (inside after "
                  "92 % -> ~75 %, final 5.5 -> ~9 mm). The full 600 minutes will show whether this is a real decline."])
    s += [Paragraph("Why ODIL ends a few millimetres off", H3),
          Paragraph("The rig ODIL aims at the exact centre, but its stiction push only acts while the ball is farther "
                    "than R. With R = 12 mm a ball stuck 8 mm off gets no more push, and ODIL's gentle tilts stay below "
                    "the breakaway angle. SAC's reward pulls to the centre at every step. Every controller takes R as an "
                    "input, so a retest telling all of them R = 5 mm measures precision fairly. (An earlier explanation, "
                    "'ODIL rests anywhere within R/2', applied only to an older version, v7.)", P),
          Paragraph("A surface defect: the taped-over hole", H3),
          Paragraph("In ODIL's first test the ball sat for about 80 s at the plate centre, where the hole is covered by "
                    "tape (probably a slight dip), despite tilts up to 3.4 deg; all 11 failures came from that spot, and "
                    "afterwards ODIL reached 15 of 19 targets. A fitted global physics model cannot represent such a "
                    "local defect, while RL can learn it implicitly -- a fair, documented limitation.", P),
          Paragraph("Next (queued, runs automatically)", H2),
          table([["Step", "What", "Rig time", "Why"],
                 ["1", "SAC to 600 rig minutes", "~4 h", "complete curve, long-run stability"],
                 ["2", "References, then PPO from scratch", "up to 30 h", "the on-policy curve"],
                 ["3", "5 mm retest of every saved controller", "~3 h", "precision when every controller is told 5 mm"],
                 ["4", "Reuse test: ODIL path tracker trained only from the 90 min of ODIL rig data (median 4.5 mm in "
                       "the fitted model)", "0 to learn, ~15 min to test", "learn the physics once, solve new tasks"],
                 ["5", "SAC learning path tracking on the rig", "measured", "the RL cost of a new task"],
                 ["later", "Repeat sessions (alternating order), ODIL without stiction push, PPO in the fitted model",
                  "~2 days", "statistics, ablation, model vs optimiser"]],
                [12 * mm, 70 * mm, 30 * mm, W - 112 * mm])]
    s += [CondPageBreak(70 * mm)]

    # ---- 8 maze ----
    s += [PageBreak(), Paragraph("8. Part 2 in pictures: the real labyrinth", H1),
          Paragraph("The ODIL path tracker follows the board's printed route at 2.5 cm/s. The route, the holes and a wall "
                    "map are extracted from a photo of the board; a planner can also find the safest line through the "
                    "corridors. The rig completed the real maze 3 times (first: 172 s); a typical run reaches about a "
                    "quarter of the route, and most falls come from arriving at bends 2-3x too fast.", P),
          img(os.path.join(ROOT, "maze", "report_media", "first_finish_099pct_of_video.jpg"), W * 0.8),
          Paragraph("Figure 14. End of the first complete run: the whole route green, the ball at the finish.", CAP),
          img(os.path.join(ROOT, "maze", "report_media", "route_map_printed_vs_safe.png"), W * 0.8),
          Paragraph("Figure 15. Walls (grey), holes (red), the printed route (green) and the planned safe route (blue).", CAP)]

    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=18 * mm, title="CyberRunner: ODIL vs RL -- visual guide", author="dimikoumou")
    doc.build(s, onFirstPage=page_deco, onLaterPages=page_deco)
    print(OUT)


if __name__ == "__main__":
    build()
