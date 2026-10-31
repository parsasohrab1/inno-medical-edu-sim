"""OSCE scenario bank, checklist scoring and progress reports for medical students.

Lab prototype on SYNTHETIC data only (TRL 4). Supports human examiners; does not replace them.
"""
import numpy as np

N_STUDENTS, N_SESSIONS, N_ITEMS = 80, 4, 12

CRITERIA = [
    ("score_ability_corr", ">=", 0.80, "correlation of checklist score with latent ability"),
    ("kr20", ">=", 0.70, "KR-20 internal consistency of checklist"),
    ("min_item_discrimination", ">=", 0.05, "lowest item-rest point-biserial correlation"),
    ("progress_detected", ">=", 0.70, "students with positive score trend detected as improving"),
]

SCENARIOS = [
    {"id": "chest_pain", "title": "Chest pain in the ER", "critical": [0, 1]},
    {"id": "asthma", "title": "Acute asthma attack", "critical": [0, 2]},
    {"id": "diabetic_foot", "title": "Diabetic foot ulcer", "critical": [0]},
    {"id": "pediatric_fever", "title": "Febrile child", "critical": [0, 3]},
]


def checklist(seed=0):
    rng = np.random.default_rng(seed)
    items = []
    for j in range(N_ITEMS):
        items.append({"item": j, "weight": int(rng.integers(1, 4)), "a": float(rng.uniform(1.0, 2.2)), "b": float(rng.uniform(-1.2, 1.2))})
    return items


def simulate(seed=0):
    rng = np.random.default_rng(seed + 7)
    items = checklist(seed)
    theta0 = rng.normal(0, 1, N_STUDENTS)
    growth = rng.normal(0.2, 0.08, N_STUDENTS)
    att = []
    for s in range(N_STUDENTS):
        for k in range(N_SESSIONS):
            th = theta0[s] + growth[s] * k
            p = [1 / (1 + np.exp(-it["a"] * (th - it["b"]))) for it in items]
            done = (rng.random(N_ITEMS) < np.array(p)).astype(int)
            att.append({"student": s, "session": k, "theta": th, "done": done, "scenario": SCENARIOS[(s + k) % len(SCENARIOS)]["id"]})
    return items, att


def score_attempt(done, items, critical=(0,)):
    w = np.array([it["weight"] for it in items]); pct = 100.0 * float((np.asarray(done) * w).sum() / w.sum())
    missed = [int(c) for c in critical if not done[c]]
    rating = 1 + int(np.digitize(pct, [35, 50, 65, 80]))
    return {"percent": pct, "rating": rating, "critical_missed": missed, "result": "FAIL (critical item missed)" if missed else ("PASS" if pct >= 50 else "FAIL")}


def kr20(M):
    M = np.asarray(M, float); k = M.shape[1]; p = M.mean(0); tot = M.sum(1)
    return float(k / (k - 1) * (1 - (p * (1 - p)).sum() / tot.var(ddof=1)))


def item_rest_corr(M):
    M = np.asarray(M, float); out = []
    for j in range(M.shape[1]):
        rest = np.delete(M, j, axis=1).sum(1)
        out.append(np.corrcoef(M[:, j], rest)[0, 1] if M[:, j].std() > 0 else 0.0)
    return np.array(out)


def progress(percents):
    x = np.arange(len(percents)); slope = np.polyfit(x, percents, 1)[0]
    return float(slope)


def student_report(student, attempts, items):
    ps = [score_attempt(a["done"], items)["percent"] for a in attempts if a["student"] == student]
    sl = progress(ps)
    return "Student %d: sessions=%d, last=%.0f%%, trend=%+.1f points/session (%s)" % (
        student, len(ps), ps[-1], sl, "improving" if sl > 1 else "flat" if sl > -1 else "declining")


def evaluate(seed=0):
    items, att = simulate(seed)
    sc = np.array([score_attempt(a["done"], items)["percent"] for a in att]); th = np.array([a["theta"] for a in att])
    M = np.array([a["done"] for a in att])
    sl = []
    for s in range(N_STUDENTS):
        ps = [x for a, x in zip(att, sc) if a["student"] == s]
        sl.append(progress(ps))
    return {"score_ability_corr": float(np.corrcoef(sc, th)[0, 1]), "kr20": kr20(M),
            "min_item_discrimination": float(item_rest_corr(M).min()), "progress_detected": float((np.array(sl) > 0).mean())}


def generate(seed=0):
    items, att = simulate(seed)
    return {"checklist_items": items,
            "attempts": [dict(student=a["student"], session=a["session"], scenario=a["scenario"], items_done="".join(map(str, a["done"])),
                              percent=round(score_attempt(a["done"], items)["percent"], 1)) for a in att],
            "scenarios": [dict(id=s["id"], title=s["title"], critical_items=",".join(map(str, s["critical"]))) for s in SCENARIOS]}


def demo(seed=0):
    items, att = simulate(seed)
    return student_report(0, att, items) + "\n" + str(score_attempt(att[0]["done"], items))
