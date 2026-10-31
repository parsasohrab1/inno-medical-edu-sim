import unittest
import numpy as np
from inno_medical_edu_sim import core


class TestEdu(unittest.TestCase):
    def setUp(self):
        self.items = core.checklist(0)

    def test_perfect_and_empty(self):
        self.assertEqual(core.score_attempt([1] * core.N_ITEMS, self.items)["percent"], 100.0)
        r = core.score_attempt([0] * core.N_ITEMS, self.items)
        self.assertEqual(r["percent"], 0.0); self.assertTrue(r["result"].startswith("FAIL"))

    def test_critical_item_forces_fail_even_with_high_score(self):
        d = [1] * core.N_ITEMS; d[0] = 0
        r = core.score_attempt(d, self.items, critical=(0,))
        self.assertGreater(r["percent"], 80); self.assertIn("critical", r["result"])

    def test_progress_slope_sign(self):
        self.assertGreater(core.progress([40, 50, 60, 70]), 0)
        self.assertLess(core.progress([70, 60, 50]), 0)

    def test_report_text(self):
        items, att = core.simulate(1)
        self.assertIn("Student 3", core.student_report(3, att, items))
