import unittest

from motion_jepa.evaluation.babel import _COARSE_BABEL_CATEGORIES, _FINE_TO_COARSE, _coarsen


class BabelCoarseningTests(unittest.TestCase):
    def test_fine_to_coarse_covers_every_known_tag(self) -> None:
        known_tags = {
            "walk", "dance", "stretch", "jump", "run", "stand", "head movements",
            "arm movements", "martial art", "hand movements", "step", "bend", "crawl",
            "lie", "raising body part", "play sport", "kick", "circular movement",
            "cartwheel", "lean", "poses", "exercise/training", "squat",
            "interact with/use object", "sideways movement", "hop", "forward movement",
            "leap", "turn", "touching body part", "look", "stand up", "lift something",
            "knee movement", "grasp object", "stances",
        }
        self.assertEqual(set(_FINE_TO_COARSE), known_tags)

    def test_no_tag_listed_in_two_categories(self) -> None:
        seen = set()
        for fines in _COARSE_BABEL_CATEGORIES.values():
            for fine in fines:
                self.assertNotIn(fine, seen, f"{fine!r} listed in multiple coarse categories")
                seen.add(fine)

    def test_coarsen_known_tag(self) -> None:
        self.assertEqual(_coarsen("walk"), "locomotion")
        self.assertEqual(_coarsen("dance"), "dance")

    def test_coarsen_unmapped_tag_passes_through(self) -> None:
        self.assertEqual(_coarsen("some_never_seen_tag"), "some_never_seen_tag")


if __name__ == "__main__":
    unittest.main()
