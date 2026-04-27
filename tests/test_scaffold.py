import unittest


class ScaffoldTests(unittest.TestCase):
    def test_package_imports(self) -> None:
        import pitcher_twin

        self.assertIn("models", pitcher_twin.__all__)
