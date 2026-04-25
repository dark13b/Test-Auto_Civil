import unittest

from knowledge_base import KnowledgeBase


class KnowledgeBaseTests(unittest.TestCase):
    def test_all_entries_have_sources(self) -> None:
        knowledge_base = KnowledgeBase()
        payload = knowledge_base.as_dict()

        for entries in payload.values():
            for entry in entries:
                self.assertIn("source", entry)
                self.assertTrue(str(entry["source"]).strip())

    def test_prompt_formatter_includes_section_labels_and_sources(self) -> None:
        knowledge_base = KnowledgeBase()

        prompt_text = knowledge_base.format_for_prompt(model_family="LGBMRegressor", feature_area="engineering")

        self.assertIn("Empirical Best Ranges", prompt_text)
        self.assertIn("[source:", prompt_text)


if __name__ == "__main__":
    unittest.main()
