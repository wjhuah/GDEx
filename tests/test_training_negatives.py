import unittest

from training_negatives import add_negative_examples


def row(tla: str, lemma: str, sentence: str, source: str = "crum") -> dict[str, str]:
    return {
        "TLA": tla,
        "lemma": lemma,
        "sent": sentence,
        "translation": f"Translation {sentence}",
        "source": source,
        "ref": "ref",
        "crum_gloss": f"gloss for {tla}",
        "label": "2.0",
        "forbidden": "0",
    }


class NegativeSamplingTests(unittest.TestCase):
    def test_other_tla_crum_sentence_becomes_low_label_hard_negative(self) -> None:
        rows = [
            row("C1", "ⲓⲱⲧ", "sentence from C1"),
            row("C2", "ⲓⲱⲧ", "sentence from C2"),
        ]
        output, stats = add_negative_examples(rows, random_ratio=0)
        hard = [example for example in output if example["source"].startswith("hard_negative:")]
        self.assertEqual(len(hard), 2)
        self.assertEqual({example["label"] for example in hard}, {"0.0"})
        self.assertEqual({example["forbidden"] for example in hard}, {"0"})
        self.assertEqual(stats["hard_negatives"], 2)

    def test_random_negatives_never_exceed_ten_percent(self) -> None:
        rows = [row("C1", "ⲓⲱⲧ", f"target {index}") for index in range(20)]
        rows.append(row("C2", "ⲙⲟⲟϣⲉ", "other entry"))
        output, stats = add_negative_examples(rows, random_ratio=0.10, seed=7)
        random_rows = [example for example in output if example["source"].startswith("random_negative:")]
        self.assertLessEqual(len(random_rows), 2)
        self.assertEqual(stats["random_negatives"], len(random_rows))
        with self.assertRaisesRegex(ValueError, "0.10"):
            add_negative_examples(rows, random_ratio=0.11)

    def test_negative_types_can_be_ablated_independently(self) -> None:
        rows = [
            row("C1", "ⲓⲱⲧ", f"target {index}")
            for index in range(20)
        ]
        rows.extend([
            row("C2", "ⲓⲱⲧ", "homograph example"),
            row("C3", "ⲙⲟⲟϣⲉ", "different lemma"),
        ])
        _, hard_stats = add_negative_examples(
            rows, random_ratio=0.10, include_random=False,
        )
        _, random_stats = add_negative_examples(
            rows, random_ratio=0.10, include_hard=False,
        )
        self.assertGreater(hard_stats["hard_negatives"], 0)
        self.assertEqual(hard_stats["random_negatives"], 0)
        self.assertEqual(random_stats["hard_negatives"], 0)
        self.assertGreater(random_stats["random_negatives"], 0)


if __name__ == "__main__":
    unittest.main()
