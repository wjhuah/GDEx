"""Render local evaluation artifacts for the currently deployed models."""

import html
import json
from .ui import shell
from .dialects import Dialects


def fmt(value):
    return "—" if value is None else f"{value:.4f}"


def table(headers, rows):
    return (
        '<div class="table-scroll"><table><thead><tr>'
        + "".join("<th>" + html.escape(str(x)) + "</th>" for x in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join("<td>" + str(x) + "</td>" for x in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def experiments(config, dialect):
    Dialects().require(dialect)
    path = config.reports / "learning/comparison.json"
    if not path.exists():
        return shell(
            "<section class=panel><h2>Evaluation</h2><p>Run scripts/compare_learning.py to build the comparison.</p></section>",
            "Evaluation",
        )
    report = json.loads(path.read_text())
    data = next(r for r in report["results"] if r["dialect"] == dialect)
    options = "".join(
        f'<option value="{d}"'
        + (" selected" if d == dialect else "")
        + f">{d}</option>"
        for d in Dialects().data
    )
    body = f'<section class=panel><h2>Inclusion and ordering</h2><form><label>Dialect <select name=dialect>{options}</select></label><button>Apply</button></form><p>Five spelling-family folds. Identical Coptic sentences are removed from training whenever they occur in the held-out fold. Added negatives use training sentences only.</p><p>{data["examples"]:,} entry–sentence examples. Source counts can overlap: {html.escape(str(data["sources"]))}.</p></section>'
    body += (
        "<section class=panel><h3>Labels</h3>"
        + table(
            ["Source", "Inclusion label", "Preference target"],
            [
                ["TAB selected", "+1", "3 → 1, using expert priority"],
                ["Crum selected", "+1", "3 → 1.5, using citation order"],
                ["TAB forbidden / ALL", "−1", "−1 in the unified model"],
                ["Generated cross-sense / unrelated", "−1", "−1 in the unified model"],
            ],
        )
        + "<p>Unranked TAB selections supervise inclusion. The unified model gives them a lower bound of 1; their relative order stays unspecified. Binary training maps −1 to class 0.</p></section>"
    )
    threshold_path = config.reports / "learning/thresholds.json"
    if threshold_path.exists():
        decision = json.loads(threshold_path.read_text())
        rows = []
        for item in decision["results"]:
            if item["dialect"] != dialect:
                continue
            r = item["selected"]
            rows.append(
                [
                    item["architecture"],
                    r["threshold"],
                    fmt(r["macro_f1"]),
                    fmt(r["precision"]),
                    fmt(r["recall"]),
                ]
            )
        body += (
            "<section class=panel><h3>Deployed inclusion operating point</h3>"
            + table(
                [
                    "Architecture",
                    "Score threshold",
                    "Macro F1",
                    "Selected-class precision",
                    "Selected-class recall",
                ],
                rows,
            )
            + "<p>The search threshold is selected on development predictions. The augmentation tables below keep threshold 0.5 fixed for comparison. All selected TAB/Crum examples retain their source status. Joint thresholds use a sigmoid transform of the continuous joint score.</p></section>"
        )
    ablation_path = config.reports / "learning/ablations.json"
    if ablation_path.exists():
        a = next(
            r
            for r in json.loads(ablation_path.read_text())["results"]
            if r["dialect"] == dialect
        )
        body += (
            "<section class=panel><h3>Feature ablations</h3><p>Each model is retrained after masking the named feature block. Threshold 0.5; same folds and augmentation setting as the full model.</p>"
            + table(
                [
                    "Architecture",
                    "Removed block",
                    "AUC",
                    "Macro F1",
                    "TAB NDCG@6",
                    "Crum NDCG@6",
                ],
                [
                    [
                        r["architecture"],
                        r["setup"],
                        fmt(r["auc"]),
                        fmt(r["macro_f1"]),
                        fmt(r["ranking"]["tab"]),
                        fmt(r["ranking"]["crum"]),
                    ]
                    for r in a["summaries"]
                ],
            )
            + "</section>"
        )
    names = {"two_models": "Inclusion + ranking", "joint": "Unified model"}
    for arch in names:
        rows = []
        for r in data["summaries"]:
            if r["architecture"] != arch:
                continue
            counts = [
                c["added"] for c in data["counts"] if c["setting"] == r["setting"]
            ]
            chosen = (
                " · selected"
                if data["choices"][arch]["setting"] == r["setting"]
                else ""
            )
            rows.append(
                [
                    r["setting"] + chosen,
                    f"{min(counts):,}–{max(counts):,}",
                    fmt(r["auc"]),
                    fmt(r["macro_f1"]),
                    fmt(r["balanced_accuracy"]),
                    fmt(r["ranking"]["tab"]["ndcg_at_6"]),
                    fmt(r["ranking"]["crum"]["ndcg_at_6"]),
                ]
            )
        choice = data["choices"][arch]
        body += (
            "<section class=panel><h3>"
            + names[arch]
            + "</h3>"
            + table(
                [
                    "Added negatives per positive",
                    "Added per training fold",
                    "AUC ↑",
                    "Macro F1 ↑",
                    "Balanced accuracy ↑",
                    "TAB positive-order NDCG@6 ↑",
                    "Crum positive-order NDCG@6 ↑",
                ],
                rows,
            )
            + f'<p>Final training: {choice["training"]:,} original pairs + {choice["added"]:,} generated negatives ({html.escape(str(choice["types"]))}).</p></section>'
        )
    body += "<section class=panel><h3>Reading the results</h3><p>AUC measures how well inclusion scores separate selected and forbidden pairs across thresholds. Macro F1 gives positive and negative classes equal importance at the deployed threshold. Balanced accuracy averages positive recall and negative recall.</p><p>NDCG measures order among the held-out selected examples, separately for TAB and Crum. The two-model ranker uses positive preferences only, so its ordering results stay fixed across negative augmentation settings. The unified model uses the continuous score for both tasks.</p><p>hard: shared target tokens, different TLA meanings. unrelated: no aligned target token. mixed: both sources. The suffix requests extra negatives per positive; source availability determines actual counts. Known positive pairs and sentences containing both explicit gloss meanings are excluded from generated negatives.</p><p>Selection uses these same five folds; these are model-development estimates. A fresh expert test set is the next independent check. ALL rules contribute many repeated sentences across entries; the fold counts below show the resulting training reductions.</p></section>"
    body += (
        "<section class=panel><h3>Training fold sizes, before augmentation</h3>"
        + table(
            ["Fold", "Original", "Positive", "Negative"],
            [
                [c["fold"] + 1, c["original"], c["positive"], c["negative"]]
                for c in data["counts"]
                if c["setting"] == "none"
            ],
        )
        + "</section>"
    )
    reference = config.reports / "learning/order_reference.json"
    if reference.exists():
        rows = [r for r in json.loads(reference.read_text()) if r["dialect"] == dialect]
        body += (
            "<section class=panel><h3>Positive-order references</h3><p>Crum grades have a positive floor, so random-order NDCG can be high. Pairwise accuracy compares the order of two source-selected examples; random order has expectation 0.5.</p>"
            + table(
                [
                    "Architecture",
                    "Source",
                    "Groups",
                    "Random-order NDCG@6",
                    "Model pairwise accuracy",
                ],
                [
                    [
                        r["architecture"],
                        r["source"],
                        r["groups"],
                        fmt(r["random_ndcg"]),
                        fmt(r["pairwise_accuracy"]),
                    ]
                    for r in rows
                ],
            )
            + "</section>"
        )
    weight_path = config.reports / "learning/negative_weight_comparison.json"
    if weight_path.exists():
        weights = [
            r for r in json.loads(weight_path.read_text()) if r["dialect"] == dialect
        ]
        body += (
            "<section class=panel><h3>Negative-weight diagnostic</h3><p>All original exclusion labels are retained. Class balancing gives each class half the total training weight. The alternative also gives each entry equal weight within its class. Same folds and augmentation; threshold 0.5.</p>"
            + table(
                [
                    "Architecture",
                    "Class-balanced AUC",
                    "Entry-balanced AUC",
                    "Class-balanced F1",
                    "Entry-balanced F1",
                ],
                [
                    [
                        r["architecture"],
                        fmt(r["class_balanced_auc"]),
                        fmt(r["entry_class_balanced_auc"]),
                        fmt(r["class_balanced_f1"]),
                        fmt(r["entry_class_balanced_f1"]),
                    ]
                    for r in weights
                ],
            )
            + "</section>"
        )
    body += '<section class=panel><h3>Data files and source integrity</h3><p><a href="/reports/learning/DATASET_EXAMPLES.json">Readable training examples with all 62 inputs</a> · <a href="/reports/learning/dataset_checks.json">Dataset counts</a> · <a href="/reports/learning/corpus_integrity.json">Damaged source spans</a></p><p>Dictionary subcategories are preserved. C993 uses the title qualifier in its definition and competing-entry features. Marked short damaged spans paired with long translations are excluded from training pairs and recommendations, with original source evidence retained.</p></section>'
    sense_path = config.reports / "learning/token_senses.json"
    if sense_path.exists():
        token = next(
            r for r in json.loads(sense_path.read_text()) if r["dialect"] == dialect
        )
        held = [
            r for r in token["contrasts"] if {r["tla"], r["rival"]} == {"C992", "C993"}
        ]
        fitted = [
            r
            for r in token.get("training_contrasts", [])
            if {r["tla"], r["rival"]} == {"C992", "C993"}
        ]
        body += (
            "<section class=panel><h3>Token-sense comparison</h3><p>This additional classifier compares entries at individual TT tokens. Base-model tables above score the inclusion and preference heads separately.</p>"
            + table(
                [
                    "Training pairs",
                    "Inputs",
                    "Derived-pair held-out AUC",
                    "Father pair: training",
                    "Father pair: held-out",
                ],
                [
                    [
                        token["training_pairs"],
                        len(token["columns"]),
                        fmt(token["auc"]),
                        str(sum(r["correct"] for r in fitted)) + "/" + str(len(fitted)),
                        str(sum(r["correct"] for r in held)) + "/" + str(len(held)),
                    ]
                ],
            )
            + "<p>Source-derived labels; spelling-family folds with repeated texts removed. Training and held-out results are shown separately.</p></section>"
        )
    ref_path = config.reports / "learning/reference_counts.json"
    if ref_path.exists():
        reference = json.loads(ref_path.read_text())
        body += (
            "<section class=panel><h3>Crum reference ledger</h3>"
            + table(
                ["Marker status", "Count"],
                [[k, v] for k, v in reference["exclusive_marker_counts"].items()],
            )
            + "<p>One marker can link several dialects or corpus versions. Training pairs are then deduplicated by entry and Coptic text.</p></section>"
        )
    for model, values in data["importance"].items():
        body += (
            "<section class=panel><details><summary>"
            + html.escape(model)
            + " · all input importances</summary><p>Gain share sums the training loss improvement at splits. Split counts record how often a feature was used. Correlated inputs can share or replace this contribution.</p>"
            + table(
                ["Feature", "Gain share", "Splits"],
                [
                    [html.escape(r["feature"]), f'{r["gain_share"]:.2%}', r["split"]]
                    for r in values
                ],
            )
            + "</details></section>"
        )
    body += '<section class=panel><p><a href="/reports/learning/REPORT.md">Local guide and findings</a> · <a href="/reports/learning/comparison.json">Raw comparison</a></p></section>'
    return shell(body, "Evaluation")


def sense_page(config):
    path = config.reports / "learning/senses.json"
    if not path.exists():
        return shell(
            "<section class=panel><h2>Sense audit</h2><p>Run scripts/audit_learning_senses.py to build the audit.</p></section>",
            "Sense audit",
        )
    data = json.loads(path.read_text())
    rows = []
    for r in data["inventory"]:
        url = f'/search?query={r["tla"]}&dialect={r["dialect"]}&ranker=research'
        rows.append(
            [
                r["dialect"],
                f'<a href="{url}">{html.escape(r["tla"]+" · "+r["lemma"])}</a>',
                html.escape(r["gloss"]),
                r["candidates"],
                r["positive"],
                r["negative"],
                r["ordered"],
            ]
        )
    body = (
        "<section class=panel><h2>Spelling groups</h2><p>Definitions and dialect forms come from the current Coptic Dictionary Online entry index, matched by TLA ID. Native TT tokens supply the target spans; POS conflicts remain visible.</p>"
        + table(
            [
                "Dialect",
                "Entry",
                "Meaning",
                "TT candidates",
                "Source positives",
                "Source negatives",
                "Ordered positives",
            ],
            rows,
        )
        + "</section>"
    )
    body += (
        "<section class=panel><h3>Held-out inclusion results at reference threshold 0.5</h3><p>Models saw other spelling families during training. These selected/forbidden labels evaluate entry–sentence inclusion. Entries with positive evidence alone need additional expert negative labels for a complete sense test.</p>"
        + table(
            [
                "Dialect",
                "Architecture",
                "Entry",
                "Positive",
                "Negative",
                "Positive recall",
                "Negative recall",
            ],
            [
                [
                    r["dialect"],
                    r["architecture"],
                    r["tla"],
                    r["positive"],
                    r["negative"],
                    fmt(r["positive_recall"]),
                    fmt(r["negative_recall"]),
                ]
                for r in data["heldout"]
            ],
        )
        + "</section>"
    )
    body += (
        "<section class=panel><h3>Cross-entry contrast test</h3><p>Each held-out source-positive sentence is scored against its source TLA and a competing homograph. The source score should be higher. Known dual-positive sentences are omitted. These are source-derived contrast labels.</p>"
        + table(
            ["Dialect", "Architecture", "Pair", "Correct / tests", "Ties"],
            [
                [
                    r["dialect"],
                    r["architecture"],
                    r["pair"],
                    f'{r["correct"]} / {r["total"]}',
                    r["ties"],
                ]
                for r in data.get("contrasts", [])
            ],
        )
        + "</section>"
    )
    if data.get("failures"):
        body += (
            "<section class=panel><details><summary>Source contrast errors and ties</summary>"
            + table(
                [
                    "Dialect",
                    "Architecture",
                    "Source / rival",
                    "Translation",
                    "Source score",
                    "Rival score",
                ],
                [
                    [
                        r["dialect"],
                        r["architecture"],
                        r["tla"] + " / " + r["rival_tla"],
                        html.escape(r["translation"]),
                        fmt(r["source_score"]),
                        fmt(r["rival_score"]),
                    ]
                    for r in data["failures"]
                ],
            )
            + "</details></section>"
        )
    token_path = config.reports / "learning/token_senses.json"
    if token_path.exists():
        token_rows = []
        for item in json.loads(token_path.read_text()):
            held = [
                r
                for r in item["contrasts"]
                if {r["tla"], r["rival"]} == {"C992", "C993"}
            ]
            trained = [
                r
                for r in item.get("training_contrasts", [])
                if {r["tla"], r["rival"]} == {"C992", "C993"}
            ]
            token_rows.append(
                [
                    item["dialect"],
                    item["training_pairs"],
                    fmt(item["auc"]),
                    str(sum(r["correct"] for r in trained)) + "/" + str(len(trained)),
                    str(sum(r["correct"] for r in held)) + "/" + str(len(held)),
                ]
            )
        body += (
            "<section class=panel><h3>Deployed token comparison</h3><p>The preceding tables evaluate the base models. Search also compares competing entries at each target token using this classifier.</p>"
            + table(
                [
                    "Dialect",
                    "Training pairs",
                    "Held-out AUC",
                    "Father: training",
                    "Father: held-out",
                ],
                token_rows,
            )
            + "</section>"
        )
    body += '<section class=panel><h3>Source correction</h3><p>The 2020 local XML gives both C771 and C772 “ox, cow”. The current CDO index gives C771 “cow” and C772 “ox”. The application now reads the current entry-specific definitions; model features and training have been rebuilt from that snapshot.</p><p><a href="https://coptic-dictionary.org/entry/C771">C771 · cow</a> · <a href="https://coptic-dictionary.org/entry/C772">C772 · ox</a> · <a href="/experiments">Current model comparison</a></p></section>'
    return shell(body, "Sense audit", "sense")
