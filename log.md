# Task log

## 2026-11-06

### Planned work
- [x] Replace the inclusion classifier with Random Forest; compare inclusion, ranking, and feature importance with the existing algorithms.
- [ ] Make Coptic input case-insensitive: `Ϩⲟⲟⲩ` and `ϩⲟⲟⲩ` should return the same entries.
- [ ] Count Crum citations and aligned examples. Analyze sentence-length distributions, means, and Z-scores within each entry: are earlier examples shorter, and does Crum avoid long examples? Distinguish printed citation fragments from full corpus sentences.
- [ ] Compare length-based adjustments with a simple cutoff above 30. Define the counting unit: characters, bound groups, or TT tokens. The adjustment formula and the note `+0.3` remain to be clarified. Regression case: the excessively long Proverbs 9:12 example.
- [ ] Audit C4327 (“be amazed, be stupefied”) by dialect and sense. Acts 20:14 (“met us”) is an incorrect recommendation, also found among the original site's automatic examples. Check Sahidic and Bohairic separately; verify the rare Sahidic usage. Count Crum entries/examples marked Sahidic and “rare,” including the scope of each qualifier.
- [ ] Create a regression vocabulary Markdown file: `Word | Sense 1 | Sense 2 | …`. Start with father/barley (C992/C998), ϩⲟⲟⲩ, ⲧⲱⲙⲧ, ⲉϩⲉ, ⲡⲉ, and ⲉ. Add unit tests covering straightforward contrasts and current errors.
- [ ] Compare algorithm scores; report classification and ranking metrics separately.

### Future work

- Example collections and clustering: select sets covering form, meaning, and syntax.
- Develop expert annotation guidelines for inclusion, sense, target tokens, ordering, and set coverage.
- Multilingual semantic features: use CDO's English, German, and French glosses; test German/French cosine similarities and their averages or combinations, starting with C4324. Assess German/French Bible parallels and machine translation for non-Bible texts.
