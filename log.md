# Task log

## 2026-11-13


20 spaces, good examlpls might less than average sentences lentgh

apply a penalty (maybe complicated)

tune a penalty based on human annotation on s

50 TLA have different exaomples , output 3 expamples 

next week: cluster idea

ϣⲁ - not a good example actually is prefix

start with form_covered depl_cover... features 

internal in a cluster!!!









#4
Recommendation score 2.354
ⲙⲁⲣⲉⲧⲉϣⲡⲏⲣⲉ ⲙⲡϣⲁ ⲙⲡⲉⲛⲥⲱⲧⲏⲣ ϫⲣⲟ ⲉⲛⲉⲛϣⲁϫⲉ ·

May the precious miracle of our Saviour overcome our(halting) speech

Proclus Homily 13 on Easter in Budge ed. · 2:1

Main clause or fragment · Target has modifier · Nominal modifier

Ranking details

WRONG！

## 2026-11-06

### Planned work
- [x] Replace the inclusion classifier with Random Forest; compare inclusion, ranking, and feature importance with the existing algorithms.

On Sahidic, changing into Random Forest improve `inclusion` module a lot 🎉:

| Classifier | AUC | Macro F1 |
| :--- | :---: | :---: |
| Tree | 0.9376 | 0.7128 |
| RF | **0.9408** | **0.8089** |

- [x] Make Coptic input case-insensitive: `Ϩⲟⲟⲩ` and `ϩⲟⲟⲩ` should return the same entries.

- [x] Count Crum citations and aligned examples. Analyze sentence-length distributions, means, and Z-scores within each entry: are earlier examples shorter, and does Crum avoid long examples? Distinguish printed citation fragments from full corpus sentences.

Sahidic: about 93.44 letters and 15.98 spaces per length. 
Spearman rank correlation coefficient = 0.05 (-1~1) based on letters & 0.0244 based on spaces.
Crum doesn't tend to avoid long sentences.

- [x] Compare length-based adjustments with a simple cutoff above 30. 

Case: the long Proverbs 9:12 example. Solved.

- [x] Audit C4327 (“be amazed, be stupefied”) by dialect and sense. Acts 20:14 (“met us”) is an incorrect recommendation, also found among the CDO's examples. Check Sahidic and Bohairic separately; verify the rare Sahidic usage. Count Crum entries/examples marked Sahidic and “rare,” including the scope of each qualifier.

⟨ⲧⲱⲙⲧ⟩ ⟪B⟫,
⟨-ⲙⲛⲧ⟩ ⟪S⟫ (rare),
⟨ⲧⲟⲙⲧ†⟩ ⟪B⟫

also:

1427.txt

339b

¶ ⟨ⲥⲟⲙⲥ⟩ ⟪S⟫ (rare) ⟪B⟫, ⟨ⲥⲁ.⟩ ⟪Sf⟫⟪F⟫ ⟦vb⟧{annotation: verb} ⟦intr⟧{annotation: intransitive (i.e. verb without immediate object, or one constructed with prep. ⲉ-)}, ⟨gloss: look⟩, ⟨gloss: behold⟩: ⟦Ac 1 9⟧{bible: Acts of the Apostles 1:9} (⟪S⟫ ⟨ⲉⲓⲱⲣⲙ⟩), ⟦1 Cor 3 10⟧{bible: First Epistle to the Corinthians 3:10} (⟪S⟫ ⟨ϭⲱϣⲧ⟩, ⟪F⟫ ⟨ⲥ. ⲉⲃⲁⲗ⟩) ⟨βλέπειν⟩, ⟦Va 57 11⟧{reference: Va} ⟨ἐμβ.⟩, ⟦Ps 12 4⟧{bible: Psalms 12:4} (⟪S⟫ ⟦diff⟧{annotation: different reading, not useful for comparison}) ⟨ἐπιβ.⟩, ⟦Job 7 8⟧{bible: Job 7:8} ⟨περιβ.⟩⟦;⟧ ⟦He 8 5⟧{bible: Epistle to the Hebrews 8:5} ⟪F⟫ (⟪S⟫ ⟨ϭ.⟩, ⟪B⟫ ⟨ⲛⲁⲩ⟩) ⟨ὁρᾶν⟩⟦;⟧ ⟦MG 25 6⟧{reference: MG} ⟨ⲁϥⲥ. ⲁϥⲛⲁⲩ⟩, ⟦C 86 295⟧{reference: C} ⟨ⲉⲧⲁϥⲥ.…ⲁϥⲛⲁⲩ⟩ ⟨ἀτενίζειν⟩⟦;⟧ ⟦Job 23 15⟧{bible: Job 23:15} ⟨κατανοεῖν⟩⟦;⟧ ⟦Lu 11 53⟧{bible: Gospel of St. Luke 11:53} (⟪S⟫ ⟦diff⟧{annotation: different reading, not useful for comparison}) ⟨ἐνέχειν⟩ ⟨تعلّق على⟩⟦;⟧ ⟦MIE 2 360⟧{reference: MIE} ⟨ⲁϥⲥ. ⲁϥⲛⲁⲩ ⲉⲡⲓⲙⲱⲟⲩ⟩ spreading = ⟦BAp 89⟧{reference: B Ap} ⟪S⟫ ⟨ⲑⲉⲱⲣⲉⲓ ⲙⲡⲙ.⟩, ⟦TstAb 238⟧{reference: Tst Ab} ⟨ⲥ. ⲛⲛⲉⲕⲃⲁⲗ⟩ & see.

1595.txt

414b

¶ ⟨ⲧⲱⲱⲙⲉ⟩ ⟪S⟫ (rare), ⟨ⲧⲱⲙⲉ⟩, ⟨ⲧⲟⲩ.⟩ ⟪A⟫, ⟨ⲧⲱⲙⲓ⟩ ⟪B⟫, ⟨ⲧⲟⲙ-⟩, ⟨ⲧⲉⲙ-⟩, ⟨ⲧⲟⲙ⸗⟩ ⟪B⟫, ⟨ⲧⲟ(ⲟ)ⲙⲉ†⟩ ⟪S⟫, ⟨ⲧⲁⲁⲙⲉ†⟩ ⟪Sa⟫ (⟦Sh⟧{reference: Sh}), ⟨ⲧⲁⲙⲉ†⟩ ⟪A2⟫, ⟨ⲧⲟⲙⲓ†⟩ ⟪B⟫ ⟦vb⟧{annotation: verb} ⟨bullet: I⟩ ⟦intr⟧{annotation: intransitive (i.e. verb without immediate object, or one constructed with prep. ⲉ-)}, ⟨gloss: join⟩: ⟦Ex 26 3⟧{bible: Exodus 26:3} ⟪B⟫ ⟨ἔχεσθαι⟩⟦;⟧ ⟦Job 14 12⟧{bible: Job 14:12} ⟪B⟫ (⟪S⟫ ⟨ⲉⲓ ⲉⲧⲟⲩⲛ-⟩) ⟨συρράπτειν⟩⟦;⟧ ⟦Ez 40 16⟧{bible: Ezekiel 40:16} ⟪Sf⟫⟪B⟫ ⟨κρυπτός⟩⟦;⟧ ⟦MG 25 255⟧{reference: MG} ⟪B⟫ broken cup ⟨ⲛⲉⲁⲡⲓⲁⲫⲟⲧ ⲧ. ⲡⲉ⟩, ⟦AM 85⟧{reference: AM} ⟪B⟫ ⟦sim⟧{annotation: similar in use or in meaning to the last quoted instance} ⟨ⲁϥⲧ.⟩⟦;⟧
¶ ⟦qual⟧{annotation: qualitative of verb; also indicated by †}, ⟨gloss: be fitting⟩, ⟨gloss: appropriate⟩: ⟦R 1 5 10⟧{reference: R} ⟪S⟫ ⟨ⲛⲟⲩϩⲱⲃ ⲁⲛ ⲉϥⲧ.⟩ = ⟦JA '75 5 233⟧{reference: JA} ⟪S⟫ ⟨ϣϣⲉ ⲁⲛ⟩ ⟨ἄπρεπες ἐστί⟩⟦;⟧ ⟦BSM 58⟧{reference: BSM} ⟪B⟫ ⟨ⲛⲉⲛⲧⲁⲓⲟ ⲉⲩⲧ.⟩ = ⟦BMis 416⟧{reference: B Mis} ⟪S⟫ ⟨ⲉⲧⲡⲣⲉⲡⲉ⟩⟦;⟧ ⟦ClPr 60 3⟧{reference: Cl Pr} ⟪S⟫ ⟨ⲉⲣⲉⲧⲙⲛⲧⲣⲉϥⲥⲙⲛ ⲛⲟⲙⲟⲥ ⲧ. ⲙⲡⲕⲁⲓⲣⲟⲥ ⲉⲧⲙⲙⲁⲩ⟩ ⟨ἁρμόζειν⟩⟦;⟧ ⟦K 245⟧{reference: K} ⟪B⟫ ⟨ⲛⲓϣⲑⲏⲛ ⲉⲧⲧ.⟩ next body, called ⟨الشعار⟩⟦;⟧ ⟦ShBor 246 83⟧{reference: Sh Bor} ⟪S⟫ ⟨ⲡϩⲱⲃ ⲧ. ⲁⲛ⟩, ⟦Z 281⟧{reference: Z} ⟪S⟫ ⟨ⲛⲧⲁⲡϫⲡⲓⲟ ϣⲱⲡⲉ ⲉϥⲧ.⟩, ⟦FR 114⟧{reference: FR} ⟪B⟫ ⟨ϯϩⲱⲇⲏ ⲉⲧⲧ.⟩, ⟦Leyd 150⟧{reference: Leyd} ⟪S⟫ ⟨ⲧⲁⲩⲟ ⲛⲉⲧⲥⲏϩ… ⲥⲉⲧ.⟩
¶ ⟨bullet: II⟩ ⟦tr⟧{annotation: transitive}, ⟨gloss: join⟩ ⟪B⟫: ⟦Ex 26 6⟧{bible: Exodus 26:6} ⟨συνάπτειν⟩⟦;⟧ ⟦Mt 19 6⟧{bible: Gospel of St. Matthew 19:6} (⟪S⟫ ⟨ϣⲱⲛⲃ⟩) ⟨συζευγνύναι⟩⟦;⟧ ⟦Is 44 13⟧{bible: Isaiah 44:13} (⟪S⟫ ⟨ⲧⲱϭⲉ⟩) ⟨ῥυθμίζειν⟩.

Things need to be elaborated: 

182b

¶ ⟨ⲙⲉⲣⲉ-⟩ ⟪S⟫⟪F⟫, ⟨ⲙⲁⲣⲉ-⟩ ⟪S⟫⟪A⟫⟪A2⟫, ⟨ⲙⲡⲁⲣⲉ-⟩ ⟪B⟫, ⟨ⲙⲉⲗⲉ-⟩ ⟪F⟫, ⟨ⲙⲁ-⟩ ⟪S⟫⟪A⟫, ⟨ⲙⲉ⸗⟩ ⟪S⟫⟪F⟫, ⟨ⲙⲁ⸗⟩ ⟪S⟫⟪A⟫, ⟨ⲙⲡⲁ⸗⟩ ⟪B⟫, ⟦pref⟧{annotation: prefix} ⟦neg⟧{annotation: negative} ⟦aor⟧{annotation: aorist}, ⟦opp⟧{annotation: as opposed to, contrasted with} ⟨ϣⲁⲣⲉ-⟩: ⟦Job 5 6⟧{bible: Job 5:6} ⟪S⟫ ⟨ⲙⲉⲣⲉϩⲓⲥⲉ ⲉⲓ⟩, ⟦Si 11 27⟧{bible: Wisdom of Jesus son of Sirach 11:27} ⟪S⟫ ⟨ⲙⲉⲕⲣ ⲡⲙⲉⲉⲩⲉ⟩ (⟦var⟧{annotation: variant, in same dialect} ⟨ⲙⲁⲕ-⟩), ⟦Jo 4 9⟧{bible: Gospel of St. John 4:9} ⟪S⟫⟪A2⟫⟪B⟫⟪F⟫ ⟨ⲙ. ⲓⲟⲩⲇⲁⲓ ⲧⲱϩ⟩, ⟦Mk 7 12⟧{bible: Gospel of St. Mark 7:12} ⟪S⟫ ⟨ⲙⲉⲧⲉⲧⲛⲕⲁⲁϥ⟩ (⟦var⟧{annotation: variant, in same dialect} ⟨ⲙⲙⲁⲧⲉⲧⲛ-⟩, ⟪B⟫ ⟨ⲛ-… ⲁⲛ⟩), ⟦Pro 27 24⟧{bible: Proverbs 27:24} ⟪S⟫ ⟨ⲙⲉⲩⲧⲁⲁϥ⟩ ⟪A⟫ ⟨ⲙⲁⲩ-⟩, ⟦EpJer 11⟧{bible: Epistle of Jeremiah 1:11} ⟪F⟫ ⟨ⲙⲉⲩⲉϣ ⲟⲩϫⲉⲓ⟩, ⟪B⟫ ⟨ⲙⲡⲁⲩ-⟩. With ⟨ⲁⲛ⟩: ⟦Jth 7 10⟧{bible: Judith 7:10} ⟪S⟫ ⟨ⲙⲉⲩⲕⲱ ⲛϩⲏⲧⲩ ⲁⲛ⟩. With ⟨ⲛⲉ-⟩ ⟦aor⟧{annotation: aorist} preterite: ⟦Ps 100 7⟧{bible: Psalms 100:7} ⟪S⟫ ⟨ⲛⲉⲙⲉⲓⲥⲟⲩⲧⲱⲛϥ⟩ (⟪B⟫ ⟨ⲙⲡⲁϥ-⟩) ⟨οὐ κατεύθυνειν⟩. ⟨ⲙⲁ-⟩ (rare) ⟦El 98⟧{reference: El} ⟪A⟫ ⟨ⲙⲁⲡⲭ︤ⲥ︥ ⳉⲱⲧⲃⲉ ⲇⲓⲕⲁⲓⲟⲥ⟩, ⟦PVi 10157⟧{reference: P Vi} ⟪A⟫ ⟨ⲙⲁⲑⲏⲇⲟⲛⲏ ⲙⲡⲏⲣⲡ ⳉ ⲣ ⲕⲁ[ⲧⲉ]ⲭⲉ ⲙⲙⲁϥ⟩, ⟦CO 300⟧{reference: C O} ⟪S⟫ ⟨ⲙⲁⲁⲡⲁ ⲃⲓⲕⲧⲱⲣ ⲣ ⲙⲛⲧⲥⲟϭ⟩ again, ⟦RE 9 137⟧{reference: RE} ⟪S⟫ ⟨ⲙⲁⲉⲓⲣⲏⲛⲏ…ϣⲱⲡⲉ⟩, ⟦Ep 437⟧{reference: Ep} ⟪S⟫ ⟨ⲙⲁⲣ ϩⲱⲃ ⲛⲁⲧⲁⲛϩⲁⲗⲱⲙⲁ⟩, ⟦PBu 257⟧{reference: P Bu} ⟪S⟫ we trust ⟨ϫⲉⲙⲁⲡⲛ︤ⲉ︥ ϯ ⲑⲉ ⲛⲥⲉⲙⲁⲧⲉ ⲙⲡⲁⲓ⟩ (⟦‹sc›⟧{annotation: scilicet} defraud us).

also:
163a

¶ ⟨ⲙⲟⲩⲕϩ⟩ ⟪S⟫⟪A⟫⟪F⟫ (⟦BMEA 10576⟧{reference: BMEA}), ⟨ⲙⲉⲕϩ-⟩, ⟨ⲙⲟⲕϩ⸗⟩ ⟪S⟫, ⟨ⲙⲁⲕϩ⸗⟩, ⟨ⲙⲁⲭϩ⸗⟩ (⟦Cl⟧{reference: Cl}) ⟪A⟫ ⟦vb⟧{annotation: verb} ⟨bullet: I⟩ ⟦tr⟧{annotation: transitive}, ⟨gloss: afflict⟩, ⟨gloss: oppress⟩ (⟦‹cf›⟧{annotation: confer} ⟨ⲙⲕⲁϩ⟩): ⟦Job 5 18⟧{bible: Job 5:18}...

¶ ⟨bullet: II⟩ ⟦intr⟧{annotation: intransitive (i.e. verb without immediate object, or one constructed with prep. ⲉ-)} (rare), ⟨gloss: labour⟩, ⟨gloss: find difficult⟩: ⟦Mor 17 13⟧{reference: Mor} ⟪S⟫ gladly thou eatest ⟨ⲉⲣ ϩⲱⲃ ⲇⲉ ϣⲁⲕⲙ.⟩ = ⟦EW 29⟧{reference: EW} ⟪B⟫ ⟨ⲉⲣ ϭⲉⲛⲛⲉ⟩ ⟨σιαίνειν⟩, ⟦BG 74⟧{reference: BG} ⟪S⟫ ⟨ⲛⲧⲁϥⲙ. ⲛⲙⲙⲁⲩ ϩⲙⲡⲕⲁⲕⲉ⟩, ⟦P 43 234⟧{reference: P} ⟪S⟫ ⟨ⲛⲉⲧⲙ. ⲛⲉⲣ ⲙⲏ⟩ (⟦‹v›⟧{annotation: vide} ⟨ⲙⲏ⟩).

- [x] Create a vocabulary Markdown file: `Word | Sense 1 | Sense 2 | …`. Start with father/barley (C992/C998), ϩⲟⲟⲩ, ⲧⲱⲙⲧ, ⲉϩⲉ, ⲡⲉ, and ⲉ. Add unit tests.

| Word | Sense 1 | Sense 2 | Sense 3 | Sense 4 | Sense 5 |
| --- | --- | --- | --- | --- | --- |
| ⲉⲓⲱⲧ | father [C992](https://coptic-dictionary.org/entry/C992) | barley [C998](https://coptic-dictionary.org/entry/C998) | father (title) [C993](https://coptic-dictionary.org/entry/C993) |  |  |
| ϩⲟⲟⲩ | day [C6965](https://coptic-dictionary.org/entry/C6965) | be bad, wicked [C6971](https://coptic-dictionary.org/entry/C6971) |  |  |  |
| ⲧⲱⲙⲧ | meet, happen, befall [C4324](https://coptic-dictionary.org/entry/C4324) | be amazed, be stupefied [C4327](https://coptic-dictionary.org/entry/C4327) |  |  |  |
| ⲉϩⲉ | cow [C771](https://coptic-dictionary.org/entry/C771) | ox [C772](https://coptic-dictionary.org/entry/C772) | yes; indeed; introduces a question [C773](https://coptic-dictionary.org/entry/C773) |  |  |
| ⲡⲉ | copula [C2788](https://coptic-dictionary.org/entry/C2788) | heaven, sky [C2790](https://coptic-dictionary.org/entry/C2790) | leap, jump [C2791](https://coptic-dictionary.org/entry/C2791) |  |  |
| ⲉ | preposition [C615](https://coptic-dictionary.org/entry/C615) | circumstantial converter [C616](https://coptic-dictionary.org/entry/C616) | relative converter [C617](https://coptic-dictionary.org/entry/C617) | focalising converter [C618](https://coptic-dictionary.org/entry/C618) | five [C614](https://coptic-dictionary.org/entry/C614) |
| ϣⲁ | begin [C5676](https://coptic-dictionary.org/entry/C5676) | rise (of sun) [C5679](https://coptic-dictionary.org/entry/C5679) |  |  |  |
| ⲛⲁⲩ | sight, vision [C2584](https://coptic-dictionary.org/entry/C2584) | hour, time [C2589](https://coptic-dictionary.org/entry/C2589) | look, see [C2578](https://coptic-dictionary.org/entry/C2578) |  |  |
| ϯ | give [C4020](https://coptic-dictionary.org/entry/C4020) | hit, strike [C4050](https://coptic-dictionary.org/entry/C4050) |  |  |  |
| ⲟⲩⲱϣ | wish, desire (noun) [C5258](https://coptic-dictionary.org/entry/C5258) | cleft, gap, interval [C5263](https://coptic-dictionary.org/entry/C5263) | wish, desire (verb) [C5257](https://coptic-dictionary.org/entry/C5257) |  |  |
| ⲙⲉ | love (noun) [C1786](https://coptic-dictionary.org/entry/C1786) | truth, justice [C1789](https://coptic-dictionary.org/entry/C1789) |  |  |  |
| ϩⲏⲧ | heart, mind [C6869](https://coptic-dictionary.org/entry/C6869) | north [C6884](https://coptic-dictionary.org/entry/C6884) | tip, edge [C6887](https://coptic-dictionary.org/entry/C6887) |  |  |
| ⲣⲟ | mouth [C3038](https://coptic-dictionary.org/entry/C3038) | goose [C3055](https://coptic-dictionary.org/entry/C3055) | strand of cord [C3054](https://coptic-dictionary.org/entry/C3054) |  |  |
| ϫⲟ | plant, sow [C7075](https://coptic-dictionary.org/entry/C7075) | put forth, spend [C7078](https://coptic-dictionary.org/entry/C7078) |  |  |  |
| ϫⲱ | head [C7097](https://coptic-dictionary.org/entry/C7097) | cup [C7098](https://coptic-dictionary.org/entry/C7098) | say, speak [C7089](https://coptic-dictionary.org/entry/C7089) |  |  |
| ⲥⲁ | side, part [C3263](https://coptic-dictionary.org/entry/C3263) | woman [C3271](https://coptic-dictionary.org/entry/C3271) | man [C3272](https://coptic-dictionary.org/entry/C3272) |  |  |
| ⲥⲟⲛ | brother (kinship) [C3583](https://coptic-dictionary.org/entry/C3583) | brother (monastic title) [C3584](https://coptic-dictionary.org/entry/C3584) |  |  |  |
| ⲥⲟⲙⲥ | look (verb) [C3551](https://coptic-dictionary.org/entry/C3551) | looking (noun) [C3553](https://coptic-dictionary.org/entry/C3553) |  |  |  |
| ϩⲱⲥ | sing [C6847](https://coptic-dictionary.org/entry/C6847) | be blocked, covered [C6849](https://coptic-dictionary.org/entry/C6849) | like, as [C11263](https://coptic-dictionary.org/entry/C11263) |  |  |
| ⲉϩⲣⲁⲓ | upward [C6769](https://coptic-dictionary.org/entry/C6769) | downward [C6775](https://coptic-dictionary.org/entry/C6775) |  |  |  |

- [x] Compare algorithm scores; report classification and ranking metrics separately.

Inclusion
Algorithm	AUC	Macro F1
Random Forest	0.9337	0.7770
LightGBM	0.9218	0.6530
XGBoost	0.9210	0.6685
CatBoost	0.9111	0.6262


Ranking
Algorithm	TAB NDCG@6	Crum NDCG@6
Random Forest	0.9497	0.8994
LightGBM	0.9510	0.9013
XGBoost	0.9523	0.9023
CatBoost	0.9128	0.9029

Hyperopt to tune XGBoost

after cluster !!



Random Forest — Inclusion: Top 10 Features
Feature	Importance
target_pos_conflict	13.17%
target_pos_match_ratio	11.69%
sense_cosine	9.96%
sense_cosine_missing	8.91%

cosine missing is the same with sense_cosine
make columns to figureing out each of them co-related

rival_count	6.67%
sense_cosine_margin	5.70%
frame_log_frequency	4.85%
sense_overlap_margin	4.16%
sense_overlap	3.92% / sen
sense_rouge	3.49%


Ranking: Top 10 Features
Feature	Importance
sense_cosine_margin	4.91%
mean_dep_distance	4.60%
sense_cosine	4.58%
frame_probability	4.55% (rethink before clustering)
mean_word_frequency	4.31%
avg_word_length	4.21%
target_position_ratio	4.10%
unique_word_ratio	4.07%
mean_log_word_frequency	3.69%
rival_cosine_max	3.22%

### Future work

- Example collections and clustering: select sets covering form, meaning, and syntax.
  - Syntax diversity: play multiple roles in the sentence.  
- Develop expert annotation guidelines.
- Multilingual semantic features: use CDO's English, German, and French glosses; test German/French cosine similarities and their averages or combinations, starting with C4324. Assess German/French Bible parallels and machine translation for non-Bible texts.
