"""Write the concise English vocabulary table."""
from pathlib import Path

ROWS = [['ⲉⲓⲱⲧ', [['father', 'C992'], ['barley', 'C998'], ['father (title)', 'C993']]],
 ['ϩⲟⲟⲩ', [['day', 'C6965'], ['be bad, wicked', 'C6971']]],
 ['ⲧⲱⲙⲧ', [['meet, happen, befall', 'C4324'], ['be amazed, be stupefied', 'C4327']]],
 ['ⲉϩⲉ', [['cow', 'C771'], ['ox', 'C772'], ['yes; indeed; introduces a question', 'C773']]],
 ['ⲡⲉ', [['copula', 'C2788'], ['heaven, sky', 'C2790'], ['leap, jump', 'C2791']]],
 ['ⲉ',
  [['preposition', 'C615'],
   ['circumstantial converter', 'C616'],
   ['relative converter', 'C617'],
   ['focalising converter', 'C618'],
   ['five', 'C614']]],
 ['ϣⲁ', [['begin', 'C5676'], ['rise (of sun)', 'C5679']]],
 ['ⲛⲁⲩ', [['sight, vision', 'C2584'], ['hour, time', 'C2589'], ['look, see', 'C2578']]],
 ['ϯ', [['give', 'C4020'], ['hit, strike', 'C4050']]],
 ['ⲟⲩⲱϣ',
  [['wish, desire (noun)', 'C5258'],
   ['cleft, gap, interval', 'C5263'],
   ['wish, desire (verb)', 'C5257']]],
 ['ⲙⲉ', [['love (noun)', 'C1786'], ['truth, justice', 'C1789']]],
 ['ϩⲏⲧ', [['heart, mind', 'C6869'], ['north', 'C6884'], ['tip, edge', 'C6887']]],
 ['ⲣⲟ', [['mouth', 'C3038'], ['goose', 'C3055'], ['strand of cord', 'C3054']]],
 ['ϫⲟ', [['plant, sow', 'C7075'], ['put forth, spend', 'C7078']]],
 ['ϫⲱ', [['head', 'C7097'], ['cup', 'C7098'], ['say, speak', 'C7089']]],
 ['ⲥⲁ', [['side, part', 'C3263'], ['woman', 'C3271'], ['man', 'C3272']]],
 ['ⲥⲟⲛ', [['brother (kinship)', 'C3583'], ['brother (monastic title)', 'C3584']]],
 ['ⲥⲟⲙⲥ', [['look (verb)', 'C3551'], ['looking (noun)', 'C3553']]],
 ['ϩⲱⲥ', [['sing', 'C6847'], ['be blocked, covered', 'C6849'], ['like, as', 'C11263']]],
 ['ⲉϩⲣⲁⲓ', [['upward', 'C6769'], ['downward', 'C6775']]]]


def render():
    lines = ["| Word | Sense 1 | Sense 2 | Sense 3 | Sense 4 | Sense 5 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for word, senses in ROWS:
        cells = [f"{gloss} [{tla}](https://coptic-dictionary.org/entry/{tla})" for gloss, tla in senses]
        lines.append("| " + " | ".join([word] + cells + [""] * (5 - len(cells))) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    (Path(__file__).resolve().parents[1] / "homographs_top20.md").write_text(render())
