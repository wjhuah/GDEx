import json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from coptic_examples.lexicon import Entry
from coptic_examples.sense import targets
class IsaiahDaysTests(unittest.TestCase):
 def test_days_of_mourning_excluded_from_cattle(self):
  c=json.loads((Path(__file__).parent/'fixtures/isaiah_days.json').read_text())
  for tla in ['C771','C772']:
   self.assertFalse(targets(c,Entry(tla,'ⲉϩⲉ','NOUN','cow',('ⲉϩⲉ','ⲉϩⲟⲟⲩ'))))
  self.assertTrue(targets(c,Entry('C6965','ϩⲟⲟⲩ','NOUN','day',('ϩⲟⲟⲩ',))))
