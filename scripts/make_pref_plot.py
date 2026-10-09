import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from lab22 import config as C, data as D

tokenizer = AutoTokenizer.from_pretrained(C.BASE_MODEL)
def n_tokens(text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)['input_ids'])

train_ds = Dataset.from_parquet(str(C.PREF_DIR / 'train.parquet'))
stats = D.length_stats(list(train_ds), count=n_tokens)

chosen = np.array([n_tokens(r['chosen'][0]['content']) for r in train_ds])
rejected = np.array([n_tokens(r['rejected'][0]['content']) for r in train_ds])

fig, ax = plt.subplots(figsize=(8, 3.5))
bins = np.linspace(0, C.MAX_LEN, 40)
ax.hist(chosen, bins=bins, alpha=0.6, label='chosen', color='#2e548a')
ax.hist(rejected, bins=bins, alpha=0.6, label='rejected', color='#c83538')
ax.set_xlabel('response tokens')
ax.set_title(f"Length: chosen longer in {stats['chosen_longer_frac']:.0%} of pairs")
ax.legend()
C.SCREENSHOTS.mkdir(parents=True, exist_ok=True)
fig.savefig(C.SCREENSHOTS / '02b-pref-length.png', dpi=120, bbox_inches='tight')
print('Saved 02b-pref-length.png successfully')
