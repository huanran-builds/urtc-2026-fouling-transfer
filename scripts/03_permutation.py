import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_validate
import sys
sys.path.insert(0, 'scripts')
from importlib import import_module

# reuse the setup from script 01
exec(open('scripts/01_grouped_cv.py').read().split('SCORING = ')[0])

rng = np.random.default_rng(42)
y_shuf = pd.Series(rng.permutation(y.values), index=y.index)

r = cross_validate(build_pipeline(), X, y_shuf, scoring=['accuracy'],
                   cv=StratifiedKFold(5, shuffle=True, random_state=42))
g = cross_validate(build_pipeline(), X, y_shuf, groups=groups,
                   scoring=['accuracy'], cv=GroupKFold(5))

print(f"SHUFFLED labels")
print(f"  random  {r['test_accuracy'].mean():.3f}")
print(f"  grouped {g['test_accuracy'].mean():.3f}")
print(f"  (expect ~0.42 and ~0.33 if there is no hidden leak)")