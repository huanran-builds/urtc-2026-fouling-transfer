import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_validate
import sys
sys.path.insert(0, 'scripts')
from importlib import import_module

# reuse the setup from script 01
exec(open('scripts/01_grouped_cv.py').read().split('SCORING = ')[0])

rng = np.random.default_rng(42)
y_shuf = pd.Series(rng.permutation(y.values), index=y.index)

r = cross_validate(build_pipeline(), X_raw, y_shuf, scoring=['accuracy'],
                   cv=StratifiedKFold(5, shuffle=True, random_state=42))
g = cross_validate(build_pipeline(), X_raw, y_shuf, groups=groups,
                   scoring=['accuracy'], cv=GroupKFold(5))

print(f"SHUFFLED labels")
print(f"  random  {r['test_accuracy'].mean():.3f}")
print(f"  grouped {g['test_accuracy'].mean():.3f}")
print(f"  (both should sit near the 0.418 majority baseline if nothing is leaking)")