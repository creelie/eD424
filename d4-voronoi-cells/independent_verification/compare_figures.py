"""Compare the PNGs in paper/figures of a working tree with the committed ones,
pixel by pixel.  Run from the root of a clone after running the scripts in
paper/figures_new (python3 fig_*.py) with matplotlib 3.10.9."""
import io, subprocess
import numpy as np
from PIL import Image
files = subprocess.check_output(['git', 'ls-files', 'paper/figures/*.png']).decode().split()
same = 0
for f in files:
    old = Image.open(io.BytesIO(subprocess.check_output(['git', 'show', 'HEAD:' + f])))
    new = Image.open(f)
    a = np.asarray(old.convert('RGBA')).astype(int)
    b = np.asarray(new.convert('RGBA')).astype(int)
    if a.shape != b.shape:
        print('%-40s size differs %s vs %s' % (f, a.shape, b.shape)); continue
    n = int((np.abs(a - b).max(axis=2) > 0).sum())
    bytes_same = subprocess.call(['git', 'diff', '--quiet', '--', f]) == 0
    same += (n == 0)
    print('%-40s %s pixels differ; bytes %s' % (f, n, 'identical' if bytes_same else 'differ'))
print('%d of %d figures pixel-identical' % (same, len(files)))
