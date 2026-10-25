# Build on the target OS with: python -m PyInstaller --noconfirm BITL.spec
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

root = Path(SPECPATH)
mp_data, mp_bins, mp_hidden = collect_all('mediapipe')
data = mp_data + [(str(root / 'models'), 'models'), (str(root / 'ar_hands' / 'web'), 'ar_hands/web')]
for folder in ('assets/sounds', 'assets/sprites'):
    if (root / folder).is_dir():
        data.append((str(root / folder), folder))
a = Analysis([str(root / 'main.py')], pathex=[str(root)], binaries=mp_bins,
             datas=data, hiddenimports=mp_hidden, excludes=[], noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='BITL', debug=False,
          bootloader_ignore_signals=False, strip=False, upx=False, console=True)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='BITL')
