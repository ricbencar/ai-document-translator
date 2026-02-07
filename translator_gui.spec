# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

# -----------------------------------------------------------------------------
# 1. COLLECT DATA (Required for docx/langdetect)
# -----------------------------------------------------------------------------
docx_ret = collect_all('docx')
lang_ret = collect_all('langdetect')
ollama_ret = collect_all('ollama')

all_datas = docx_ret[0] + lang_ret[0] + ollama_ret[0]
all_binaries = docx_ret[1] + lang_ret[1] + ollama_ret[1]
all_hidden = docx_ret[2] + lang_ret[2] + ollama_ret[2]

# -----------------------------------------------------------------------------
# 2. ANALYSIS
# -----------------------------------------------------------------------------
block_cipher = None

a = Analysis(
    ['translator_gui.py'],
    pathex=[],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=all_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # -------------------------------------------------------------------------
    # SAFE EXCLUSIONS ONLY
    # We only exclude large external libraries that we know you aren't using.
    # We removed 'distutils', 'setuptools', 'xml', etc. to stop the crash.
    # -------------------------------------------------------------------------
    excludes=[
        'matplotlib', 'numpy', 'pandas', 'scipy', 'IPython', 'pytest', 
        'PIL', 'cv2', 'notebook', 'sklearn',
        'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'wx', 'kivy'
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# -----------------------------------------------------------------------------
# 3. BUILD
# -----------------------------------------------------------------------------
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='translator_gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False, 
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)