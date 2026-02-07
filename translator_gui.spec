# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

# -----------------------------------------------------------------------------
# 1. COLLECT HIDDEN DEPENDENCIES
# -----------------------------------------------------------------------------
docx_datas, docx_binaries, docx_hidden = collect_all('docx')
ollama_datas, ollama_binaries, ollama_hidden = collect_all('ollama')

# Manually collecting langdetect data just in case
lang_datas, lang_binaries, lang_hidden = collect_all('langdetect')

all_datas = docx_datas + lang_datas + ollama_datas
all_binaries = docx_binaries + lang_binaries + ollama_binaries

# FORCE THESE IMPORTS:
# We explicitly add langdetect and its submodules to ensure code is bundled.
all_hidden = docx_hidden + ollama_hidden + lang_hidden + [
    'langdetect',
    'langdetect.detect',
    'langdetect.detector_factory',
    'langdetect.profiles'
]

# -----------------------------------------------------------------------------
# 2. ANALYSIS
# -----------------------------------------------------------------------------
block_cipher = None

a = Analysis(
    ['translator_gui.py'],
    pathex=[],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=all_hidden,  # <--- The forced list goes here
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Exclude conflicting GUI libs to prevent crashes and bloat
    excludes=[
        'matplotlib', 'numpy', 'pandas', 'scipy', 'IPython', 'pytest', 'PIL',
        'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'qtpy', 'pywin32'
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# -----------------------------------------------------------------------------
# 3. BUILD EXECUTABLE
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
    console=False,  # Keep this False for GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)