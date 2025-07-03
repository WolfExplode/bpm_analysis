# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'numpy',
        'pandas', 
        'scipy',
        'scipy.signal',
        'scipy.io',
        'plotly',
        'plotly.graph_objects',
        'plotly.subplots',
        'ttkbootstrap',
        'datetime',
        'logging',
        'json',
        'enum',
        'typing'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['pydub', 'pyaudioop', 'audioop'],  # Exclude problematic audio modules
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='BPM_Analyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to False for GUI application
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # You can add an icon file here if you have one
) 