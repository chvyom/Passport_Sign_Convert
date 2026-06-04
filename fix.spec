# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[
        ('C:\\Users\\vyom1\\AppData\\Local\\Programs\\Python\\Python313\\Lib\\site-packages\\mediapipe\\tasks\\c\\libmediapipe.dll', 'mediapipe/tasks/c'),
    ],
    datas=[
        ('face_processing', 'face_processing'),
        ('sign_processing', 'sign_processing'),
        ('face_processing/blaze_face_full_range.tflite', 'face_processing'),
        ('face_processing/selfie_segmenter.tflite', 'face_processing'),
        ('sign_processing/yolov8s.onnx', 'sign_processing'),
    ],
    hiddenimports=[
        'customtkinter',
        'cv2',
        'numpy',
        'mediapipe',
        'mediapipe.tasks',
        'mediapipe.tasks.c',
        'mediapipe.tasks.python',
        'mediapipe.tasks.python.vision',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='PassportDocumentProcessor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)