import PyInstaller.__main__
import os
import shutil

def build():
    # Clean previous build
    if os.path.exists('dist'):
        shutil.rmtree('dist')
    if os.path.exists('build'):
        shutil.rmtree('build')

    print("Starting build process...")

    PyInstaller.__main__.run([
        'main.py',
        '--name=MeetingTranslator',
        '--noconsole',
        '--onedir',
        '--clean',
        '--collect-all=faster_whisper',
        '--collect-all=ctranslate2',
        '--collect-all=tokenizers',
        '--collect-all=torch',  # Ensure full torch is collected
        '--paths=.', 
    ])

    print("PyInstaller build complete.")

    dist_dir = os.path.join("dist", "MeetingTranslator")
    internal_dir = os.path.join(dist_dir, "_internal")

    # Post-processing: avoid duplicate OpenMP runtimes.
    # Both ctranslate2 and torch ship libiomp5md.dll. Loading two different copies
    # (different paths) can cause WinError 1114 during torch DLL initialization.
    # Keep ctranslate2's copy and remove torch's copies.
    torch_lib_dir = os.path.join(internal_dir, "torch", "lib")
    torch_iomp = os.path.join(torch_lib_dir, "libiomp5md.dll")
    torch_iomp_stub = os.path.join(torch_lib_dir, "libiompstubs5md.dll")

    for path in (torch_iomp, torch_iomp_stub):
        if os.path.exists(path):
            print(f"Removing duplicate OpenMP runtime: {path}")
            os.remove(path)

    # Ensure libiomp5md.dll exists in _internal root for torch/ctranslate2 to find
    ct2_iomp = os.path.join(internal_dir, "ctranslate2", "libiomp5md.dll")
    root_iomp = os.path.join(internal_dir, "libiomp5md.dll")
    
    if os.path.exists(ct2_iomp) and not os.path.exists(root_iomp):
        print(f"Copying OpenMP runtime to root: {ct2_iomp} -> {root_iomp}")
        shutil.copy2(ct2_iomp, root_iomp)

    print("Build output in dist/MeetingTranslator")

if __name__ == "__main__":
    build()
