import sys
modules = ["torch", "torchvision", "PIL", "cv2", "gradio", "numpy", "scipy", "matplotlib"]
print(f"Python: {sys.version}")
for m in modules:
    try:
        mod = __import__(m)
        ver = getattr(mod, "__version__", "installed")
        print(f"  [+] {m}: {ver}")
    except ImportError as e:
        print(f"  [-] {m}: Not Installed ({e})")
