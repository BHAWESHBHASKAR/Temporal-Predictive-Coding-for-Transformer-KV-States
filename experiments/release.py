"""
DeltaFormer Release Script
Run this to reproduce all benchmarks and generate all graphs.
"""

import subprocess, sys, os, time

SCRIPTS = [
    ("deltaformer_honest.py", "Full benchmarks + 5 graphs (5 min)"),
    ("deltaformer_final.py", "Training pipeline (30 sec)"),
    ("full_context_test.py", "Scale verification (5 min)"),
    ("stability_test.py", "Long-horizon test (10 min)"),
]

def main():
    print("=" * 60)
    print("DELTAFORMER RELEASE SUITE")
    print("=" * 60)
    
    # Check requirements
    try:
        import torch
        print(f"[OK] PyTorch {torch.__version__}")
        import transformers
        print(f"[OK] Transformers {transformers.__version__}")
        import matplotlib
        print(f"[OK] matplotlib")
    except ImportError as e:
        print(f"[FAIL] Missing dependency: {e}")
        print("Run: pip install torch transformers matplotlib")
        sys.exit(1)
    
    if not torch.cuda.is_available():
        print("[WARN] CUDA not available. Benchmarks will be slow on CPU.")
    else:
        print(f"[OK] CUDA: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
    
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(scripts_dir)
    
    print(f"\nScripts directory: {scripts_dir}")
    print(f"\nRunning {len(SCRIPTS)} scripts...")
    print()
    
    for script, desc in SCRIPTS:
        path = os.path.join(scripts_dir, script)
        if not os.path.exists(path):
            print(f"[SKIP] {script} not found")
            continue
        
        print(f"[RUN] {script} - {desc}")
        print("-" * 50)
        t0 = time.time()
        
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        elapsed = time.time() - t0
        
        if result.returncode == 0:
            print(f"[OK] Completed in {elapsed:.0f}s")
        else:
            print(f"[FAIL] Return code {result.returncode}")
            print(f"  Error: {result.stderr[:200]}")
        print()
    
    # Verify outputs
    print("=" * 60)
    print("OUTPUTS GENERATED")
    print("=" * 60)
    
    expected = [
        "memory_graph.png", "generation_drift.png", "quality_tradeoff.png",
        "prediction_gain.png", "residual_analysis.png", "deltaformer_paper_figures.png"
    ]
    for f in expected:
        path = os.path.join(scripts_dir, f)
        size = os.path.getsize(path) / 1024 if os.path.exists(path) else 0
        status = f"{size:.0f} KB" if size > 0 else "MISSING"
        print(f"  {f:<40} {status}")
    
    print(f"\nAll results in: {scripts_dir}")
    print("Paper: DELTAFORMER_REPORT.md")
    print("README: README.md")

if __name__ == "__main__":
    main()
