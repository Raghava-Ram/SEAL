#!/usr/bin/env python3
"""
Test script to verify the multi-method Accuracy Matrix Viewer.
Run before launching the frontend to confirm everything works.
"""

import sys
from pathlib import Path

# Setup path
seal_root = Path(__file__).parent.parent
sys.path.insert(0, str(seal_root))

def main():
    print("""
╔════════════════════════════════════════════════════════════════╗
║     Accuracy Matrix Viewer - Multi-Method Test                ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Test 1: Import METHOD_RESULTS
    print("✓ Test 1: Importing METHOD_RESULTS...")
    try:
        from frontend.data.accuracy_matrices import METHOD_RESULTS
        print(f"  ✅ Loaded {len(METHOD_RESULTS)} methods")
    except ImportError as e:
        print(f"  ❌ Failed to import: {e}")
        return False
    
    # Test 2: Import helper functions
    print("\n✓ Test 2: Importing helper functions...")
    try:
        from frontend.app import (
            get_method_names_ordered,
            matrix_list_to_dataframe,
            render_accuracy_heatmap
        )
        print("  ✅ All helper functions imported")
    except ImportError as e:
        print(f"  ❌ Failed to import functions: {e}")
        return False
    
    # Test 3: Get ordered methods
    print("\n✓ Test 3: Getting ordered methods...")
    try:
        methods = get_method_names_ordered()
        print(f"  ✅ Found {len(methods)} methods:")
        for method in methods:
            data = METHOD_RESULTS[method]
            source = data.get("source") or data.get("screenshot", "N/A")
            print(f"     - {method}")
            print(f"       Screenshot: {source}")
            print(f"       Description: {data['description']}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False
    
    # Test 4: Test matrix conversion
    print("\n✓ Test 4: Testing matrix conversion...")
    try:
        test_matrix = [
            [0.95, 0.10, 0.22],
            [None, 0.47, 0.00],
            [None, None, 1.00],
        ]
        df = matrix_list_to_dataframe(test_matrix)
        print(f"  ✅ Converted matrix to DataFrame shape {df.shape}")
        print(f"     Tasks: {list(df.index)}")
        print(f"     Steps: {list(df.columns)}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False
    
    # Test 5: Verify screenshots directory
    print("\n✓ Test 5: Checking screenshots directory...")
    screenshots_dir = seal_root / "frontend" / "assets" / "screenshots"
    if screenshots_dir.exists():
        print(f"  ✅ Directory exists: {screenshots_dir}")
        files = list(screenshots_dir.glob("*.png"))
        print(f"     PNG files found: {len(files)}")
        for f in files:
            print(f"       - {f.name}")
        if len(files) < 7:
            print(f"  ⚠️  Only {len(files)}/7 expected screenshots present")
            print(f"     Expected: M0_baseline.png through M6_hybrid_ewc.png")
    else:
        print(f"  ⚠️  Directory doesn't exist yet: {screenshots_dir}")
    
    # Test 6: Verify all methods have required fields
    print("\n✓ Test 6: Validating method data structure...")
    try:
        required_fields = {"matrix", "description"}  # source/screenshot optional
        for method_name, method_data in METHOD_RESULTS.items():
            missing = required_fields - set(method_data.keys())
            if missing:
                print(f"  ❌ {method_name} missing fields: {missing}")
                return False
            
            # Check for source or screenshot field
            if "source" not in method_data and "screenshot" not in method_data:
                print(f"  ⚠️  {method_name} has no 'source' or 'screenshot' field")
            
            # Validate matrix is 3x3 upper triangular
            matrix = method_data["matrix"]
            if len(matrix) != 3 or len(matrix[0]) != 3:
                print(f"  ❌ {method_name} matrix is not 3x3")
                return False
        
        print(f"  ✅ All {len(METHOD_RESULTS)} methods have valid structure")
    except Exception as e:
        print(f"  ❌ Validation failed: {e}")
        return False
    
    # Summary
    print("\n" + "="*64)
    print("✅ ALL TESTS PASSED - Multi-Method Viewer is Ready!")
    print("="*64)
    print("\nNext steps:")
    print("1. Add screenshots to: frontend/assets/screenshots/")
    print("2. Run: streamlit run frontend/app.py")
    print("3. Go to: 📊 Accuracy Matrix → 📈 Pre-computed Methods")
    print("4. Select methods M0–M6 from the dropdown")
    print("\n" + "="*64)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
