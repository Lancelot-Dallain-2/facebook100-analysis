"""
Fix Unicode emoji characters for Windows compatibility
"""
import os

files_to_fix = [
    'q3_assortativity.py',
    'q4_link_prediction.py', 
    'q5_label_propagation.py',
    'q6_community.py',
    'main.py',
    'verify_setup.py'
]

replacements = {
    '📊': '[STATS]',
    '📋': '[ANALYSIS]',
    '✓': '[OK]',
    '✗': '[FAIL]',
    '⚠': '[WARN]',
    '⏳': '[WAIT]',
    'ℹ': '[INFO]'
}

for filename in files_to_fix:
    if not os.path.exists(filename):
        print(f"Skipping {filename} (not found)")
        continue
    
    print(f"Processing {filename}...")
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    for emoji, replacement in replacements.items():
        content = content.replace(emoji, replacement)
    
    if content != original_content:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ Fixed {filename}")
    else:
        print(f"  - No changes needed for {filename}")

print("\nAll files processed!")
