#!/usr/bin/env python3
"""
Automated fix for Keras 3 compatibility
Run this script in your project directory
"""
import os
import re

def fix_file(filename, old_pattern, new_pattern):
    """Replace pattern in file"""
    if not os.path.exists(filename):
        print(f"⚠️  {filename} not found, skipping...")
        return
    
    with open(filename, 'r') as f:
        content = f.read()
    
    if old_pattern in content:
        content = content.replace(old_pattern, new_pattern)
        with open(filename, 'w') as f:
            f.write(content)
        print(f"✓ Fixed {filename}")
    else:
        print(f"✓ {filename} already OK (or pattern not found)")

print("=" * 60)
print("FIXING KERAS 3 COMPATIBILITY ISSUES")
print("=" * 60)

# Fix utils.py
fix_file('utils.py', 
         'model.compile(optimizer=optimizers.legacy.Adam(1e-3)',
         'model.compile(optimizer=optimizers.Adam(learning_rate=1e-3)')

# Fix client_flask.py
fix_file('client_flask.py',
         'opt = tf.keras.optimizers.legacy.Adam(lr)',
         'opt = optimizers.Adam(learning_rate=lr)')

# Fix gtv_simulation.py
fix_file('gtv_simulation.py',
         'opt = tf.keras.optimizers.legacy.SGD(self.lr)',
         'opt = optimizers.SGD(learning_rate=self.lr)')

print("=" * 60)
print("✓ ALL FIXES APPLIED!")
print("=" * 60)
print("\nNow run: python run_demo.py")