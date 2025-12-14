#!/usr/bin/env python3
"""Fix indentation in nba_gui_dashboard_v2.py"""

with open('nba_gui_dashboard_v2.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Lines 1001-1085 need to be dedented by 4 spaces (they have 12 instead of 8)
fixed_lines = []
for i, line in enumerate(lines, 1):
    if 1001 <= i <= 1085:
        # Remove 4 spaces from start if present
        if line.startswith('            '):
            fixed_lines.append(line[4:])
        else:
            fixed_lines.append(line)
    else:
        fixed_lines.append(line)

with open('nba_gui_dashboard_v2.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print("Fixed indentation for lines 1001-1085")
