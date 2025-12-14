"""Fix indentation in dialog from line 495 to 998"""
with open('nba_gui_dashboard_v2.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Lines 495-997 need 4 more spaces
for i in range(494, 997):  # 0-indexed
    if i < len(lines):
        line = lines[i]
        # Only indent if line isn't already properly indented and isn't blank
        if line and not line.startswith('            ') and line.strip():
            # Add 4 spaces
            lines[i] = '    ' + line

with open('nba_gui_dashboard_v2.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Fixed indentation")
