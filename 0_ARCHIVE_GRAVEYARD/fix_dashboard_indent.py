"""Fix indentation in GameDetailDialog.init_ui method"""

with open('nba_gui_dashboard_v2.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

output = []
in_init_ui = False
after_try = False
try_indent = 0

for i, line in enumerate(lines):
    # Detect init_ui method
    if 'def init_ui(self):' in line:
        in_init_ui = True
        output.append(line)
        continue
    
    # Detect try block start
    if in_init_ui and line.strip() == 'try:':
        after_try = True
        try_indent = len(line) - len(line.lstrip())
        output.append(line)
        continue
    
    # Detect except block (end of try)
    if after_try and 'except Exception as e:' in line:
        after_try = False
        in_init_ui = False
        output.append(line)
        continue
    
    # Fix indentation for lines in try block
    if after_try:
        current_indent = len(line) - len(line.lstrip())
        # If line has less than 12 spaces and isn't blank, add 4 spaces
        if line.strip() and current_indent < (try_indent + 8):
            output.append('    ' + line)
        else:
            output.append(line)
    else:
        output.append(line)

with open('nba_gui_dashboard_v2.py', 'w', encoding='utf-8') as f:
    f.writelines(output)

print("✓ Fixed indentation in init_ui method")
