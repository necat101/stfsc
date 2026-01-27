
filename = "src/main.rs"

def check_braces(filename):
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return

    stack = []
    
    for i, line in enumerate(lines):
        line_num = i + 1
        
        in_string = False
        in_char = False
        in_line_comment = False
        
        j = 0
        while j < len(line):
            char = line[j]
            
            if in_line_comment:
                break 
            
            if char == '"' and not in_char:
                if j > 0 and line[j-1] == '\\':
                    pass
                else:
                    in_string = not in_string
            
            if char == "'" and not in_string:
                pass 
                
            if not in_string and not in_char:
                if char == '/' and j + 1 < len(line) and line[j+1] == '/':
                    in_line_comment = True
                    j += 1
                elif char == '{':
                    stack.append(line_num)
                    if len(stack) <= 2:
                        print(f"DEBUG: Depth {len(stack)} opened at {line_num}: {line.strip()}")
                elif char == '}':
                    if not stack:
                        print(f"ERROR: Unexpected closing brace '}}' at Line {line_num}, Column {j+1}")
                        return
                    
                    if len(stack) <= 2:
                        print(f"DEBUG: Depth {len(stack)} closed at {line_num}: {line.strip()}")
                    
                    stack.pop()
            
            j += 1
            
    if stack:
        print(f"ERROR: Unclosed opening brace '{{' at Line {stack[-1]}")
    else:
        print("Success: Braces are balanced.")

check_braces(filename)
