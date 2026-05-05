import os

def generate_tree(startpath, output_file):
    ignore_dirs = {
        '.git', '.venv', '__pycache__', '.pytest_cache', 
        '.gemini', '.gemini-clipboard', '.idea', '.vscode', 
        'wandb', 'node_modules', '.mypy_cache', 'site-packages'
    }
    ignore_files = {
        '.DS_Store', 'Thumbs.db', '.gitignore', '.gitattributes'
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Project Structure: {os.path.basename(os.path.abspath(startpath))}\n\n")
        f.write("```text\n")
        
        for root, dirs, files in os.walk(startpath):
            # Modify dirs in-place to skip ignored directories
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            level = root.replace(startpath, '').count(os.sep)
            indent = '│   ' * (level)
            f.write(f"{indent}├── {os.path.basename(root)}/\n")
            # Filter files first
            filtered_files = [file for file in files if file not in ignore_files]
            filtered_files.sort()
            
            subindent = '│   ' * (level + 1)
            for i, file in enumerate(filtered_files):
                if i >= 5:
                    f.write(f"{subindent}├── ... ({len(filtered_files) - 5} more files)\n")
                    break
                f.write(f"{subindent}├── {file}\n")
        
        f.write("```\n")

if __name__ == "__main__":
    generate_tree('.', 'structure_for_claude.md')
    print("Structure saved to structure_for_claude.md")
