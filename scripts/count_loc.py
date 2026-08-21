#!/usr/bin/env python3
"""
Enterprise LOC & Code Metrics Measurement Tool
"""

import os
import sys
from typing import Dict, Any, Tuple

EXCLUDE_DIRS = {'node_modules', '.git', 'venv', '.venv', '__pycache__', 'dist', 'build', 'coverage', 'htmlcov', '.pytest_cache', 'eggs', '.eggs', 'local_data', 'artifacts', '.idea', '.vscode'}

EXTENSIONS = {'.py': 'Python', '.ts': 'TypeScript', '.tsx': 'React TSX', '.js': 'JavaScript', '.jsx': 'React JSX', '.css': 'CSS', '.sql': 'SQL', '.sh': 'Shell', '.ps1': 'PowerShell', '.json': 'JSON Config', '.toml': 'TOML Config', '.yaml': 'YAML', '.yml': 'YAML', '.md': 'Documentation'}

def analyze_file(filepath: str, ext: str) -> Tuple[int, int, int]:
    code = 0
    comments = 0
    blanks = 0
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            in_multiline = False
            for line in f:
                s = line.strip()
                if not s:
                    blanks += 1
                    continue
                if ext == '.py':
                    if s.startswith('#'):
                        comments += 1
                    elif s.startswith(('"""', "'''")) and s.endswith(('"""', "'''")) and len(s) > 3:
                        comments += 1
                    elif s.startswith(('"""', "'''")):
                        in_multiline = not in_multiline
                        comments += 1
                    elif in_multiline:
                        comments += 1
                        if s.endswith(('"""', "'''")):
                            in_multiline = False
                    else:
                        code += 1
                elif ext in ['.ts', '.tsx', '.js', '.jsx', '.css']:
                    if s.startswith('//'):
                        comments += 1
                    elif s.startswith('/*') and s.endswith('*/'):
                        comments += 1
                    elif s.startswith('/*'):
                        in_multiline = True
                        comments += 1
                    elif in_multiline:
                        comments += 1
                        if '*/' in s:
                            in_multiline = False
                    else:
                        code += 1
                elif ext == '.sql':
                    if s.startswith('--'):
                        comments += 1
                    else:
                        code += 1
                else:
                    if s.startswith(('#', '//', '--')):
                        comments += 1
                    else:
                        code += 1
    except Exception:
        pass
    return code, comments, blanks

def categorize_path(rel_path: str) -> str:
    parts = rel_path.replace('\\', '/').split('/')
    top = parts[0] if parts else 'root'
    if top in ['tests']:
        return 'Test Suite'
    elif top in ['infrastructure', 'docker', 'nginx', 'aws']:
        return 'Infrastructure & DevOps'
    elif top in ['docs']:
        return 'Documentation'
    elif top in ['scripts']:
        return 'Scripts & Tooling'
    elif top in ['apps', 'packages', 'ml', 'backend', 'pipelines', 'frontend']:
        return 'Production Application'
    return 'Root Config'

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    by_language = {}
    by_category = {}
    by_module = {}
    total_code = 0
    total_comments = 0
    total_blanks = 0
    file_count = 0

    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in EXTENSIONS:
                full_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(full_path, root_dir)
                lang = EXTENSIONS[ext]
                cat = categorize_path(rel_path)
                parts = rel_path.replace('\\', '/').split('/')
                module = '/'.join(parts[:2]) if len(parts) > 1 else parts[0]
                code, comments, blanks = analyze_file(full_path, ext)
                file_count += 1
                total_code += code
                total_comments += comments
                total_blanks += blanks
                by_language[lang] = by_language.get(lang, 0) + code
                by_category[cat] = by_category.get(cat, 0) + code
                by_module[module] = by_module.get(module, 0) + code

    print("=" * 80)
    print("  ENTERPRISE CUSTOMER CHURN PREDICTION & MLOPS PLATFORM - LOC AUDIT")
    print("=" * 80)
    print(f"Target Root: {root_dir}")
    print(f"Total Files Scanned: {file_count:,}")
    print(f"Total Physical Lines: {total_code + total_comments + total_blanks:,}")
    print(f"  -> Meaningful Source LOC : {total_code:,}")
    print(f"  -> Comment Lines        : {total_comments:,}")
    print(f"  -> Blank Lines          : {total_blanks:,}")
    print("-" * 80)
    print("LOC BY CATEGORY:")
    for cat, loc in sorted(by_category.items(), key=lambda x: x[1], reverse=True):
        pct = (loc / total_code * 100) if total_code > 0 else 0
        print(f"  {cat:<28} : {loc:>8,} lines ({pct:>5.1f}%)")
    print("-" * 80)
    print("LOC BY LANGUAGE:")
    for lang, loc in sorted(by_language.items(), key=lambda x: x[1], reverse=True):
        pct = (loc / total_code * 100) if total_code > 0 else 0
        print(f"  {lang:<28} : {loc:>8,} lines ({pct:>5.1f}%)")
    print("-" * 80)
    print("LOC BY TOP MODULES:")
    for mod, loc in sorted(by_module.items(), key=lambda x: x[1], reverse=True)[:15]:
        pct = (loc / total_code * 100) if total_code > 0 else 0
        print(f"  {mod:<28} : {loc:>8,} lines ({pct:>5.1f}%)")
    print("=" * 80)
    target = 100000
    progress = (total_code / target * 100) if target > 0 else 0
    print(f"100K LOC GOAL PROGRESS: {total_code:,} / {target:,} ({progress:.1f}%)")
    print("=" * 80)

if __name__ == '__main__':
    main()
