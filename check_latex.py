#!/usr/bin/env python3
"""
Basic LaTeX syntax checker for main_paper.tex
Checks for common issues that would prevent compilation.
"""

import re
import sys

def check_latex_syntax(filepath):
    """Check for common LaTeX syntax errors."""
    errors = []
    warnings = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')
    
    # Check for balanced braces
    brace_count = 0
    for i, line in enumerate(lines, 1):
        # Skip comments
        clean_line = re.sub(r'(?<!\\)%.*', '', line)
        for char in clean_line:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
            if brace_count < 0:
                errors.append(f"Line {i}: Unmatched closing brace")
                brace_count = 0
    
    if brace_count != 0:
        errors.append(f"Unbalanced braces: {brace_count} unclosed")
    
    # Check for \\ before commands (common error from our edits)
    double_backslash_cmds = re.findall(r'\\\\(Cref|emph|cite|ref|label)', content)
    if double_backslash_cmds:
        warnings.append(f"Found double-backslash commands: {set(double_backslash_cmds)}")
    
    # Check for undefined labels
    labels_defined = set(re.findall(r'\\label\{([^}]+)\}', content))
    labels_referenced = set(re.findall(r'\\(?:ref|Cref)\{([^}]+)\}', content))
    undefined_refs = labels_referenced - labels_defined
    if undefined_refs:
        warnings.append(f"Potentially undefined references: {undefined_refs}")
    
    # Check for missing figure files
    figures = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', content)
    
    print("=" * 60)
    print("LaTeX Syntax Check Results")
    print("=" * 60)
    
    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")
    
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for warning in warnings:
            print(f"  - {warning}")
    
    if figures:
        print(f"\n📊 FIGURES REFERENCED ({len(figures)}):")
        for fig in figures:
            print(f"  - {fig}")
    
    print(f"\n📋 STATISTICS:")
    print(f"  - Total lines: {len(lines)}")
    print(f"  - Labels defined: {len(labels_defined)}")
    print(f"  - Labels referenced: {len(labels_referenced)}")
    
    if not errors and not warnings:
        print("\n✅ No syntax errors or warnings detected!")
        return 0
    elif errors:
        print("\n❌ Syntax errors found. Fix before compilation.")
        return 1
    else:
        print("\n⚠️  Warnings found. Review before compilation.")
        return 0

if __name__ == '__main__':
    filepath = sys.argv[1] if len(sys.argv) > 1 else 'main_paper.tex'
    sys.exit(check_latex_syntax(filepath))
