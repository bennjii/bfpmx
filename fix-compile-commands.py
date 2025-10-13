#!/usr/bin/env python3
import json
import subprocess
import sys

def get_system_includes(compiler='clang++'):
    """Extract system include paths from compiler."""
    result = subprocess.run(
        [compiler, '-v', '-x', 'c++', '/dev/null', '-fsyntax-only'],
        capture_output=True,
        text=True
    )

    lines = result.stderr.split('\n')
    includes = []
    capture = False

    for line in lines:
        if '#include <...> search starts here:' in line:
            capture = True
            continue
        if 'End of search list' in line:
            break
        if capture and line.strip():
            # Strip everything after (and including) parentheses like "(framework directory)"
            path = line.strip().split('(')[0].strip()
            if path:  # Only add non-empty paths
                includes.append(f'-isystem{path}')

    # Add sysroot for macOS
    try:
        sysroot = subprocess.run(
            [compiler, '-print-sysroot'],
            capture_output=True,
            text=True
        ).stdout.strip()
        if sysroot:
            includes.append(f'-isysroot{sysroot}')
    except:
        pass

    return ' '.join(includes)

def fix_compile_commands(input_file, output_file=None):
    """Add system includes to compile_commands.json."""
    if output_file is None:
        output_file = input_file

    with open(input_file, 'r') as f:
        commands = json.load(f)

    system_flags = get_system_includes()

    for cmd in commands:
        if 'command' in cmd:
            cmd['command'] += f' {system_flags}'
        elif 'arguments' in cmd:
            cmd['arguments'].extend(system_flags.split())

    with open(output_file, 'w') as f:
        json.dump(commands, f, indent=2)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: fix-compile-commands.py compile_commands.json")
        sys.exit(1)

    fix_compile_commands(sys.argv[1])
    print(f"Fixed {sys.argv[1]}")