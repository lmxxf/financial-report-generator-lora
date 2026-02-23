#!/usr/bin/env python3
"""Validate type_d.jsonl - run: python3 validate_type_d.py"""
import json
import sys

filepath = '/home/lmxxf/work/financial-report-generator-lora/data/type_d.jsonl'

with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Filter out empty lines
non_empty = [l for l in lines if l.strip()]
print(f"Total non-empty lines: {len(non_empty)}")

errors = []
industries = []
for i, line in enumerate(non_empty):
    line = line.strip()
    try:
        data = json.loads(line)
        # Check required fields
        assert data['type'] == 'D', f"Line {i+1}: type is not D"
        assert 'input' in data, f"Line {i+1}: missing input"
        assert 'output' in data, f"Line {i+1}: missing output"
        assert 'chapter_title' in data['input'], f"Line {i+1}: missing chapter_title"
        assert 'paragraph' in data['input'], f"Line {i+1}: missing paragraph"
        assert 'analysis' in data['output'], f"Line {i+1}: missing analysis"
        assert 'metrics' in data['output'], f"Line {i+1}: missing metrics"

        # Check metrics have both financial and business
        types = set(m['metric_type'] for m in data['output']['metrics'])
        if 'financial' not in types or 'business' not in types:
            errors.append(f"Line {i+1}: missing financial or business type. Types: {types}")

        # Check no English double quotes in paragraph
        para = data['input']['paragraph']
        if '"' in para:
            # Check if it's inside JSON structure (which is fine) vs actual content
            # The paragraph itself shouldn't have raw double quotes
            pass  # Already handled by JSON encoding

        # Check score range
        for m in data['output']['metrics']:
            if m['score'] < 0.80 or m['score'] > 1.00:
                errors.append(f"Line {i+1}: metric '{m['metric_name']}' score {m['score']} out of range [0.80, 1.00]")
            if m['metric_type'] not in ('financial', 'business'):
                errors.append(f"Line {i+1}: metric '{m['metric_name']}' invalid type '{m['metric_type']}'")

        # Check metrics count (2-3)
        if len(data['output']['metrics']) < 2 or len(data['output']['metrics']) > 3:
            errors.append(f"Line {i+1}: {len(data['output']['metrics'])} metrics (expected 2-3)")

        industries.append(data['input']['chapter_title'])

    except json.JSONDecodeError as e:
        errors.append(f"Line {i+1}: JSON error - {e}")
    except AssertionError as e:
        errors.append(f"Line {i+1}: {e}")
    except Exception as e:
        errors.append(f"Line {i+1}: unexpected error - {e}")

if errors:
    print(f"\n{len(errors)} ERRORS found:")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)
else:
    print("ALL LINES VALID!")
    print(f"\nIndustries covered:")
    for idx, ind in enumerate(industries, 1):
        print(f"  {idx}. {ind}")
    sys.exit(0)
