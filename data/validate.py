import json

errors = []
line_count = 0
type_b_count = 0
empty_metrics = 0
with_metrics = 0
category_counts = {
    "cause_background": 0,
    "vague_generic": 0,
    "sub_item": 0,
    "qualifier": 0,
    "pure_qualitative": 0,
    "time_period": 0,
}

with open('/home/lmxxf/work/financial-report-generator-lora/data/type_b.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        line_count += 1
        try:
            obj = json.loads(line)
            assert obj['type'] == 'B', f'Line {i}: type is not B'
            assert 'chapter_title' in obj['input'], f'Line {i}: missing chapter_title'
            assert 'paragraph' in obj['input'], f'Line {i}: missing paragraph'
            assert 'analysis' in obj['output'], f'Line {i}: missing analysis'
            assert 'metrics' in obj['output'], f'Line {i}: missing metrics'

            type_b_count += 1
            metrics = obj['output']['metrics']
            if len(metrics) == 0:
                empty_metrics += 1
            else:
                with_metrics += 1
                for m in metrics:
                    assert 'metric_name' in m, f'Line {i}: missing metric_name'
                    assert 'metric_type' in m, f'Line {i}: missing metric_type'
                    assert m['metric_type'] in ('financial', 'business'), f'Line {i}: invalid metric_type'
                    assert 'score' in m, f'Line {i}: missing score'
                    assert 0.80 <= m['score'] <= 1.0, f'Line {i}: score out of range: {m["score"]}'
                    assert 'reason' in m, f'Line {i}: missing reason'

        except json.JSONDecodeError as e:
            errors.append(f'Line {i}: JSON parse error: {e}')
        except AssertionError as e:
            errors.append(str(e))

print(f'Total lines: {line_count}')
print(f'Type B entries: {type_b_count}')
print(f'Entries with metrics: {with_metrics}')
print(f'Entries with empty metrics: {empty_metrics}')
print(f'Errors: {len(errors)}')
for e in errors:
    print(f'  ERROR: {e}')

if len(errors) == 0 and line_count == 72:
    print('\nALL 72 ENTRIES VALID!')
elif len(errors) == 0:
    print(f'\nAll entries valid but count is {line_count}, expected 72')
