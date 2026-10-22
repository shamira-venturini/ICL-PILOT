import pytest

from icl_pilot.chat.annotation_recovery import restore_annotations_to_reference


def chat(*lines):
    return '@UTF8\n@Begin\n' + ''.join('*CHI:\t' + line + '\n' for line in lines) + '@End\n'


def test_restores_reference_case_words_and_utterance_boundaries():
    reference = chat('he go .', 'then he walk home .')
    annotated = chat('He go [* m:03s:a] . Then he walk [* m:03s:a] home .')
    output, report = restore_annotations_to_reference(reference, annotated)
    assert output == chat('he go [* m:03s:a] .', 'then he walk [* m:03s:a] home .')
    assert len(report['insertions']) == 2


def test_does_not_repair_child_grammar_or_insert_missing_error_labels():
    reference = chat('he had lots of balloon .')
    annotated = chat('he lots of balloon [* m:0s:a] .')
    output, report = restore_annotations_to_reference(reference, annotated)
    assert output == chat('he had lots of balloon [* m:0s:a] .')
    assert not report['new_error_labels_inferred']


def test_annotation_on_changed_word_is_removed():
    reference = chat('he had a ball .')
    annotated = chat('he has [* m:base:ed] a ball .')
    output, report = restore_annotations_to_reference(reference, annotated)
    assert output == reference
    assert report['annotations'][0]['status'] == 'removed'


def test_repeated_word_ambiguity_does_not_create_an_arbitrary_anchor():
    reference = chat('her her her knee hurt .')
    annotated = chat('her [* s:r:gc:pro] her knee hurt .')
    output, report = restore_annotations_to_reference(reference, annotated)
    assert output == reference
    assert report['annotations'][0]['reason'] == 'changed_or_ambiguous_word_anchor'


def test_reviewed_incorrect_annotation_is_removed_with_reason():
    reference = chat('he lay down .')
    annotated = chat('he lay [* m:base:ed] down .')
    output, report = restore_annotations_to_reference(reference, annotated, decisions={
        0: {'action': 'remove', 'reason': 'lay is the grammatical past of lie in this context'}})
    assert output == reference
    assert report['annotations'][0]['review']['reason'].startswith('lay is')


def test_existing_retracing_and_new_error_code_are_not_duplicated():
    reference = chat('he go [/] go .')
    annotated = chat('he go [/] go [* m:03s:a] .')
    output, report = restore_annotations_to_reference(reference, annotated)
    assert output == chat('he go [/] go [* m:03s:a] .')
    assert report['annotations'][0]['status'] == 'already_in_reference'


def test_nonword_target_reconstruction_and_error_keep_order():
    reference = chat('he goed home .')
    annotated = chat('he goed [: went] [* m:=ed] home .')
    output, _ = restore_annotations_to_reference(reference, annotated)
    assert output == annotated


def test_new_error_on_retraced_word_precedes_reference_retracing_code():
    reference = chat('he go [/] go .')
    annotated = chat('he go [* m:03s:a] [/] go .')
    output, _ = restore_annotations_to_reference(reference, annotated)
    assert output == annotated


def test_removing_reference_annotation_requires_reference_review():
    reference = chat('he go [* m:03s:a] .')
    with pytest.raises(ValueError, match='authoritative reference'):
        restore_annotations_to_reference(reference, reference, decisions={
            0: {'action': 'remove', 'reason': 'unsupported'}})


def test_interposed_speaker_is_not_a_child_error_anchor():
    reference = chat('&*EXA:no no .')
    annotated = chat('&*EXA:no [/] no .')
    output, report = restore_annotations_to_reference(reference, annotated)
    assert output == reference
    assert report['annotations'][0]['reason'] == 'no_CHI_word_anchor'


def test_changed_neighbor_prevents_unsafe_error_transfer():
    reference = chat('he did go home .')
    annotated = chat('he go [* m:03s:a] home .')
    output, report = restore_annotations_to_reference(reference, annotated)
    assert output == reference
    assert report['annotations'][0]['reason'] == 'changed_or_ambiguous_adjacent_context'


def test_explicit_context_review_can_preserve_a_uniquely_anchored_tag():
    reference = chat('he go to aeroplane .')
    annotated = chat('he go [* m:03s:a] home .')
    output, report = restore_annotations_to_reference(reference, annotated, decisions={
        0: {'action': 'retain', 'reason': 'Changed destination does not affect the singular subject and bare verb'}})
    assert output == chat('he go [* m:03s:a] to aeroplane .')
    assert report['annotations'][0]['reason'] == 'reviewed_context_with_unique_word_anchor'


def test_context_review_cannot_force_an_ambiguous_repeated_anchor():
    reference = chat('her her her knee hurt .')
    annotated = chat('her [* s:r:gc:pro] her knee hurt .')
    output, report = restore_annotations_to_reference(reference, annotated, decisions={
        0: {'action': 'retain', 'reason': 'proposed review'}})
    assert output == reference
    assert report['annotations'][0]['status'] == 'removed'


def test_shortening_and_numeral_comparison_does_not_change_reference():
    reference = chat('he (i)s happy .', 'he got five balloon .')
    annotated = chat('he is [* m:vsg:a] happy .', 'he got 5 balloon [* m:0s:a] .')
    output, _ = restore_annotations_to_reference(reference, annotated)
    assert output == chat('he (i)s [* m:vsg:a] happy .', 'he got five balloon [* m:0s:a] .')


def test_annotation_inside_a_reference_contraction_is_not_forced():
    reference = chat("my tummie's hurting .")
    annotated = chat("my tummie [: tummy]'s hurting .")
    output, report = restore_annotations_to_reference(reference, annotated)
    assert output == reference
    assert report['annotations'][0]['status'] == 'removed'


def test_dependent_tiers_may_not_be_silently_discarded():
    with pytest.raises(ValueError, match='Dependent tiers differ'):
        restore_annotations_to_reference(chat('he go .'), chat('he go .') + '%err:\tmessage\n')
    with pytest.raises(ValueError, match='Morphosyntactic'):
        restore_annotations_to_reference(chat('he go .'), chat('he go .') + '%mor:\tpro|he v|go .\n')


def test_output_is_reference_plus_only_recorded_insertions():
    reference = chat('he go .', 'he goed back .')
    annotated = chat('he go [* m:03s:a] . he goed [: went] [* m:=ed] back .')
    output, report = restore_annotations_to_reference(reference, annotated)
    projected = output
    for item in reversed(report['insertions']):
        # Reverse reference offsets become output offsets after earlier inserts.
        start = item['reference_position'] + sum(len(v['text']) for v in report['insertions']
                                                 if v['reference_position'] < item['reference_position'])
        assert projected[start:start + len(item['text'])] == item['text']
        projected = projected[:start] + projected[start + len(item['text']):]
    assert projected == reference
