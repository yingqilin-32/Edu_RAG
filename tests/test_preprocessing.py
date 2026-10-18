import pytest

from edu_rag.ingest import CourseMetadata, Dataset, prepare_dataset
from edu_rag.preprocessing import PreprocessingError, clean_text, split_text
from edu_rag.embedding import BGEEncoder, EmbeddingError


def test_cleaning_removes_noise_but_preserves_math_and_code():
    raw = "\ufeff面积\xa0=\u200b长×宽，1/2 < 2/3。\r\n\r\n重复讲解。\n\n重复讲解。\n\n```python\n    x = 1  + 2\n```"
    cleaned = clean_text(raw)
    assert "面积 =长×宽，1/2 < 2/3。" in cleaned.text
    assert cleaned.text.count("重复讲解。") == 1
    assert "    x = 1  + 2" in cleaned.text
    assert cleaned.duplicate_blocks == 1


def test_page_numbers_only_removed_at_page_edges():
    cleaned = clean_text("课程页眉\n讲解\n\n3\n\n计算\n第1页\f课程页眉\n后续讲解\n2", boilerplate_lines=["课程页眉"])
    assert "课程页眉" not in cleaned.text and "第1页" not in cleaned.text
    assert "\n\n3\n\n" in cleaned.text
    assert cleaned.removed_lines == 4
    chunks = split_text("测试", cleaned.text, len, max_tokens=100)
    assert {chunk.source_page for chunk in chunks} == {1, 2}


def test_long_prose_keeps_every_sentence_under_budget_with_section_labels():
    sentences = [f"第{index}个知识要点必须完整保留。" for index in range(40)]
    raw = "## 面积\n\n" + "".join(sentences)
    chunks = split_text("几何", raw, len, max_tokens=120, overlap_tokens=20)
    assert len(chunks) > 1
    for sentence in sentences:
        assert sentence in "".join(chunk.content for chunk in chunks)
    assert all(len(chunk.content) <= 120 for chunk in chunks)
    assert all(chunk.content.startswith("# 几何\n### 面积\n\n") for chunk in chunks)
    assert all(chunk.metadata["token_count"] == len(chunk.content) for chunk in chunks)


def test_formula_and_table_remain_whole():
    formula = "$$\nS = a \\times b\n$$"
    table = "| 长 | 宽 |\n| --- | --- |\n| 8 | 3 |"
    chunks = split_text("面积", "一段讲解。\n\n" + formula + "\n\n" + table, len, max_tokens=60, overlap_tokens=0)
    assert sum(formula in chunk.content for chunk in chunks) == 1
    assert sum(table in chunk.content for chunk in chunks) == 1


@pytest.mark.parametrize("block", ["$$\n" + "a" * 100 + "\n$$", "| " + "列" * 100 + " |"])
def test_oversized_atomic_block_is_rejected_instead_of_truncated(block):
    with pytest.raises(PreprocessingError, match="超过"):
        split_text("测试", block, len, max_tokens=40, overlap_tokens=0)


def test_code_content_and_repeated_lines_inside_it_are_not_deduplicated():
    code = '```python\ntext = """hello\n\n\nhello"""\n\nprint(text)\n\nprint(text)\n```'
    assert code in clean_text(code).text


def test_nested_list_indentation_is_preserved():
    raw = "- 测量\n    - 使用相同单位\n        - 厘米"
    assert clean_text(raw).text == raw


def test_encoder_rejects_oversize_input_before_model_can_truncate(monkeypatch):
    encoder = BGEEncoder.__new__(BGEEncoder)
    encoder.max_length = 10
    monkeypatch.setattr("edu_rag.embedding.count_tokens", len)
    with pytest.raises(EmbeddingError, match="未截断"):
        encoder.encode(["这是超过设定长度的输入文本"])


def test_empty_cleaned_document_and_unclosed_formula_are_rejected():
    with pytest.raises(PreprocessingError, match="为空"):
        split_text("测试", "\ufeff\u200b", len)
    with pytest.raises(PreprocessingError, match="结束标记"):
        split_text("测试", "$$\nx=2", len)


def test_one_exercise_is_never_split_and_overlong_input_fails_before_import():
    dataset = Dataset.model_validate({"dataset_id": "test", "course_code": "TEST", "license_note": "原创",
        "topics": [{"code": "test", "title": "题目", "objective": "测试", "content": "完整知识说明。",
                    "exercises": [{"stem": "长" * 600, "answer": "1", "explanation": "说明"}]}]})
    with pytest.raises(PreprocessingError, match="单道题目"):
        prepare_dataset(dataset, token_counter=len)


def test_course_metadata_cannot_mislabel_grade_as_another_school_stage():
    with pytest.raises(ValueError, match="学段"):
        CourseMetadata(title="测试", curriculum="原创", school_stage="primary", grade=9,
                       subject="math", textbook_edition="测试版")
