from pathlib import Path

import pytest

from edu_rag.config import PROJECT_ROOT
from edu_rag.ingest import Dataset, ReferenceSource, load_dataset


@pytest.mark.parametrize("semester,topics,exercises", [("first", 8, 16), ("second", 7, 14)])
def test_grade_one_datasets_keep_semesters_and_authorship_explicit(semester, topics, exercises):
    dataset = load_dataset(PROJECT_ROOT / f"data/g1/math_{semester}.json")
    assert dataset.course.grade == 1 and dataset.course.semester == semester
    assert len(dataset.topics) == topics
    assert sum(len(topic.exercises) for topic in dataset.topics) == exercises
    assert "非教材" in dataset.course.textbook_edition
    assert "本项目" in dataset.license_note
    assert dataset.reference_sources[0].usage == "topic_reference"
    assert str(dataset.reference_sources[0].url) == "https://www.sohu.com/a/431916810_99911694"


@pytest.mark.parametrize("url", ["javascript:alert(1)", "file:///C:/private.txt"])
def test_reference_link_must_be_an_http_page(url):
    with pytest.raises(ValueError):
        ReferenceSource(url=url, title="主题参考")
