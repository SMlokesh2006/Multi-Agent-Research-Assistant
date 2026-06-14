"""Tests for the state module."""

from src.state import (
    AnalysisResult,
    Finding,
    PageContent,
    ResearchState,
    SearchResult,
    create_initial_state,
)


class TestSearchResult:
    """Tests for SearchResult data model."""

    def test_to_dict_roundtrip(self):
        result = SearchResult(
            url="https://example.com",
            title="Example",
            snippet="A test snippet",
            score=0.95,
        )
        data = result.to_dict()
        restored = SearchResult.from_dict(data)
        assert restored.url == result.url
        assert restored.title == result.title
        assert restored.snippet == result.snippet
        assert restored.score == result.score

    def test_from_dict_defaults(self):
        data = {"url": "https://example.com", "title": "Test", "snippet": "Snippet"}
        result = SearchResult.from_dict(data)
        assert result.score == 0.0


class TestPageContent:
    """Tests for PageContent data model."""

    def test_to_dict_roundtrip(self):
        page = PageContent(
            url="https://example.com",
            title="Example Page",
            raw_content="Full text content...",
            summary="A summary of the content.",
            word_count=150,
            extraction_method="tavily",
        )
        data = page.to_dict()
        restored = PageContent.from_dict(data)
        assert restored.url == page.url
        assert restored.summary == page.summary
        assert restored.extraction_method == "tavily"


class TestFinding:
    """Tests for Finding data model."""

    def test_to_dict_roundtrip(self):
        finding = Finding(
            claim="AI improves productivity",
            evidence="Study showed 40% increase",
            source_url="https://example.com/study",
            confidence=0.85,
        )
        data = finding.to_dict()
        restored = Finding.from_dict(data)
        assert restored.claim == finding.claim
        assert restored.confidence == 0.85


class TestAnalysisResult:
    """Tests for AnalysisResult data model."""

    def test_empty_analysis(self):
        analysis = AnalysisResult()
        data = analysis.to_dict()
        assert data["key_findings"] == []
        assert data["conflicts"] == []
        assert data["needs_human_review"] is False

    def test_full_analysis_roundtrip(self):
        analysis = AnalysisResult(
            key_findings=[
                Finding(
                    claim="Test claim",
                    evidence="Test evidence",
                    source_url="https://example.com",
                    confidence=0.9,
                )
            ],
            conflicts=["Source A says X, Source B says Y"],
            knowledge_gaps=["No data on Z"],
            follow_up_queries=["What about Z?"],
            needs_human_review=True,
            review_reason="Significant conflicts found",
        )
        data = analysis.to_dict()
        restored = AnalysisResult.from_dict(data)
        assert len(restored.key_findings) == 1
        assert restored.key_findings[0].claim == "Test claim"
        assert restored.needs_human_review is True


class TestResearchState:
    """Tests for ResearchState creation."""

    def test_create_initial_state(self):
        state = create_initial_state("What is AI?")
        assert state["query"] == "What is AI?"
        assert state["status"] == "initialized"
        assert state["search_results"] == []
        assert state["iteration"] == 0
        assert state["max_iterations"] == 3

    def test_create_initial_state_custom_iterations(self):
        state = create_initial_state("Test query", max_iterations=5)
        assert state["max_iterations"] == 5
