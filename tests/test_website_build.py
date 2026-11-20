"""The public site builds from the repository's documents, and every link inside it leads somewhere."""
import pytest

pytest.importorskip("markdown_it")

from website import build  # noqa: E402


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    return build.build(tmp_path_factory.mktemp("site"))


def test_no_link_inside_the_site_is_broken(site):
    assert build.broken(site) == []


def test_the_main_pages_and_the_replay_are_there(site):
    for page, _ in build.NAV:
        assert (site.out / page).is_file()
    assert (site.out / "demo" / "index.html").is_file()
    assert (site.out / "results" / "training-2026-11-15" / "README.html").is_file()


def test_every_page_carries_the_prototype_notice(site):
    for page in site.written:
        assert "not therapy and not a crisis service" in (site.out / page).read_text(encoding="utf-8")
