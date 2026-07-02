from __future__ import annotations

from string import ascii_lowercase, digits

from hypothesis import given, settings, strategies as st

from monoid_agent_kernel.core.scope import domain_patterns_within, scope_within


ATOM = st.text(alphabet=ascii_lowercase + digits, min_size=1, max_size=8)
LABEL = st.text(alphabet=ascii_lowercase + digits, min_size=1, max_size=8)
SMALL_INT = st.integers(min_value=0, max_value=1_000_000)
PROPERTY_SETTINGS = settings(max_examples=40, deadline=None)


@st.composite
def domain_names(draw, *, min_labels: int = 2, max_labels: int = 4) -> str:
    labels = draw(st.lists(LABEL, min_size=min_labels, max_size=max_labels))
    return ".".join(labels)


DOMAIN_PATTERN = st.one_of(
    domain_names(),
    domain_names().map(lambda domain: f"*.{domain}"),
    st.just("*"),
)


@st.composite
def scopes(draw) -> dict[str, object]:
    scope: dict[str, object] = {}
    if draw(st.booleans()):
        scope["max_results"] = draw(SMALL_INT)
    if draw(st.booleans()):
        scope["features"] = draw(st.lists(ATOM, unique=True, max_size=5))
    if draw(st.booleans()):
        scope["region"] = draw(ATOM)
    if draw(st.booleans()):
        scope["allowed_domains"] = draw(st.lists(DOMAIN_PATTERN, unique=True, max_size=4))
    return scope


@st.composite
def list_narrowing_cases(draw) -> tuple[list[str], list[str], str]:
    outer = draw(st.sets(ATOM, max_size=6))
    if outer:
        subset = draw(st.sets(st.sampled_from(sorted(outer)), max_size=len(outer)))
    else:
        subset = set()
    extra = f"extra_{draw(ATOM)}"
    return sorted(subset), sorted(outer), extra


@st.composite
def domain_pattern_chains(draw) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    root = draw(domain_names(min_labels=2, max_labels=2))
    branch = f"{draw(LABEL)}.{root}"
    leaf = f"{draw(LABEL)}.{branch}"
    requested, middle, signed = draw(
        st.sampled_from(
            (
                (leaf, f"*.{branch}", f"*.{root}"),
                (f"*.{leaf}", f"*.{branch}", f"*.{root}"),
                (leaf, f"*.{branch}", "*"),
                (branch, f"*.{root}", "*"),
                (f"*.{branch}", f"*.{root}", "*"),
            )
        )
    )
    return (requested,), (middle,), (signed,)


@PROPERTY_SETTINGS
@given(scopes())
def test_scope_within_is_reflexive(scope: dict[str, object]) -> None:
    assert scope_within(scope, scope)


@PROPERTY_SETTINGS
@given(cap=SMALL_INT, slack=SMALL_INT)
def test_scope_within_numeric_caps_narrow_by_smaller_values(cap: int, slack: int) -> None:
    broader = cap + slack

    assert scope_within({"max_results": cap}, {"max_results": broader})
    if slack > 0:
        assert not scope_within({"max_results": broader}, {"max_results": cap})


@PROPERTY_SETTINGS
@given(list_narrowing_cases())
def test_scope_within_list_caps_narrow_by_subset(case: tuple[list[str], list[str], str]) -> None:
    subset, outer, extra = case

    assert scope_within({"features": subset}, {"features": outer})
    assert not scope_within({"features": [*outer, extra]}, {"features": outer})


@PROPERTY_SETTINGS
@given(value=ATOM, other=ATOM)
def test_scope_within_scalars_narrow_by_equality(value: str, other: str) -> None:
    different = f"other_{other}"

    assert scope_within({"region": value}, {"region": value})
    assert not scope_within({"region": different}, {"region": value})


@PROPERTY_SETTINGS
@given(st.lists(DOMAIN_PATTERN, unique=True, max_size=4).map(tuple))
def test_domain_patterns_within_is_reflexive(patterns: tuple[str, ...]) -> None:
    assert domain_patterns_within(patterns, patterns)


@PROPERTY_SETTINGS
@given(domain_pattern_chains())
def test_domain_patterns_within_is_transitive_for_wildcard_chains(
    chain: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]],
) -> None:
    requested, middle, signed = chain

    assert domain_patterns_within(requested, middle)
    assert domain_patterns_within(middle, signed)
    assert domain_patterns_within(requested, signed)
