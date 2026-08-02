"""The spot-position model: ordinals 1 to 5, L for last, and the counting methods.

Ground truth is docs/media-domain-from-the-trade.md, section "Positions: the
product is wrong today". These tests hold three lines:

  * the vocabulary is 1, 2, 3, 4, 5 and L, and L is its own position rather than
    the fifth ordinal;
  * extending the vocabulary moved no money: the shipped rate card prices every
    (position, break size) pair exactly as the previous derived rule did;
  * a preferred-position percentage always names the counting method it used,
    and refuses to exist at all while the preferred set is unset.
"""

from __future__ import annotations

import pytest

from kairos.optimize.positions import (
    AGENCY_METHOD,
    CHANNEL_METHOD,
    GOLD_POSITION,
    LAST_TOKEN,
    MIDDLE_TOKEN,
    POSITION_TOKENS,
    Appearance,
    canonical_token,
    normalize_position_scope,
    occupied_tokens,
    parse_preferred,
    preferred_position_rate,
    premium_token,
    position_options,
    resolve_preferred,
)
from kairos.optimize.pricing import PricingModel

# The rate card as it stood before positions 4, 5 and L became addressable, and
# the exact rule that priced against it. The parity test below replays it.
_LEGACY_TABLE = {1: 1.30, 2: 1.15, 3: 1.05, "default_middle": 1.00, "last": 1.20}


def _legacy_position_premium(position: int, break_size: int) -> float:
    if position < 1:
        raise ValueError("position must be >= 1")
    if position in _LEGACY_TABLE:
        return float(_LEGACY_TABLE[position])
    if position == break_size and position > 3:
        return float(_LEGACY_TABLE["last"])
    return float(_LEGACY_TABLE["default_middle"])


def test_vocabulary_is_one_to_five_plus_l() -> None:
    assert POSITION_TOKENS == ("1", "2", "3", "4", "5", LAST_TOKEN)
    assert LAST_TOKEN == "L"
    # L is its own position, so it is not the fifth ordinal under another name.
    assert LAST_TOKEN != "5"
    keys = [option["key"] for option in position_options()]
    assert keys == ["1", "2", "3", "4", "5", "L", GOLD_POSITION]
    assert all(option["he"] and option["en"] for option in position_options())


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("first", "1"), ("Second", "2"), ("third", "3"), ("fourth", "4"),
        ("fifth", "5"), ("last", "L"), ("l", "L"), ("L", "L"),
        ("אחרון", "L"), ("ראשון", "1"), (2, "2"), ("2", "2"),
        (GOLD_POSITION, GOLD_POSITION), ("", None), (None, None),
    ],
)
def test_canonical_token_folds_every_spelling(raw, expected) -> None:
    assert canonical_token(raw) == expected


def test_normalize_position_scope_gives_one_vocabulary() -> None:
    assert normalize_position_scope("first,last") == "1,L"
    assert normalize_position_scope("ANY") == "ANY"
    assert normalize_position_scope("") == "ANY"
    assert normalize_position_scope("1, אחרון") == "1,L"
    assert normalize_position_scope(GOLD_POSITION) == GOLD_POSITION


def test_a_spot_can_hold_an_ordinal_and_l_at_once() -> None:
    """A break with three spots has a first and a last; the third is both."""
    assert occupied_tokens(3, 3) == ("3", "L")
    assert occupied_tokens(1, 1) == ("1", "L")
    assert occupied_tokens(1, 3) == ("1",)
    assert occupied_tokens(4, 7) == ("4",)
    # Without a break size, L cannot be asserted and is not guessed.
    assert occupied_tokens(3, None) == ("3",)


def test_premium_token_prefers_a_priced_ordinal_then_l_then_middle() -> None:
    priced = {"1", "2", "3", MIDDLE_TOKEN, LAST_TOKEN}
    assert premium_token(1, 5, priced) == "1"
    assert premium_token(3, 3, priced) == "3"        # a priced ordinal wins
    assert premium_token(5, 5, priced) == LAST_TOKEN  # the tail is L
    assert premium_token(4, 9, priced) == MIDDLE_TOKEN
    # Pricing position 4 makes it win at 4, and only at 4.
    assert premium_token(4, 4, {*priced, "4"}) == "4"
    assert premium_token(5, 5, {*priced, "4"}) == LAST_TOKEN
    with pytest.raises(ValueError):
        premium_token(0, 3, priced)


def test_shipped_rate_card_prices_every_slot_exactly_as_before() -> None:
    """Extending the vocabulary must move no money on the shipped rate card.

    Positions 4 and 5 are addressable but UNSET, which is the no-op: every
    (position, break size) pair still resolves to the same multiplier the
    previous derived rule produced.
    """
    model = PricingModel.from_yaml()
    for break_size in range(1, 13):
        for position in range(1, break_size + 1):
            assert model.position_premium(position, break_size) == pytest.approx(
                _legacy_position_premium(position, break_size)
            ), f"position {position} of {break_size} moved"


def test_pricing_position_four_is_a_deliberate_change_not_a_default() -> None:
    model = PricingModel.from_yaml()
    assert model.position_key(4, 4) == LAST_TOKEN
    assert model.position_premium(4, 4) == 1.20
    edited = PricingModel.from_config({"premiums": {"position_in_break": {"4": 1.4}}})
    assert edited.position_key(4, 4) == "4"
    assert edited.position_premium(4, 4) == 1.4
    # Pricing the fourth position prices it everywhere, including mid-break,
    # which is the point of making it addressable.
    assert edited.position_key(4, 9) == "4"
    assert edited.position_premium(4, 9) == 1.4
    # Every other slot is untouched by that one edit.
    assert edited.position_premium(5, 5) == 1.20
    assert edited.position_premium(5, 9) == 1.00


def test_legacy_last_key_still_reads_as_l() -> None:
    model = PricingModel.from_weights(
        {"premiums": {"position_in_break": {1: 1.4, "last": 1.25, "default_middle": 1.0}}}
    )
    assert model.position_premiums["L"] == 1.25
    assert model.position_premium(1, 2) == 1.4


def test_preferred_set_is_tri_state_and_unset_by_default() -> None:
    assert parse_preferred(None) is None
    assert parse_preferred("") is None
    assert parse_preferred([]) == frozenset()
    assert parse_preferred("first,last") == frozenset({"1", "L"})
    model = PricingModel.from_yaml()
    assert model.preferred_positions() == (None, "unset")


def test_preferred_set_resolves_agreement_then_client_then_channel() -> None:
    per_advertiser = {"ADV_02": "1,2"}
    assert resolve_preferred(
        agreement="L", per_advertiser=per_advertiser, advertiser="ADV_02",
        channel_default="1,2,3",
    ) == (frozenset({"L"}), "agreement")
    assert resolve_preferred(
        per_advertiser=per_advertiser, advertiser="ADV_02", channel_default="1,2,3",
    ) == (frozenset({"1", "2"}), "advertiser")
    assert resolve_preferred(
        per_advertiser=per_advertiser, advertiser="ADV_09", channel_default="1,2,3",
    ) == (frozenset({"1", "2", "3"}), "channel_default")
    assert resolve_preferred() == (None, "unset")


def test_preferred_set_is_configurable_through_the_rate_card() -> None:
    model = PricingModel.from_config({
        "preferred_positions": {
            "channel_default": "1,2,L",
            "per_advertiser": {"ADV_02": "first,last"},
        }
    })
    assert model.preferred_positions() == (frozenset({"1", "2", "L"}), "channel_default")
    assert model.preferred_positions("ADV_02") == (frozenset({"1", "L"}), "advertiser")
    assert model.preferred_positions("ADV_02", agreement="3") == (frozenset({"3"}), "agreement")


def test_a_percentage_refuses_to_exist_while_the_preferred_set_is_unset() -> None:
    rows = [Appearance("b1", 1, 4), Appearance("b1", 4, 4)]
    for method in (AGENCY_METHOD, CHANNEL_METHOD):
        rate = preferred_position_rate(rows, None, method)
        assert rate.percent is None
        assert rate.basis == "unset"
        # Even a refusal names the method it would have used.
        assert rate.method == method
        assert rate.method_label_he and rate.method_label_en


def test_every_percentage_states_its_counting_method() -> None:
    rows = [Appearance("b1", 1, 4), Appearance("b1", 4, 4), Appearance("b2", 2, 5)]
    agency = preferred_position_rate(rows, ["1", "L"], AGENCY_METHOD)
    channel = preferred_position_rate(rows, ["1", "L"], CHANNEL_METHOD)
    assert agency.method == AGENCY_METHOD
    assert "out of breaks appeared in" in agency.method_label_en
    assert channel.method == CHANNEL_METHOD
    assert "out of total broadcasts" in channel.method_label_en
    # Top and tail of one break: two of three broadcasts hold a preferred slot.
    assert (agency.numerator, agency.denominator) == (2, 3)
    assert (channel.numerator, channel.denominator) == (2, 3)
    assert agency.breaks_appeared_in == 2
    assert agency.basis == "real"


def test_the_two_methods_diverge_when_one_broadcast_holds_two_preferred_slots() -> None:
    """A one-spot break is both the top and the tail, so the numerators differ.

    The agency method counts the preferred POSITIONS obtained, so that single
    broadcast obtained two. The channel method counts BROADCASTS, so it obtained
    one. This is the case the trade calls contested, and it is exactly why the
    method has to be printed next to the number.
    """
    rows = [Appearance("b1", 1, 1), Appearance("b2", 3, 6)]
    agency = preferred_position_rate(rows, ["1", "L"], AGENCY_METHOD)
    channel = preferred_position_rate(rows, ["1", "L"], CHANNEL_METHOD)
    assert (agency.numerator, agency.denominator, agency.percent) == (2, 2, 100.0)
    assert (channel.numerator, channel.denominator, channel.percent) == (1, 2, 50.0)
    assert agency.percent != channel.percent


def test_an_unknown_method_is_refused_rather_than_guessed() -> None:
    with pytest.raises(ValueError):
        preferred_position_rate([], ["1"], "whatever")


def test_no_appearances_is_unavailable_not_zero_percent() -> None:
    rate = preferred_position_rate([], ["1", "L"], CHANNEL_METHOD)
    assert rate.percent is None
    assert rate.basis == "unavailable"
