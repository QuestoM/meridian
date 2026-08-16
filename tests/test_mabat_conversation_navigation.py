"""Mabat keeps conversation navigation with chat and reserves actions for actions."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "tv-break-dashboard" / "src" / "kai"


def test_conversation_tab_owns_new_and_history_controls():
    panel = (ROOT / "AssistantPanel.jsx").read_text(encoding="utf-8")
    toolbar = (ROOT / "AssistantConversationToolbar.jsx").read_text(encoding="utf-8")
    surface = panel + toolbar

    assert "<AssistantConversationsRail" in panel
    assert "onNew={startConversation}" in panel
    assert "onToggleHistory={toggleHistory}" in panel
    assert "aria-controls=\"assistant-conversation-history\"" in surface
    assert "'Conversation history', 'היסטוריית שיחות'" in surface
    assert "hidden={!historyOpen}" in panel


def test_actions_tab_contains_no_conversation_list():
    panel = (ROOT / "AssistantPanel.jsx").read_text(encoding="utf-8")
    actions = (ROOT / "AssistantConversationsSidebar.jsx").read_text(encoding="utf-8")

    assert "'Actions', 'פעולות'" in panel
    assert "assistant-dock-tab-actions" in panel
    assert "assistant-dock-panel-actions" in actions
    assert "AssistantConversationsRail" not in actions
    assert "<AssistantConversationsRail" not in actions
    assert "'Applied actions', 'פעולות שבוצעו'" in actions


def test_picking_history_returns_to_the_conversation_thread():
    panel = (ROOT / "AssistantPanel.jsx").read_text(encoding="utf-8")
    history = (ROOT / "AssistantConversationsRail.jsx").read_text(encoding="utf-8")

    assert "onSelect={returnFromHistory}" in panel
    assert "conv.select(id); if (onSelect) onSelect(id);" in history
    assert "aria-current={conv.activeId === id ? 'true' : undefined}" in history
    assert "threadRef.current?.focus({ preventScroll: true })" in panel


def test_new_conversation_only_closes_history_after_the_server_creates_it():
    panel = (ROOT / "AssistantPanel.jsx").read_text(encoding="utf-8")
    api = (ROOT / "AssistantConversationsApi.jsx").read_text(encoding="utf-8")

    assert "const createdId = await conv.create();" in panel
    assert "if (!createdId) return;" in panel
    assert "return createdId;" in api
    assert "return null;" in api
    assert "setQuestion('');" in panel
    assert "setRefs([]);" in panel


def test_history_retry_remains_reachable_after_an_index_error():
    toolbar = (ROOT / "AssistantConversationToolbar.jsx").read_text(encoding="utf-8")
    history = (ROOT / "AssistantConversationsRail.jsx").read_text(encoding="utf-8")

    assert "listState !== 'error'" in toolbar
    assert "onClick={conv.refreshList}" in history
