
from core.memory import ConversationStore
from core.llm import LLMRouter

class Dummy(LLMRouter):
    def __init__(self): pass
    def complete(self, prompt: str, system=None): return "summary"

def test_roundtrip(tmp_path):
    db = tmp_path / "conv.sqlite"
    mem = ConversationStore(str(db))
    sid = mem.start_session("u")
    mem.append_message(sid, "user", "hello")
    mem.append_message(sid, "assistant", "world")
    assert mem.get_messages(sid)[0]["content"] == "hello"
    assert isinstance(mem.summarize(sid, Dummy()), str)
