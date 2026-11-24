"""Run with: python -m pytest tests/"""
import pytest

from toygpt.tokenization.bpe import BPETokenizer, BYTE_CHARACTERS, pretokenize

CORPUS = [
    " = Valkyria Chronicles III = \n",
    " Senjō no Valkyria 3 : Unrecorded Chronicles is a tactical role @-@ playing video game .\n",
    " The game began development in 2010 , carrying over a large portion of the work .\n",
    " It's the third game in the series , and they'll release it in Japan .\n",
] * 20

ROUND_TRIP = [
    "",
    "The history of the game",
    " = Valkyria Chronicles III = \n",
    " café — 1€ 日本語 🙂\n",          # multi-byte chars never seen in training
    "  double  spaces\t\ttabs \n\n",  # whitespace must survive exactly
    "don't I'll we've",
]


@pytest.fixture(scope="module")
def tok():
    t = BPETokenizer(vocab_size=300)
    t.train(CORPUS)
    return t


@pytest.mark.parametrize("text", ROUND_TRIP)
def test_round_trip(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_vocab_layout(tok):
    assert tok.vocab[:256] == BYTE_CHARACTERS
    assert tok.vocab[32] == "Ġ" and tok.vocab[10] == "Ċ"
    assert sorted(tok.merges.values()) == list(range(256, 256 + len(tok.merges)))
    assert tok.eot_id == len(tok.vocab) - 1
    assert all(0 <= i < len(tok.vocab) for i in tok.encode(" café — 1€ 🙂"))


def test_encode_matches_training_split(tok):
    for word in ["ĠValkyria", "ĠChronicles", "Ġgame"]:
        text = word.replace("Ġ", " ")
        assert tok.encode(text) == tok.splits[word]


def test_merges_compress(tok):
    text = " The game began development"
    assert len(tok.encode(text)) < len(text.encode("utf-8"))


def test_special_token_is_one_id(tok):
    ids = tok.encode("end<|endoftext|> start")
    assert tok.eot_id in ids
    assert ids == tok.encode("end") + [tok.eot_id] + tok.encode(" start")
    assert tok.decode(ids) == "end<|endoftext|> start"


def test_pretokenize_byte_maps():
    assert pretokenize(" café\n") == ["ĠcafÃ©", "Ċ"]


def test_save_load(tok, tmp_path):
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.vocab == tok.vocab
    assert loaded.merges == tok.merges
    assert loaded.special_tokens == tok.special_tokens
    for text in ROUND_TRIP:
        assert loaded.encode(text) == tok.encode(text)
