#!/usr/bin/env python3
"""ポジティブ/ネガティブ判定ツール(形態素解析ベース)

仕組み:
  1. 形態素解析で文を単語に分け、各語の基本形(原形)を取り出す
  2. 基本形を感情辞書と照合して +1 / -1 を付ける
  3. 直後の「ない」「ません」などの否定表現で極性を反転する
  4. 合計スコアで ポジティブ / ネガティブ / 中立 を判定する

形態素解析器は fugashi(MeCab) -> janome -> sudachipy の順に自動で探し、
どれも無ければ内蔵の簡易トークナイザーで動く(精度は本物の解析器より低い)。

使い方:
  python3 positive_judge.py "今日はとても楽しかった!"
  python3 positive_judge.py -f input.txt          # 1行1文
  python3 positive_judge.py -v "嬉しくない"        # 形態素の一覧も表示
  python3 positive_judge.py --dict my_dict.tsv "..."  # 辞書を追加(語<TAB>スコア)
"""
import argparse
import gzip
import hashlib
import io
import os
import sys
import tarfile
import unicodedata
import urllib.parse
import urllib.request
import zipfile

# ---------------------------------------------------------------- 感情辞書
POS_NOUN = (
    "幸せ 幸福 感謝 成功 最高 素敵 笑顔 希望 愛 喜び 楽しみ 安心 元気 上手 好き 大好き "
    "綺麗 きれい 完璧 感動 魅力 勝利 満足 快適 平和 自由 優秀 祝福 誇り 幸運 充実 好調 "
    "親切 立派 見事 ありがとう おめでとう ハッピー ラッキー"
).split()
NEG_NOUN = (
    "不安 失敗 最悪 嫌い 嫌 悲しみ 苦痛 絶望 怒り 不満 後悔 憂鬱 疲れ 問題 危険 孤独 "
    "恐怖 地獄 不幸 災害 無理 迷惑 残念 退屈 不調 ストレス トラブル ミス"
).split()

POS_ADJ_I = (
    "嬉しい うれしい 楽しい たのしい 美しい 素晴らしい 優しい 面白い おいしい 美味しい "
    "頼もしい 明るい かわいい 可愛い すごい 凄い ありがたい 良い 愛しい 力強い 心地よい"
).split()
NEG_ADJ_I = (
    "悲しい 辛い つらい 苦しい 悪い 怖い 寂しい 淋しい ひどい 酷い 汚い 痛い 恐ろしい "
    "うるさい 暗い 悔しい"
).split()

# 活用しない固定表現(基本形, 極性)
POS_FIXED = ["いい"]
NEG_FIXED = ["つまらない", "くだらない", "ごめん"]
NEUTRAL_FIXED = ["いいえ"]  # 「いい」の誤検出を防ぐ

POS_GODAN = "喜ぶ 楽しむ 笑う 輝く 癒す 微笑む 弾む 好む".split()
POS_ICHIDAN = "褒める 恵まれる 満たされる 報われる 癒される".split()
NEG_GODAN = "悲しむ 泣く 怒る 嘆く 苦しむ 悩む 困る 失う 嫌う 憎む 恨む 焦る 落ち込む".split()
NEG_ICHIDAN = "疲れる".split()

# 否定表現(基本形・表層形)
NEG_BASES = {"ない", "無い", "ぬ", "ん", "ず", "ません", "なかっ", "なく"}

ROWS = {
    "う": "わいうえお", "く": "かきくけこ", "ぐ": "がぎぐげご", "す": "さしすせそ",
    "つ": "たちつてと", "ぬ": "なにぬねの", "ぶ": "ばびぶべぼ", "む": "まみむめも",
    "る": "らりるれろ",
}
EUPHONY = {"う": "っ", "つ": "っ", "る": "っ", "む": "ん", "ぶ": "ん", "ぬ": "ん",
           "く": "い", "ぐ": "い", "す": "し"}


def godan_forms(base):
    stem, last = base[:-1], base[-1]
    a, i, u, e, o = ROWS[last]
    return {stem + a, stem + i, base, stem + e, stem + o, stem + EUPHONY[last]}


def ichidan_forms(base):
    stem = base[:-1]
    return {stem, base, stem + "れ", stem + "よ"}


def adj_i_forms(base):
    stem = base[:-1]
    return {base, stem + "く", stem + "かっ", stem + "くて", stem + "ければ", stem + "さ", stem + "そう"}


def build_lexicon(extra=None):
    """polarity: 基本形 -> +1/-1,  surf_map: 表層形 -> (基本形, 品詞)"""
    polarity, surf = {}, {}

    def add(base, pol, pos, forms):
        if pol:
            polarity[base] = pol
        for f in forms:
            surf[f] = (base, pos)

    for w in POS_NOUN:
        add(w, 1, "名詞", [w])
    for w in NEG_NOUN:
        add(w, -1, "名詞", [w])
    for w in POS_ADJ_I:
        add(w, 1, "形容詞", adj_i_forms(w))
    for w in NEG_ADJ_I:
        add(w, -1, "形容詞", adj_i_forms(w))
    for w in POS_FIXED:
        add(w, 1, "形容詞", [w])
    for w in NEG_FIXED:
        add(w, -1, "形容詞", [w])
    for w in NEUTRAL_FIXED:
        add(w, 0, "感動詞", [w])
    for w in POS_GODAN:
        add(w, 1, "動詞", godan_forms(w))
    for w in POS_ICHIDAN:
        add(w, 1, "動詞", ichidan_forms(w))
    for w in NEG_GODAN:
        add(w, -1, "動詞", godan_forms(w))
    for w in NEG_ICHIDAN:
        add(w, -1, "動詞", ichidan_forms(w))
    # 否定の助動詞(簡易トークナイザー用)
    for w, b in [("ない", "ない"), ("なかっ", "ない"), ("なく", "ない"),
                 ("ません", "ません"), ("ませ", "ません")]:
        surf[w] = (b, "助動詞")
    # 外部辞書
    for word, score in (extra or {}).items():
        polarity[word] = 1 if score > 0 else -1
        surf[word] = (word, "名詞")
    return polarity, surf


# ---------------------------------------------------------------- 外部辞書(Web / ファイル)
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "positive_judge")

# 辞書を指定しないときに自動で取得する辞書(日本語評価極性辞書: 東北大学 乾・岡崎研究室)
# ※ 利用条件(出典の明記など)は配布元で確認すること
DEFAULT_DICT_URLS = [
    "https://www.cl.ecei.tohoku.ac.jp/resources/sent_lex/wago.121808.pn",
    "https://www.cl.ecei.tohoku.ac.jp/resources/sent_lex/pn.csv.m3.120408.trim",
]


def _sign(raw, min_score):
    """極性の表記(数値 / p / n / ポジ / ネガ)を +1 / -1 / 0 にそろえる"""
    raw = raw.strip()
    try:
        v = float(raw)
        return 0 if abs(v) < min_score else (1 if v > 0 else -1)
    except ValueError:
        pass
    low = raw.lower()
    if low.startswith("p") or raw.startswith("ポジ"):
        return 1
    if low.startswith("n") and not raw.startswith("ニュ") or raw.startswith("ネガ"):
        return -1
    return 0  # e / ニュートラル / 不明


def parse_dict_text(text, min_score=0.5):
    """対応形式(行ごとに自動判定):
      - 単語感情極性対応表風   語:読み:品詞:スコア
      - 日本語評価極性辞書(用言編)風  ポジ(…)<TAB>語 / ネガ(…)<TAB>語
      - 日本語評価極性辞書(名詞編)風  語<TAB>p|n|e<TAB>…
      - 単純TSV               語<TAB>スコア(数値 / p / n / ポジ / ネガ)
    ※ 各形式は想定のもので、配布元の実ファイルとの照合は利用者側で行うこと。
    """
    out = {}
    for line in text.splitlines():
        line = line.strip().lstrip("﻿")
        if not line or line.startswith("#"):
            continue
        word, sign = None, 0
        if "\t" in line:
            parts = [p.strip() for p in line.split("\t")]
            if len(parts) < 2:
                continue
            a, b = parts[0], parts[1]
            if a.startswith(("ポジ", "ネガ")):
                word, sign = b, _sign(a, min_score)
            else:
                word, sign = a, _sign(b, min_score)
        elif line.count(":") >= 3:
            parts = line.split(":")
            word = parts[0]
            try:
                sign = _sign(parts[-1], min_score)
            except Exception:
                continue
        else:
            continue
        word = word.replace(" ", "").replace("　", "")
        if word and sign:
            out[word] = sign
    return out


def _decode(data):
    for enc in ("utf-8-sig", "euc_jp", "cp932"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _iter_blobs(path):
    """ファイル / zip / tar(.gz) / gz から中身のバイト列を順に返す"""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if not name.endswith("/"):
                    yield name, zf.read(name)
        return
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as tf:
            for m in tf.getmembers():
                if m.isfile():
                    yield m.name, tf.extractfile(m).read()
        return
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    yield os.path.basename(path), data


def load_dict_file(path, min_score=0.5):
    merged = {}
    for name, data in _iter_blobs(path):
        low = name.lower()
        if "readme" in low or low.endswith((".pdf", ".html", ".md")):
            continue
        merged.update(parse_dict_text(_decode(data), min_score))
    return merged


def fetch_dict(url, refresh=False):
    """URLから辞書を取得して ~/.cache/positive_judge/ に保存(2回目以降はキャッシュを使う)"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    base = os.path.basename(urllib.parse.urlparse(url).path) or "dict"
    cached = os.path.join(CACHE_DIR, hashlib.sha1(url.encode()).hexdigest()[:12] + "_" + base)
    if refresh or not os.path.exists(cached):
        req = urllib.request.Request(url, headers={"User-Agent": "positive_judge/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
        except Exception as exc:
            raise RuntimeError(f"辞書を取得できませんでした: {url}\n  理由: {exc}") from exc
        with open(cached, "wb") as fh:
            fh.write(data)
    return cached


def load_extra_dict(path, min_score=0.5):
    """後方互換: ローカルファイルから辞書を読む"""
    return load_dict_file(path, min_score)


# ---------------------------------------------------------------- 内蔵トークナイザー
def char_type(c):
    if "぀" <= c <= "ゟ":
        return "hira"
    if "゠" <= c <= "ヿ":
        return "kata"
    if "一" <= c <= "鿿" or c in "々〆":
        return "kanji"
    if c.isdigit():
        return "num"
    if c.isalpha():
        return "alpha"
    if c.isspace():
        return "space"
    return "punct"


def builtin_tokenize(text, surf):
    """辞書の最長一致 + 文字種で区切る簡易分かち書き(本物の形態素解析ではない)"""
    text = unicodedata.normalize("NFKC", text)
    maxlen = max(len(s) for s in surf)
    out, buf, buftype, i, n = [], "", None, 0, len(text)

    def flush():
        nonlocal buf, buftype
        if buf:
            out.append((buf, buf, "未知語"))
        buf, buftype = "", None

    while i < n:
        hit = None
        for L in range(min(maxlen, n - i), 0, -1):
            seg = text[i:i + L]
            if seg in surf:
                hit = seg
                break
        if hit:
            flush()
            base, pos = surf[hit]
            out.append((hit, base, pos))
            i += len(hit)
            continue
        c = text[i]
        t = char_type(c)
        if t == "space":
            flush()
        elif t == "punct":
            flush()
            out.append((c, c, "記号"))
        else:
            if buftype is not None and t != buftype:
                flush()
            buf += c
            buftype = t
        i += 1
    flush()
    return out


# ---------------------------------------------------------------- 形態素解析バックエンド
def make_backend(name, surf):
    order = ["fugashi", "janome", "sudachi", "builtin"] if name == "auto" else [name]
    for n in order:
        try:
            if n == "fugashi":
                import fugashi
                tagger = fugashi.Tagger()

                def tok(text, tagger=tagger):
                    res = []
                    for w in tagger(text):
                        f = w.feature
                        lemma = (getattr(f, "lemma", None) or w.surface).split("-")[0]
                        res.append((w.surface, lemma, getattr(f, "pos1", "") or ""))
                    return res
                return "fugashi (MeCab)", tok
            if n == "janome":
                from janome.tokenizer import Tokenizer
                jt = Tokenizer()

                def tok(text, jt=jt):
                    return [(t.surface,
                             t.surface if t.base_form == "*" else t.base_form,
                             t.part_of_speech.split(",")[0]) for t in jt.tokenize(text)]
                return "janome", tok
            if n == "sudachi":
                from sudachipy import dictionary, tokenizer as st
                sd = dictionary.Dictionary().create()
                mode = st.Tokenizer.SplitMode.C

                def tok(text, sd=sd, mode=mode):
                    return [(m.surface(), m.dictionary_form(), m.part_of_speech()[0])
                            for m in sd.tokenize(text, mode)]
                return "sudachipy", tok
            if n == "builtin":
                return "builtin (簡易トークナイザー)", lambda text: builtin_tokenize(text, surf)
        except Exception as exc:  # 未インストール・辞書なしなど
            if name != "auto":
                sys.exit(f"バックエンド {n} を使えません: {exc}")
    sys.exit("使えるバックエンドがありません")


# ---------------------------------------------------------------- 判定
def is_short_hira(s):
    return len(s) <= 2 and all(char_type(c) == "hira" for c in s)


# 節の区切り(ここより先の否定は、前の語にはかけない)
CLAUSE_BREAKS = {"が", "けど", "けれど", "けれども", "のに", "ので", "から", "ものの"}
# 「楽しいとは言えない」「嬉しいとは思わない」のように、否定との間に挟まる語
HEDGE_BASES = {"言える", "言う", "思う", "思える", "感じる", "見える", "いえる"}
# 「ないわけではない」「ないことはない」の二重否定をつなぐ語
FILLERS = {"わけ", "訳", "こと"}


def _is_neg(tok):
    surface, base, _ = tok
    return base in NEG_BASES or surface in NEG_BASES


def count_negations(tokens, idx):
    """極性語(tokens[idx])の後ろ、同じ節の中にある否定の数を数える"""
    n, j, negs, gap = len(tokens), idx + 1, 0, 0
    while j < n:
        surface, base, _ = tokens[j]
        if not surface or surface in CLAUSE_BREAKS or char_type(surface[0]) == "punct":
            break
        if _is_neg(tokens[j]):
            negs += 1
            gap = 0
        elif is_short_hira(surface) or base in HEDGE_BASES or surface in FILLERS or base in FILLERS:
            gap += 1
            if gap > 3:
                break
        else:
            break
        j += 1
    return negs


# 逆接: この後ろに言いたいことが来ることが多い(「期待していたのに、がっかりした」)
CONTRAST_WORDS = {"けど", "けれど", "けれども", "のに", "ものの"}


def last_contrast_index(tokens):
    """最後の逆接の位置を返す(なければ -1)。「が」は主語の「が」と区別する"""
    last = -1
    for i, (surface, _base, _pos) in enumerate(tokens):
        if surface in CONTRAST_WORDS:
            last = i
        elif surface == "が":
            prev_pos = tokens[i - 1][2] if i > 0 else ""
            nxt = tokens[i + 1][0] if i + 1 < len(tokens) else ""
            if prev_pos in ("動詞", "形容詞", "助動詞") or nxt == "、":
                last = i
    return last


def judge(tokens, polarity, contrast_weight=2):
    hits, total = [], 0
    last = last_contrast_index(tokens) if contrast_weight > 1 else -1
    for idx, (surface, base, _pos) in enumerate(tokens):
        pol = polarity.get(base) or polarity.get(surface)
        if not pol:
            continue
        negs = count_negations(tokens, idx)
        weight = contrast_weight if last >= 0 and idx > last else 1
        value = (-pol if negs % 2 == 1 else pol) * weight  # 否定が奇数回なら反転、偶数回なら元のまま
        total += value
        hits.append((surface, base, value, negs, weight))
    pos_n = sum(1 for h in hits if h[2] > 0)
    neg_n = sum(1 for h in hits if h[2] < 0)
    if not hits:
        label = "判定不能(感情語なし)"
    elif pos_n and neg_n and total == 0:
        label = "混在(ポジ・ネガ両方あり)"
    elif total > 0:
        label = "ポジティブ"
    elif total < 0:
        label = "ネガティブ"
    else:
        label = "中立"
    return label, total, hits


def report(text, backend_name, tokenize, polarity, verbose, contrast_weight=2):
    tokens = tokenize(text)
    label, total, hits = judge(tokens, polarity, contrast_weight)
    print(f"入力: {text}")
    pos_n = sum(1 for h in hits if h[2] > 0)
    neg_n = sum(1 for h in hits if h[2] < 0)
    print(f"結果: {label}  (スコア {total:+d} / ポジ語 {pos_n}・ネガ語 {neg_n})")
    for surface, base, value, negs, weight in hits:
        mark = "+" if value > 0 else "-"
        note = "" if negs == 0 else (" ※否定で反転" if negs % 2 else " ※二重否定(元の極性)")
        if weight > 1:
            note += f" ※逆接の後ろ ×{weight}"
        shown = surface if surface == base else f"{surface} → {base}"
        print(f"  {mark} {shown}{note}")
    if verbose:
        print("  [形態素]")
        for surface, base, pos in tokens:
            print(f"    {surface}\t{base}\t{pos}")
    print()


def main():
    ap = argparse.ArgumentParser(description="形態素解析によるポジティブ判定")
    ap.add_argument("texts", nargs="*", help="判定する文")
    ap.add_argument("-f", "--file", help="テキストファイル(1行1文)")
    ap.add_argument("--dict", action="append", default=[],
                    help="追加の感情辞書ファイル(zip/tar.gz可、複数指定可)")
    ap.add_argument("--dict-url", action="append", default=[],
                    help="Webの感情辞書のURL(複数指定可。~/.cache/positive_judge/ に保存)")
    ap.add_argument("--refresh", action="store_true", help="キャッシュを使わず辞書を再取得")
    ap.add_argument("--no-default-dict", action="store_true",
                    help="既定の辞書(日本語評価極性辞書)を自動取得せず、内蔵辞書だけを使う")
    ap.add_argument("--contrast-weight", type=int, default=2,
                    help="逆接(けど・のに・が)の後ろの感情語の重み(既定 2、1で無効)")
    ap.add_argument("--ignore", action="append", default=[],
                    help="判定から外す語(複数指定可。例: --ignore サービス)")
    ap.add_argument("--min-score", type=float, default=0.5,
                    help="数値スコアの辞書で、この絶対値未満の語は無視(既定 0.5)")
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "fugashi", "janome", "sudachi", "builtin"])
    ap.add_argument("-v", "--verbose", action="store_true", help="形態素の一覧も表示")
    args = ap.parse_args()

    texts = list(args.texts)
    if args.file:
        with open(args.file, encoding="utf-8-sig") as fh:  # BOM付きUTF-8も読める
            texts += [l.strip() for l in fh if l.strip()]
    if not texts and not sys.stdin.isatty():
        texts = [l.strip() for l in sys.stdin if l.strip()]
    if not texts:
        ap.print_help()
        return

    # 辞書: 指定が無ければ既定の辞書を自動取得(初回のみ通信。以後は保存したものを使う)
    extra = {}
    use_default = not (args.dict or args.dict_url or args.no_default_dict)
    urls = list(DEFAULT_DICT_URLS) if use_default else list(args.dict_url)
    for url in urls:
        try:
            extra.update(load_dict_file(fetch_dict(url, args.refresh), args.min_score))
        except RuntimeError as exc:
            if use_default:
                print(f"※ 既定の辞書を取得できなかったため、内蔵辞書だけで判定します。\n  {exc}",
                      file=sys.stderr)
                extra = {}
                break
            sys.exit(str(exc))
    for path in args.dict:
        extra.update(load_dict_file(path, args.min_score))
    if extra:
        print(f"外部辞書: {len(extra)} 語を読み込みました")
    polarity, surf = build_lexicon(extra)
    for word in args.ignore:
        polarity.pop(word, None)
    name, tokenize = make_backend(args.backend, surf)

    print(f"バックエンド: {name}")
    if name.startswith("builtin"):
        print("※ 形態素解析器が未インストールのため簡易トークナイザーで動作中です。")
    print()
    for t in texts:
        report(t, name, tokenize, polarity, args.verbose, args.contrast_weight)


if __name__ == "__main__":
    main()
