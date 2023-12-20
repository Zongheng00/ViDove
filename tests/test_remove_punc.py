import sys
sys.path.append('./src')
from srt_util.srt import SrtScript, SrtSegment

zh_test1 = "再次，如果你对一些福利感兴趣，你也可以。"
zh_en_test1 = "GG。Classic在我今年解说的最奇葩的系列赛中获得了胜利。"

def form_srt_class(src_lang, tgt_lang, source_text="", translation="", duration="00:00:00,740 --> 00:00:08,779"):
    segment = [0, duration, source_text, translation, ""]
    return SrtScript(src_lang, tgt_lang, [segment])

def test_zh():
    srt = form_srt_class(src_lang="EN", tgt_lang="ZH", translation=zh_test1)
    srt.remove_trans_punctuation()
    assert srt.segments[0].translation == "再次 如果你对一些福利感兴趣 你也可以 "

def test_zh_en():
    srt = form_srt_class(src_lang="EN", tgt_lang="ZH", translation=zh_en_test1)
    srt.remove_trans_punctuation()
    assert srt.segments[0].translation == "GG Classic在我今年解说的最奇葩的系列赛中获得了胜利 "

