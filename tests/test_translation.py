import sys
sys.path.append('../src')
from translators import translation,LLM_task
from srt_util.srt import SrtScript


def test_LLM_task():
    task = "如果输入中含有‘测试’则返回true，若没有则返回false"
    assert "true" in LLM_task.LLM_task("gpt-4", "测试", task, temp = 0.15),"LLM prompt 1 failed"
    assert "false" in LLM_task.LLM_task("gpt-4", "错误样例", task, temp = 0.15),"LLM prompt 2 failed"

def test_pmptsel():
    prompt = translation.prompt_selector("test1", "test2", "test3")
    assert "test1" in prompt,"prompt selector failed"
    assert "test2" in prompt,"prompt selector failed"
    assert "test3" in prompt,"prompt selector failed"

def test_translation_def():
    test1 = SrtScript.parse_from_srt_file("EN","ZH","./translation_test/test1.srt")
    translation.get_translation(test1, "gpt-4", video_name = "v-name")
    assert test1.segments[1].translation != "", "translation write in & empty prompt failed"

def test_translation_pmpt():
    test1 = SrtScript.parse_from_srt_file("EN","ZH","./translation_test/test1.srt")
    pmpt = translation.prompt_selector("EN", "ZH", "StarCraft2")
    translation.get_translation(test1, "gpt-4", "v-name", pmpt)
    task = "输入如果为中文，返回true,反之返回false"
    assert "true" in LLM_task.LLM_task("gpt-4", test1.segments[1].translation, task, temp = 0.15),"EN to ZH failed"

def test_translation_pmpt2():
    test1 = SrtScript.parse_from_srt_file("ZH","EN","./translation_test/test2.srt")
    pmpt = translation.prompt_selector("ZH", "EN", "StarCraft2")
    translation.get_translation(test1, "gpt-4", "v-name", pmpt)
    task = "输入如果为英文，返回true,反之返回false"
    assert "true" in LLM_task.LLM_task("gpt-4", test1.segments[1].translation, task, temp = 0.15),"ZH to EN failed"

def test_translation_pmpt3():
    test1 = SrtScript.parse_from_srt_file("ZH","ZH","./translation_test/test2.srt")
    pmpt = translation.prompt_selector("ZH", "ZH", "StarCraft2")
    translation.get_translation(test1, "gpt-4", "v-name", pmpt)
    task = "输入如果为中文，返回true,反之返回false"
    assert "true" in LLM_task.LLM_task("gpt-4", test1.segments[1].translation, task, temp = 0.15), "ZH to ZH failed"

test_pmptsel()
test_LLM_task()
test_translation_def()
test_translation_pmpt()
test_translation_pmpt2()
test_translation_pmpt3()