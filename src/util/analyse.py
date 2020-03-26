import jieba


class WordStriper(object):
    def __init__(self, stop_word_file="data/stop_words.txt"):
        self._stop_word_file = stop_word_file
        with open(self._stop_word_file, "r") as f:
            self._stop_words = f.read().split("\n")

    def strip(self, text, HMM=True, cut_all=False):
        _text = text.replace("\n", "").replace("\t", "")
        for stop_word in self._stop_words:
            _text = _text.replace(stop_word, " ")
        # print(text)
        print(_text)
        seg_list = jieba.cut(_text, HMM=HMM, cut_all=cut_all)
        # word_list = []
        # for seg in seg_list:
        #     if seg.strip() not in self._stop_words:
        #         word_list.append(seg.strip())
        # print(list(seg_list))
        return list(seg_list)
