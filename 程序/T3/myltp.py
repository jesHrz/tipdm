import sys, os
from pyltp import SentenceSplitter, Segmentor, Postagger, Parser, NamedEntityRecognizer, SementicRoleLabeller

class MyLTP():
    def __init__(self):
        ROOTDIR = os.path.join(os.path.dirname(__file__), os.pardir)
        # sys.path = [os.path.join(ROOTDIR, "lib")] + sys.path
        # Set your own model path
        self.MODELDIR=os.path.join(ROOTDIR, "./ltp_data")
        # Init LTP Model
        self.segmentor = Segmentor()
        self.postagger = Postagger()
        self.parser = Parser()
        self.recognizer = NamedEntityRecognizer()
        self.labeller = SementicRoleLabeller()
        self.segmentor.load(os.path.join(self.MODELDIR, "cws.model"))
        self.postagger.load(os.path.join(self.MODELDIR, "pos.model"))
        self.parser.load(os.path.join(self.MODELDIR, "parser.model"))
        self.recognizer.load(os.path.join(self.MODELDIR, "ner.model"))
        self.labeller.load(os.path.join(self.MODELDIR, "pisrl.model"))

    # 下述函数返回值均为 list, list[0] 为第一个句子的运行结果
    # ---------------------------- 分词 -------------------------------
    def MySegmentor(self, paragraph):
        # 段落分成句子
        sentences = SentenceSplitter.split(paragraph)
        result = []
        for sentence in sentences:
            words = self.segmentor.segment(sentence)
            # 输出
            # print("\t".join(words))
            result.append(words)
        return result

    # ---------------------------- 词性标注 -------------------------------
    def MyPostagger(self, words):
        result = []
        for word in words:
            postags = self.postagger.postag(word)
            # list-of-string parameter is support in 0.1.5
            # postags = postagger.postag(["中国","进出口","银行","与","中国银行","加强","合作"])
            # 输出
            # print("\t".join(postags))
            result.append(postags)
        return result

    # ---------------------------- 依存句法分析 -------------------------------
    def MyParser(self, words, postags):
        result = []
        for index in range(0, len(words)):
            arcs = self.parser.parse(words[index], postags[index])
            # 输出
            # print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
            result.append(arcs)
        return result

    # ---------------------------- 命名实体识别 -------------------------------
    def MyRecognizer(self, words, postags):
        result = []
        for index in range(0, len(words)):
            netags = self.recognizer.recognize(words[index], postags[index])
            # 输出
            # print("\t".join(netags))
            result.append(netags)
        return result

    # ---------------------------- 语义角色标注 -------------------------------
    def MyRoleLabller(self, words, postags, arcs):
        result = []
        for index in range(0, len(words)):
            roles = self.labeller.label(words[index], postags[index], arcs[index])
            # 输出
            # for role in roles:
            #     print(role.index, "".join(
            #             ["%s:(%d,%d)" % (arg.name, arg.range.start, arg.range.end) for arg in role.arguments]))
            result.append(roles)
        return result

if __name__ == "__main__":
    # 创建对象
    Myltp = MyLTP()
    # 语句
    paragraph = '中国进出口银行与中国银行加强合作。中国进出口银行与中国银行加强合作！'
    # 分词结果
    words = Myltp.MySegmentor(paragraph)
    # 词性标注
    postags = Myltp.MyPostagger(words)
    # 依存句法分析
    arcs = Myltp.MyParser(words, postags)
    # 命名实体识别
    netags = Myltp.MyRecognizer(words, postags)
    # 语义角色标注
    roles = Myltp.MyRoleLabller(words, postags, arcs)