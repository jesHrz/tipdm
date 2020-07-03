import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer,f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

all_classes: list = list()


# CountVectorizer 词频 + 多项式朴素贝叶斯 分类
# messages: list 留言信息
# labels: list 留言的分类
# ratio: float 测试集比例
# normal: 归一化
# 0.88
def count_train(messages: list, labels: list, ratio: float, normal: bool) -> None:
    message_train, message_test, label_train, label_test = train_test_split(messages, labels, test_size=ratio)
    count_vec = CountVectorizer()
    message_train = count_vec.fit_transform(message_train)
    message_test = count_vec.transform(message_test)

    print('count + NB')

    if normal == True:
        scaler = preprocessing.StandardScaler(with_mean=False).fit(message_train)
        message_train=scaler.transform(message_train)
        message_test=scaler.transform(message_test)

    clf = MultinomialNB()
    clf.fit(message_train, label_train)

    label_test_predict = clf.predict(message_test)

    print("测试集大小:", len(label_test), "得分:", clf.score(message_test, label_test))
    print(classification_report(label_test, label_test_predict, target_names=all_classes, zero_division="warn"))


# tfidf + 多项式朴素贝叶斯 分类
# messages: list 留言信息
# labels: list 留言的分类
# ratio: float 测试集比例
# normal: 归一化
# 0.72
def tfidf_train(messages: list, labels: list, ratio: float, normal: bool) -> None:
    message_train, message_test, label_train, label_test = train_test_split(messages, labels, test_size=ratio)
    tfidf_vec = TfidfVectorizer()
    message_train = tfidf_vec.fit_transform(message_train)
    message_test = tfidf_vec.transform(message_test)

    print('tfidf + NB')

    if normal == True:
        scaler = preprocessing.StandardScaler(with_mean=False).fit(message_train)
        message_train=scaler.transform(message_train)
        message_test=scaler.transform(message_test)

    clf = MultinomialNB()
    clf.fit(message_train, label_train)

    label_test_predict = clf.predict(message_test)
    print("测试集大小:", len(label_test), "得分:", clf.score(message_test, label_test))
    print(classification_report(label_test, label_test_predict, target_names=all_classes, zero_division="warn"))


# CountVectorizer 词频 + SVM 分类
# messages: list 留言信息
# labels: list 留言的分类
# ratio: float 测试集比例
# normal: 归一化
def svm_train(messages: list, labels: list, ratio: float, normal:bool) -> None:
    message_train, message_test, label_train, label_test = train_test_split(messages, labels, test_size=ratio)
    count_vec = CountVectorizer()
    message_train = count_vec.fit_transform(message_train)
    message_test = count_vec.transform(message_test)

    print('count + SVM')

    if normal == True:
        scaler = preprocessing.StandardScaler(with_mean=False).fit(message_train)
        message_train=scaler.transform(message_train)
        message_test=scaler.transform(message_test)

    c = 3
    kernel = "linear"

    clf = SVC(decision_function_shape="ovo", C=c, kernel=kernel)
    clf.fit(message_train, label_train)

    label_test_predict = clf.predict(message_test)
    print("测试集大小:", len(label_test), "得分:", clf.score(message_test, label_test))
    print(classification_report(label_test, label_test_predict, target_names=all_classes, zero_division="warn"))


# tfidf 词频 + SVM 分类
# messages: list 留言信息
# labels: list 留言的分类
# ratio: float 测试集比例
# normal: 归一化
def tfidf_svm_train(messages: list, labels: list, ratio: float, normal:bool) -> None:
    message_train, message_test, label_train, label_test = train_test_split(messages, labels, test_size=ratio)
    tfidf_vec = TfidfVectorizer()
    message_train = tfidf_vec.fit_transform(message_train)
    message_test = tfidf_vec.transform(message_test)

    print('tfidf + SVM')

    if normal == True:
        scaler = preprocessing.StandardScaler(with_mean=False).fit(message_train)
        message_train=scaler.transform(message_train)
        message_test=scaler.transform(message_test)

    c = 9
    gamma = 0.08
    kernel = "rbf"

    clf = SVC(decision_function_shape="ovo", C=c, gamma=gamma, kernel=kernel)
    clf.fit(message_train, label_train)


    label_test_predict = clf.predict(message_test)
    print("测试集大小:", len(label_test), "得分:", clf.score(message_test, label_test))
    print(classification_report(label_test, label_test_predict, target_names=all_classes, zero_division="warn"))


# 搜索最优参数
# CountVectorizer 词频 + SVM 分类
# messages: list 留言信息
# labels: list 留言的分类
# ratio: float 测试集比例
# normal: 归一化
def search_svm_train(messages: list, labels: list, ratio: float, normal: bool) -> None:
    message_train, message_test, label_train, label_test = train_test_split(messages, labels, test_size=ratio)
    count_vec = CountVectorizer()
    message_train = count_vec.fit_transform(message_train)
    message_test = count_vec.transform(message_test)

    print('count + SVM')

    if normal == True:
        scaler = preprocessing.StandardScaler(with_mean=False).fit(message_train)
        message_train=scaler.transform(message_train)
        message_test=scaler.transform(message_test)

    svc = SVC()
    parameters = [
        {
            'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
            'gamma': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000],
            'kernel': ['rbf']
        },
        {
            'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
            'kernel': ['linear']
        }
    ]
    f_scorer = make_scorer(f1_score, average = 'weighted')
    clf = GridSearchCV(svc, parameters, cv=5, n_jobs=-1,verbose=1,scoring=f_scorer)
    clf.fit(message_train, label_train)
    print(clf.best_params_)
    best_model = clf.best_estimator_
    label_test_predict = best_model.predict(message_test)

    print("测试集大小:", len(label_test), "得分:", clf.score(message_test, label_test))
    print(classification_report(label_test, label_test_predict, target_names=all_classes, zero_division="warn"))

# 搜索最优参数
# tfidf 词频 + SVM 分类
# messages: list 留言信息
# labels: list 留言的分类
# ratio: float 测试集比例
# normal: 归一化
def search_tfidf_svm_train(messages: list, labels: list, ratio: float, normal: bool) -> None:
    message_train, message_test, label_train, label_test = train_test_split(messages, labels, test_size=ratio)
    tfidf_vec = TfidfVectorizer()
    message_train = tfidf_vec.fit_transform(message_train)
    message_test = tfidf_vec.transform(message_test)

    print('tfidf + SVM')

    if normal == True:
        scaler = preprocessing.StandardScaler(with_mean=False).fit(message_train)
        message_train=scaler.transform(message_train)
        message_test=scaler.transform(message_test)

    svc=SVC()
    parameters = [
        {
            'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
            'gamma': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000],
            'kernel': ['rbf']
        },
    {
        'C': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
        'kernel': ['linear']
    }
    ]
    f_scorer = make_scorer(f1_score, average = 'weighted')
    clf = GridSearchCV(svc, parameters, cv=5, n_jobs=-1,verbose=1,scoring=f_scorer)
    clf.fit(message_train, label_train)
    print(clf.best_params_)
    best_model = clf.best_estimator_
    label_test_predict = best_model.predict(message_test)

    print("测试集大小:", len(label_test), "得分:", best_model.score(message_test, label_test))
    print(classification_report(label_test, label_test_predict, target_names=all_classes, zero_division="warn"))



# tfidf + KNN 分类
# messages: list 留言信息
# labels: list 留言的分类
# ratio: float 测试集比例
# normal: 归一化
def tfidf_KNN_train(messages: list, labels: list, ratio: float, normal: bool) -> None:
    message_train, message_test, label_train, label_test = train_test_split(messages, labels, test_size=ratio)
    tfidf_vec = TfidfVectorizer()
    message_train = tfidf_vec.fit_transform(message_train)
    message_test = tfidf_vec.transform(message_test)

    print('tfidf + KNN')

    if normal == True:
        scaler = preprocessing.StandardScaler(with_mean=False).fit(message_train)
        message_train=scaler.transform(message_train)
        message_test=scaler.transform(message_test)

    knn = KNeighborsClassifier(n_neighbors=34,p=2,weights='distance')

    # 训练模型
    knn.fit(message_train, label_train)
    label_test_predict = knn.predict(message_test)

    print("测试集大小:", len(label_test), "得分:", knn.score(message_test, label_test))
    print(classification_report(label_test, label_test_predict, target_names=all_classes, zero_division="warn"))


# count + KNN 分类
# messages: list 留言信息
# labels: list 留言的分类
# ratio: float 测试集比例
# normal: 归一化
def KNN_train(messages: list, labels: list, ratio: float, normal: bool) -> None:
    message_train, message_test, label_train, label_test = train_test_split(messages, labels, test_size=ratio)
    count_vec = CountVectorizer()
    message_train = count_vec.fit_transform(message_train)
    message_test = count_vec.transform(message_test)

    print('count + KNN')

    if normal == True:
        scaler = preprocessing.StandardScaler(with_mean=False).fit(message_train)
        message_train = scaler.transform(message_train)
        message_test = scaler.transform(message_test)

    knn = KNeighborsClassifier(n_neighbors=33, p=2, weights='distance')

    # 训练模型
    knn.fit(message_train, label_train)
    label_test_predict = knn.predict(message_test)

    print("测试集大小:", len(label_test), "得分:", knn.score(message_test, label_test))
    print(classification_report(label_test, label_test_predict, target_names=all_classes, zero_division="warn"))

# tfidf + KNN 分类寻优
# messages: list 留言信息
# labels: list 留言的分类
# ratio: float 测试集比例
# normal: 归一化
# k=18/24 p=2 weight=distance 0.88
def search_tfidf_KNN_train(messages: list, labels: list, ratio: float, normal: bool) -> None:
    message_train, message_test, label_train, label_test = train_test_split(messages, labels, test_size=ratio)
    tfidf_vec = TfidfVectorizer()
    message_train = tfidf_vec.fit_transform(message_train)
    message_test = tfidf_vec.transform(message_test)

    print('tfidf + KNN')

    if normal == True:
        scaler = preprocessing.StandardScaler(with_mean=False).fit(message_train)
        message_train=scaler.transform(message_train)
        message_test=scaler.transform(message_test)

    knn = KNeighborsClassifier()
    # 训练模型
    param_grid = [
        {
            'weights': ['uniform'],
            'n_neighbors': [i for i in range(1, 50)]
        },
        {
            'weights': ['distance'],
            'n_neighbors': [i for i in range(1, 50)],
            'p': [i for i in range(1, 6)]
        }
    ]
    f_scorer = make_scorer(f1_score, average = 'weighted')
    grid_search = GridSearchCV(knn, param_grid,cv=5 ,n_jobs=-1,verbose=1,scoring=f_scorer)

    # 训练模型
    grid_search.fit(message_train, label_train)
    print(grid_search.best_params_)
    best_model = grid_search.best_estimator_
    label_test_predict = best_model.predict(message_test)

    print("测试集大小:", len(label_test), "得分:", best_model.score(message_test, label_test))
    print(classification_report(label_test, label_test_predict, target_names=all_classes, zero_division="warn"))


# count + KNN 分类寻优
# messages: list 留言信息
# labels: list 留言的分类
# ratio: float 测试集比例
# normal: 归一化
# k=18/24 p=2 weight=distance 0.88
def search_KNN_train(messages: list, labels: list, ratio: float, normal: bool) -> None:
    message_train, message_test, label_train, label_test = train_test_split(messages, labels, test_size=ratio)
    count_vec = CountVectorizer()
    message_train = count_vec.fit_transform(message_train)
    message_test = count_vec.transform(message_test)

    print('count + KNN')

    if normal == True:
        scaler = preprocessing.StandardScaler(with_mean=False).fit(message_train)
        message_train = scaler.transform(message_train)
        message_test = scaler.transform(message_test)

    knn = KNeighborsClassifier()
    # 训练模型
    param_grid = [
        {
            'weights': ['uniform'],
            'n_neighbors': [i for i in range(1, 50)]
        },
        {
            'weights': ['distance'],
            'n_neighbors': [i for i in range(1, 50)],
            'p': [i for i in range(1, 6)]
        }
    ]
    f_scorer = make_scorer(f1_score, average='weighted')
    grid_search = GridSearchCV(knn, param_grid, cv=5, n_jobs=-1, verbose=1, scoring=f_scorer)

    # 训练模型
    grid_search.fit(message_train, label_train)
    print(grid_search.best_params_)
    best_model = grid_search.best_estimator_
    label_test_predict = best_model.predict(message_test)

    print("测试集大小:", len(label_test), "得分:", best_model.score(message_test, label_test))
    print(classification_report(label_test, label_test_predict, target_names=all_classes, zero_division="warn"))




def main():
    # 读取分类数据
    global all_classes

    read = np.load('data/messages_new.npy', allow_pickle=True)
    messages = [item[0] for item in read]
    labels = [item[1] for item in read]


    for label in labels:
        if label not in all_classes:
            all_classes.append(label)

    Normalize = True

    count_train(messages, labels, 0.2, Normalize)
    tfidf_train(messages, labels, 0.2, Normalize)
    svm_train(messages, labels, 0.2, Normalize)   #{'C': 3, 'kernel': 'linear'} 得分: 0.8935939196525515
    tfidf_svm_train(messages, labels, 0.2, Normalize)     #{'C': 7, 'gamma': 0.1, 'kernel': 'rbf'}  得分: 0.9201954397394136
    KNN_train(messages, labels, 0.2, Normalize)   #{'n_neighbors': 33, 'p': 2, 'weights': 'distance'}  得分: 0.6351791530944625
    tfidf_KNN_train(messages, labels, 0.2, Normalize)     #{'n_neighbors': 34, 'p': 2, 'weights': 'distance'}  得分: 0.8762214983713354


    #调参函数
    # search_svm_train(messages, labels, 0.2, Normalize)
    # search_tfidf_svm_train(messages, labels, 0.2, Normalize)
    # search_KNN_train(messages, labels, 0.2, Normalize)
    # search_tfidf_KNN_train(messages, labels, 0.2, Normalize)



if __name__ == "__main__":
    main()
