#coding: utf-8

'''
这个文件的主要用途是将这个比赛作为一个文本分类的比赛来看
这样的话，我就主要利用essay.csv这个文件了
这是一个比较naive的benchmark，所以不要期望有太大的结果
'''

from read_conf import config
import csv
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation

#得到按照训练集重新弄得y
def get_new_y(conf):

    train_id = []
    train_ans = {}

    print "read from train essay"
    train_f = open(conf["train_essay"])
    reader = csv.reader(train_f)
    for line in reader:
        train_id.append(line[0])

    print "read from outcome"
    outcome = open(conf["outcome"])
    out_reader = csv.reader(outcome)
    g = lambda x: "1" if x=="t" else "0"
    a = 0
    for line in out_reader:
        train_ans[line[0]] = g(line[1])
        if a%1000 == 0:
            print a
        a += 1

    print "write y"
    y = open(conf["y_essay"],"w")
    for one in train_id:
        y.write("%s,%s\n"%(one,train_ans[one]))

    y.close()



#读取并分割train，test
def split_train_test(conf,test_id):
    print "split train test"
    f = open(conf["essay"])

    #开train test文件
    train_f = open(conf["train_essay"],"w")
    test_f = open(conf["test_essay"],"w")

    reader = csv.reader(f)
    train_w = csv.writer(train_f)
    test_w = csv.writer(test_f)
    a = 0
    for line in reader:
        if a == 0:
            a += 1
            continue

        if line[0] in test_id:
            test_w.writerow(line)
        else:
            train_w.writerow(line)
        
        if a% 100000 == 0:
            print a    
        a += 1

    train_f.close()
    test_f.close()
    
#读取test的所有项目id
def get_test_id(conf):
    f = open(conf["sample"])
    reader = csv.reader(f)

    test_id = []
    a = 0
    for line in reader:
        if a == 0 :
            a += 1
            continue

        test_id.append(line[0])
        a += 1
    f.close()
    return set(test_id)

#读入train test文件
def read_file(conf):
    train_f = open(conf["train_essay"])
    test_f = open(conf["test_essay"])
    y_f = open(conf["y_essay"])
    train,test,y = [],[],[]
    test_id = []

    print "read train"
    reader = csv.reader(train_f)
    for line in reader:
        train.append(line[2]+line[3])

    print "read test"
    reader = csv.reader(test_f)    
    for line in reader:
        test.append(line[2]+" "+line[3])
        test_id.append(line[0])

    print "read y"
    reader = csv.reader(y_f)
    for line in reader:
        y.append(int(line[1]))

    return train,test,y,test_id
    
#取tf-idf
def get_tfidf(train,test):
    print "make tf-idf"
    vectorizer = TfidfVectorizer(max_features=None,min_df=3,max_df=1.0,sublinear_tf=True,ngram_range=(1,2),smooth_idf=True,token_pattern=r'\w{1,}',analyzer='word',strip_accents='unicode')

    length_train = len(train)
    print "fit train tf-idf"
    x_all = train + test
    x_all = vectorizer.fit_transform(x_all)
    x = x_all[:length_train]
    t = x_all[length_train:]

    return x,t

#训练
def train(conf,ctype):
    train,test,y,test_id = read_file(conf)

    #得到tf-idf
    train,test = get_tfidf(train,test)

    y = np.array(y)
    clf = LogisticRegression(penalty='l2',dual=True,fit_intercept=False,C=2,tol=0.0001,class_weight=None, random_state=None, intercept_scaling=1.0)

    if ctype == "cv":
        
        print "交叉训练"
        print "cross validation",np.mean(cross_validation.cross_val_score(clf,train,y,cv=3,scoring='roc_auc',n_jobs=3))
    elif ctype == "predict":
        clf.fit(train,y)
        predict = clf.predict_proba(test)[:,1]

        f = open(conf["result_essay"],"w")
        f.write("projectid,is_exciting\n")
        for it in range(len(test_id)):
            f.write("%s,%s\n"%(test_id[it],predict[it]))
    
    
if __name__ == '__main__':
    print "hello"

    #读取数据文件的conf文件，获取地址
    dp = config("../conf/dp.conf")

    if len(sys.argv)!=2:
        print "usage python essay_bench.py <usage>"
        print "usage:split=> train test essay split"
        print "usage:get_y=> get well writen y"
        print "usage:train=> fit train and predict test"
        sys.exit(1)
        
    if sys.argv[1] == "split":
        #step 1: 先把train和test分开
        
        #sub step 1: 先获取test文件所有的id
        test_id = get_test_id(dp)
    
        #sub step 2: 读取所有的essay文件，在此分割,并且将其写入train和test文件中
        split_train_test(dp,test_id)

    elif sys.argv[1] == "get_y":
        #sub step 3: 取出所有的outcome中的结果，然后和train中的匹配一下，重新弄个y
        get_new_y(dp)

    elif sys.argv[1] == "predict":
        train(dp,"predict")
    elif sys.argv[1] == "cv":
        train(dp,"cv")
    else:
        print "error:=> no such option"

    
