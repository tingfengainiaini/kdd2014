#coding: utf-8
'''
author:yaoming
这个文件是最初的尝试之一，只利用了projects.csv里面的feature，作为初始benchmark之一
大概完成任务如下：
1.reset:将projects.csv按照essay.csv的顺序排列
2.split:将projects中的train和test分开
3.encode:对projects的train和test分别进行one-hot encode并将结果写入文件
4.cv:进行cv
5.predict:进行训练预测，输出result.csv

'''
from read_conf import config
import csv,sys
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn import svm

#将project.csv按照essay的顺序重新排列
def reset_project(conf):

    projects_dict = {}
    pf = open(conf["project"],'r')
    pf_reader = csv.reader(pf)
    count = 0
    for line in pf_reader:
        if count == 0:
            count += 1
            continue
        projects_dict[line[0]] = line
    pf.close()
    print "got project dict"
    
    order_list = []
    ef = open(conf["essay"],'r')
    ef_reader = csv.reader(ef)
    count = 0
    for line in ef_reader:
        if count == 0:
            count += 1
            continue
        order_list.append(line[0])
    ef.close()
    print "got order list"

    resf = open(conf["reset_project"],'w')
    ef_writer = csv.writer(resf)
    count = 0
    for one_id in order_list:
        ef_writer.writerow(projects_dict[one_id])
        count += 1
        if count % 100000 == 0:
            print count
    resf.close()
    print "got ordered　project"
    
#读取并分割train，test
def split_train_test(conf,test_id):
    print "split train test"
    f = open(conf["reset_project"])

    #开train test文件
    train_f = open(conf["train_project"],"w")
    test_f = open(conf["test_project"],"w")

    reader = csv.reader(f)
    train_w = csv.writer(train_f)
    test_w = csv.writer(test_f)
    a = 0
    for line in reader:
        
        if line[0] in test_id:
            test_w.writerow(line)
        else:
            train_w.writerow(line)

        a += 1
        if a% 100000 == 0:
            print a    
        
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

    
#为category的feature建立字典,形成类别数目，以便进行one-hot编码
bool_to_int = {'f':0,'t':1}
#school_city = {}
#school_state = {}
school_metro = {}
#school_county = {}
teacher_prefix = {}
focus_subject = {}
focus_area = {}
resource_type = {}
poverty_level = {}
grade_level = {}

#定义一个project类来对projects进行处理
class project:
    def __init__(self,line):
        self.projectid = line[0]
        
        #关于学校ＩＤ和提交的老师的ＩＤ相关的３个feature直接扔了没要
        #关于学校地点的feature，暂时只使用了经纬度，其他地点相关的feature全都扔了
        self.school_latitude = float(line[4])
        self.school_longitude = float(line[5])

#        if not school_city.has_key(line[6]):
#            school_city[line[6]] = len(school_city)
#        self.school_city = school_city[line[6]]

#        if not school_state.has_key(line[7]):
#            school_state[line[7]] = len(school_state)
#        self.school_state = school_state[line[7]]

        if not school_metro.has_key(line[9]):
            school_metro[line[9]] = len(school_metro)
        self.school_metro = school_metro[line[9]]

#        if not school_county.has_key(line[11]):
#            school_county[line[11]] = len(school_county)
#        self.school_county = school_county[line[11]]

        self.school_charter = bool_to_int[line[12]]
        self.school_magnet = bool_to_int[line[13]]
        self.school_year_round = bool_to_int[line[14]]
        self.school_nlns = bool_to_int[line[15]]
        self.school_kipp = bool_to_int[line[16]]
        self.school_charter_ready_promise = bool_to_int[line[17]]

        if not teacher_prefix.has_key(line[18]):
            teacher_prefix[line[18]] = len(teacher_prefix)
        self.teacher_prefix = teacher_prefix[line[18]]

        self.teacher_teach_for_america = bool_to_int[line[19]]
        self.teacher_ny_teaching_fellow = bool_to_int[line[20]]

        if not focus_subject.has_key(line[21]):
            focus_subject[line[21]] = len(focus_subject)
        self.primary_focus_subject = focus_subject[line[21]]

        if not focus_area.has_key(line[22]):
            focus_area[line[22]] = len(focus_area)
        self.primary_focus_area = focus_area[line[22]]

        if not focus_subject.has_key(line[23]):
            focus_subject[line[23]] = len(focus_subject)
        self.secondary_focus_subject = focus_subject[line[23]]

        if not focus_area.has_key(line[24]):
            focus_area[line[24]] = len(focus_area)
        self.secondary_focus_area = focus_area[line[24]]

        if not resource_type.has_key(line[25]):
            resource_type[line[25]] = len(resource_type)
        self.resource_type = resource_type[line[25]]

        if not poverty_level.has_key(line[26]):
            poverty_level[line[26]] = len(poverty_level)
        self.poverty_level = poverty_level[line[26]]

        if not grade_level.has_key(line[27]):
            grade_level[line[27]] = len(grade_level)
        self.grade_level = grade_level[line[27]]

        if line[28] == '':
             self.fulfillment_labor_materials = 0
        else:
            self.fulfillment_labor_materials = float(line[28])
        self.total_price_excluding_optional_support = float(line[29])
        self.total_price_including_optional_support = float(line[30])
        if line[31] == '':
            self.students_reached = 0
        else:
            self.students_reached = float(line[31])
        self.eligible_double_your_impact_match = bool_to_int[line[32]]
        self.eligible_almost_home_match = bool_to_int[line[33]]

        self.feature_list = [self.school_latitude,self.school_longitude,self.school_metro,\
                             self.school_charter,self.school_magnet,self.school_year_round,\
                             self.school_nlns,self.school_kipp,self.school_charter_ready_promise,\
                             self.teacher_prefix,self.teacher_teach_for_america,\
                             self.teacher_ny_teaching_fellow,self.primary_focus_subject,\
                             self.primary_focus_area,self.secondary_focus_subject,\
                             self.secondary_focus_area,self.resource_type,self.poverty_level,\
                             self.grade_level,self.fulfillment_labor_materials,\
                             self.total_price_excluding_optional_support,\
                             self.total_price_including_optional_support,\
                             self.students_reached,self.eligible_double_your_impact_match,\
                             self.eligible_almost_home_match]

        #读取prjects.csv转换成字典，projectid:feature_list
def read_project(conf):
    train_dict = {}
    test_dict = {}
    train_f = open(conf["train_project"],'r')
    test_f = open(conf["test_project"],'r')
    train_reader = csv.reader(train_f)
    test_reader = csv.reader(test_f)
    for line in train_reader:
        train_dict[line[0]] = (project(line)).feature_list
    for line in test_reader:
        test_dict[line[0]] = (project(line)).feature_list
    print "got train and test origin dict"
    #print train_dict
    return train_dict,test_dict

#对数据进行one-hot encoding
def onehot_encode(data_dict):
    new_dict = {}
    #这个mask是one-hot编码所需要的对应的catrgory　feature所对应的index,也就是只对mask所指的feature进行编码
    mask = np.array([2,9,12,13,14,15,16,17,18])
    enc = OneHotEncoder(categorical_features = mask)
    input_list = data_dict.values()
    enc.fit(input_list)
    count = 0
    for one_id in data_dict.keys():
        new_dict[one_id] = (enc.transform(data_dict[one_id]).toarray())[0].tolist()
        count += 1
        if count % 100000 == 0:
            print count
        #if count == 1:
        #    print data_dict[one_id]
        #    print new_dict[one_id]
    return new_dict
    
#读取已经进行预处理的数据，准备进行训练
def read_data(conf):
    #read train test y
    train,test,y,test_label = [],[],[],[]

    f = open(conf["train_oh_project"])
    reader = csv.reader(f)
    for line in reader:
        sub_line = line[1:]
        sub_line = [float(i) for i in sub_line]
        train.append(sub_line)

    f.close()

    f = open(conf["test_oh_project"])
    reader = csv.reader(f)
    for line in reader:
        test_label.append(line[0])
        sub_line = line[1:]
        sub_line = [float(i) for i in sub_line]
        test.append(sub_line)

    f.close()
        
    f = open(conf["y_essay"])
    reader = csv.reader(f)
    for line in reader:
        y.append(float(line[1]))

    f.close()
    return train,test,y,test_label

    
def train_and_predict(conf,ctype):
    """
    
    Arguments:
    - `conf`:
    """
    #read train test y
    print "load data..."
    train,test,y,test_label = read_data(conf)
    train,test,y = np.array(train),np.array(test),np.array(y)

    print "train shape",train.shape
    print "test shape",test.shape

    print "norm"
    scaler = preprocessing.StandardScaler().fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    print "pca"
    pca = PCA(n_components=113,whiten=True)
    pca.fit(train)
    train = pca.transform(train)
    test = pca.transform(test)

    clf = LogisticRegression(penalty='l2',dual=True,fit_intercept=False,C=2,tol=1e-9,class_weight=None, random_state=None, intercept_scaling=1.0)
    #clf = GaussianNB()
    #clf = MultinomialNB()
    #clf = GradientBoostingClassifier(n_estimators=400)
    #clf = RandomForestClassifier(n_estimators=400)
    #clf = RandomForestClassifier(n_estimators=100,max_depth=8,min_samples_leaf=4,n_jobs=3)
    #clf = SGDClassifier(loss="log", penalty="l2",alpha=0.1)
    #clf = svm.SVC(C = 1.0, kernel = 'rbf', probability = True)
    if ctype == "cv":
        print "交叉验证"
        hehe = cross_validation.cross_val_score(clf,train,y,cv=3,scoring='roc_auc',n_jobs=-1)
        print hehe
        print np.mean(hehe)

    elif ctype =="predict":
        clf.fit(train,y)
        predict = clf.predict_proba(test)[:,1]

        if len(predict)!=len(test_label):
            print "predict!=test label"
            sys.exit(1)

        rf = open(conf["result"],"w")
        rf.write("projectid,is_exciting\n")
        for i in range(len(predict)):
            rf.write("%s,%s\n"%(test_label[i],predict[i]))

    
if __name__ == '__main__':
    dp_conf = config("../conf/dp.conf")
    if len(sys.argv)!=2:
        print "usage python projects_benchmark.py <usage>"
        print "usage:reset=> reset projects.csv"
        print "usage:split=> train test project split"
        print "usage:encode=> train test one hot encode"
        print "usage:cv=> cross validiction"
        print "usage:predict=> train and predict test"
        sys.exit(1)
    
    if sys.argv[1] == 'reset':
        reset_project(dp_conf)

    if sys.argv[1] == 'split':
        test_id = get_test_id(dp_conf)
        split_train_test(dp_conf,test_id)

    if sys.argv[1] == 'encode':
        train_dict,test_dict = read_project(dp_conf)
        print "one-hot encoding!"
        train_oh_dict = onehot_encode(train_dict)
        print "train encode finish"
        test_oh_dict = onehot_encode(test_dict)
        print "test encode finish"

        with open(dp_conf["train_oh_project"],'w') as train_f:
            train_writer = csv.writer(train_f)
            for one_id in train_oh_dict.keys():
                line = [one_id]
                line.extend(train_oh_dict[one_id])
                train_writer.writerow(line)

        with open(dp_conf["test_oh_project"],'w') as test_f:
            test_writer = csv.writer(test_f)
            for one_id in test_oh_dict.keys():
                line = [one_id]
                line.extend(test_oh_dict[one_id])
                test_writer.writerow(line)

        print "oh_project.csv files have been generated!"

    if sys.argv[1] == 'cv':
        train_and_predict(dp_conf,"cv")
        
    if sys.argv[1] == 'predict':
        train_and_predict(dp_conf,"predict")
