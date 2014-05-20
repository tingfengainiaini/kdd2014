#coding: utf-8
'''
author:yaoming
这个文件和主体算法无关，只是我用来计算学校ID和老师ID的set的个数来看是否有价值将这两条作为feature加进去
'''
from read_conf import config
import csv,sys
import numpy as np

def count_different(conf,col_num_list):
    teacher_id,school_id1,school_id2 = [],[],[]
    with open(conf["project"],'r') as pf:
        reader = csv.reader(pf)
        for line in reader:
            teacher_id.append(line[col_num_list[0]])
            school_id1.append(line[col_num_list[1]])
            school_id2.append(line[col_num_list[2]])
    print "teacher id"
    print len(set(teacher_id))
    print "school id1"
    print len(set(school_id1))
    print "school id2"
    print len(set(school_id2))


if __name__ == "__main__":
    dp_conf = config("../conf/dp.conf")
    col_list = [1,2,3]
    count_different(dp_conf,col_list)
