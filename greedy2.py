import string
import time
import numpy as np
from itertools import chain, combinations
import random
import numpy as np
import itertools
# list(itertools.product(*arrays))
import sys
import string
import time
import numpy as np
from itertools import chain, combinations

import string
import numpy as np
alphabet = list(string.ascii_lowercase)

all_probs=["problem1.in", "problem2.in","problem3.in","problem4.in","problem5.in", "problem6.in","problem7.in", "problem8.in", "problem9.in", "problem10.in", "problem11.in", "problem12.in", "problem13.in", "problem14.in", "problem15.in", "problem16.in", "problem17.in", "problem18.in", "problem19.in", "problem20.in", "problem21.in"]

def write_or_not(_name, score):
    content = loadFromFile(_name)
    stuff = float(content[2])
    if stuff > score:
        return False
    return True


def write(prob,probs,_f="here.txt"):
    _max = probs[0]
    ks = probs[1]
    f = open(_f.strip(".in")+".out", 'w')
    f.write("FILE: " + prob + '\n')
    f.write("Max Score: \n")
    f.write(str(_max) +'\n')
    f.write("KnapSack: \n")
    for i in ks:
        f.write(str(i) + '\n')

def weight(item):
    return item[2]

def cost(item):
    return item[3]

def value(item):
    return item[4]

def loadFromFile(_name):
    with open(_name) as f:
        content = f.readlines()
    content = [x.strip("\n") for x in content]
    return content

def get_params(_name):
    content = loadFromFile(_name)
    p = float(content[0])   # Pounds
    m = float(content[1])   # Dollars
    n = float(content[2])   # Num_Items
    c = float(content[3])   # Num_Constraints
    incomp_classes = []
    items = []
    for x in content[3:]:
        if ";" in x:
            temp = x.split(";")
            t = [temp[0]]
            t.extend([float(x) for x in temp[1:]])
            items.append(t)

        elif ',' in x:
            temp = x.split(",")
            incomp_classes.append([int(u) for u in temp])
    return p, m, n, c, items, incomp_classes



def make_dict(items, incomp_classes):
    set_of_inc_clss = set([x[1] for x in items])
    _dic = dict()
    j = 0
    for i in set_of_inc_clss:
        _dic[str(int(i))] = set()
    for i in incomp_classes:
        for classes in i:
            try:
                _dic[str(classes)].update(i)
            except KeyError:
                continue
    return _dic

def kp_G_A(p=0, m=0, n=0, items=[], c=0, incomp_classes=[], _file="problem1.in"):
    if p == 0 and m==0 and n==0:
        params = get_params(_file)
        p, m, items = params[0], params[1], params[4]
    lst1, lst2, lst3, lst4, lst5, lst6,lst7,lst8,lst9,lst10, sorted_lsts = [], [], [], [], [], [], [], [], [], [], []
    for it in items:
        try:
            lst1.append((it, it[1+2]/it[0+2]))
        except Exception:
            lst1.append((it, 10000000000))
        try:
            lst2.append((it, it[2+2]/it[0+2]))
        except Exception:
            lst2.append((it, 10000000000))
        try:
            lst3.append((it, (it[2+2]-it[1+2])/it[0+2]))
        except Exception:
            lst3.append((it, 10000000000))
        try:
            lst4.append((it, (it[2+2]/it[1+2])/it[0+2]))
        except Exception:
            lst4.append((it, 10000000000))
        try:
            lst5.append((it, it[2+2]/it[1+2]))
        except Exception:
            lst5.append((it, 10000000000))
        try:
            lst6.append((it, it[2+2]/(it[1+2]+it[0+2])))
        except Exception:
            lst6.append((it, 10000000000))
        try:
            lst7.append((it, weight(it)/p))
        except Exception:
            lst7.append((it, 10000000000))
        try:
            lst8.append((it, weight(it)/m))
        except Exception:
            lst8.append((it, 10000000000))
        try:
            lst9.append((it, (value(it)-cost(it))**2/(p*m)))
        except Exception:
            lst9.append((it, 10000000000))
        try:
            lst10.append((it, (value(it)-cost(it))**2/(p+m)))
        except Exception:
            lst10.append((it, 10000000000))
    for lst in [lst1, lst2, lst3, lst4, lst5, lst6]:
        sorted_lsts.append(sorted(lst, key=lambda x: x[1]))
    sols = []
    u, leave = 1, []
    for lst in sorted_lsts:
        knapsack, tot_weight,tot_value,tot_cost = [],0,0,0
        while len(lst) > 0:
            item = lst.pop()[0]
            if (weight(item) + tot_weight) <= p and (cost(item) + tot_cost) <= m:
                knapsack.append(item[0])
                tot_weight += weight(item)
                tot_value += value(item)
                tot_cost += cost(item)
                leave.append([000, tot_value+(m-tot_cost), knapsack])
            else:
                leave.append([000, tot_value+(m-tot_cost), knapsack])
                break
        sols.append([u, tot_value + (m-tot_cost), knapsack])
        u+=1
    leave.extend(sols)
    sor = sorted(leave, key=lambda x: x[1])
    return sor[-1], sols

def make_classes(_set,items):
    all_classes = dict()
    for i in _set:
        all_classes[str(i)] = []
    for i in _set:
        for x in items:
            if x[1] == i:
                all_classes[str(i)].append(x)
    return all_classes


def M_Greedy():
    all_probs=["problem1.in", "problem2.in","problem3.in","problem4.in","problem5.in", "problem6.in","problem7.in", "problem8.in", "problem9.in", "problem10.in", "problem11.in", "problem12.in", "problem13.in", "problem14.in", "problem15.in", "problem16.in", "problem17.in", "problem18.in", "problem19.in", "problem20.in", "problem21.in"]
    problems = {}
    for problem in all_probs:
        problems[problem] = [0,0]
    for t in all_probs:
        best = kp_GREEDY(_file = t)
        problems[t][0] = best[1]
        problems[t][1] = best[2]
    G_write(problems)
    return problems


def kp_GREEDY(p=0, m=0, n=0, items=[], c=0, incomp_classes=[], _file="problem1.in"):
    if p == 0 and m==0 and n==0:
        params = get_params(_file)
        p, m, n, c, items, incomp_classes = params[0], params[1],  params[2],  params[3],\
        params[4],  params[5]
    scoress = []
    knapsackss = []
    _dict = make_dict(items, incomp_classes)

    set_inc_clss = set([x[1] for x in items])
    all_classes = make_classes(set_inc_clss, items)
    best_class = [] #[class_name, class_score, class_knapsack]
    for _cls in all_classes:
        a,b=kp_G_A(p=p,m=m,n=n,c=c,items=all_classes[_cls])
        best_class.append([_cls, a[1], a[2]])
    best = sorted(best_class, key = lambda x: x[1])
    return best[-1]

def Gwrite(prob,probs,_f="here.txt"):
    _max = probs[0]
    ks = probs[1]
    f = open(_f.strip(".in")+"_greedy.out", 'w')
    f.write("FILE: " + prob + '\n')
    f.write("Max Score: \n")
    f.write(str(_max) +'\n')
    f.write("KnapSack: \n")
    for i in ks:
        f.write(str(i) + '\n')

def G_write(problems):
    for i in problems:
        _f = i
        if problems[i][0] == 0 and problems[i][1] == 0:
            continue
        try:
            if write_or_not(i.strip(".in") + "_greedy.out", problems[i][0]):
                Gwrite(i, problems[i], _f = _f) 
        except:
            print("File " + _f + " not found!")
            Gwrite(i, problems[i], _f = _f) 

import random
def make(file, all_classes, _dict):
    st = time.clock()
    params = get_params(file)
    p, m, n, c, items, incomp_classes = params[0], params[1],  params[2],  params[3],\
    params[4],  params[5]
    scoress = []
    knapsackss = []
    # _dict = make_dict(items, incomp_classes)
    # _dict  = eval(cont[1])
    print("1")
    clssss = [x[1] for x in items]
    set_inc_clss = set(clssss)
    print("2")
    # all_classes = make_classes(set_inc_clss, items)
    # _dict  = eval(cont[1])
    # all_classes =     all_classes = eval(cont[0])
    print("3")
    # return all_classes
    good_classes = []
    fake_items = []
    # return clssss, all_classes
    k = 0
    cld =list([key for key in all_classes])
    random.shuffle(cld)
    if time.clock() - st > 360*3:
        return
    for _cls in cld:
        if k > 500:
            continue
        if time.clock() - st > 360*3:
            return
        print(_cls)
        # random.shuffle(clssss)
        print(k)
        k+=1
        fake_items.append([0, _cls])
        if not check_valid(fake_items, incomp_classes, _dict):
            if time.clock() - st > 360*3:
                return
            print("oh")
            fake_items.pop()

    return _cls,all_classes, fake_items

def check_valid(items, incomp_classes, _dict):
    classes = []
    for item in items:
        classes.append(item[1])
    for cls in classes:
        c = 0
        for _cls in classes:
            # print(_cls + ":(")
            # print("This " + str(int(float(_cls))) + " in THIS " + str(int(float(cls))))
            # print(_dict[str(int(float(cls)))])
            if int(float(_cls)) in _dict[str(int(float(cls)))]:
                # print("HEY")
                if _cls != cls:
                    c += 1
        if c >= 1:
            return False
    return True
def runner():
    all_probs=["problem1.in", "problem2.in","problem3.in","problem5.in", "problem7.in", "problem8.in", "problem9.in",  "problem11.in", "problem12.in", "problem13.in",  "problem15.in", "problem17.in", "problem18.in", "problem19.in", "problem20.in", "problem21.in"]
    for i in range(30):
        for pr in all_probs:
            impr_G(_file=pr)

def impr_G(p=0, m=0, n=0, items=[], c=0, incomp_classes=[], _file="problem.in"):
    bestofbest = []
    cont = loadFromFile(str(10))[0].split(' ; ')
    _dict  = eval(cont[1])
    all_classes =     all_classes = eval(cont[0])
    for i in range(5):
        params = get_params(_file)
        p, m, n, c, items, incomp_classes = params[0], params[1],  params[2],  params[3],\
        params[4],  params[5]
        # try:
        _cls, all_classes, fake_items = make(_file, all_classes, _dict)
        # except:
            # return
        good_classes = [f[1] for f in fake_items]

        # for _cls in all_classes:
        its = []
        best_class = [] #[class_name, class_score, class_knapsack]
        u = [all_classes[clss] for clss in good_classes]
        for uu in u:
            its.extend(uu)
        a,b=kp_G_A(p=p,m=m,n=n,c=c,items=its)
        best_class.append([_cls, a[1], a[2]])
        bestofbest.append([_cls, a[1], a[2]])
        print("HERE")
    # best = sorted(best_class, key = lambda x: x[1], reverse = True)
        _write_Greedy(bestofbest, _file)
    return bestofbest

def _write_Greedy(problems, file):
    problems = sorted(problems, key = lambda x: x[1], reverse = True)
    p = problems[0]
    _f = file
    try:
        if write_or_not(_f.strip(".in") + ".out", p[1]):
            f = open(_f.strip(".in") + ".out", 'w')
            f.write("FILE: " + _f + '\n')
            f.write("Max Score: \n")
            f.write(str(p[1]) +'\n')
            f.write("KnapSack: \n")
            for i in p[2]:
                f.write(str(i) + '\n')
    except: 
        print("File " + _f + " not found!")
        f = open(_f.strip(".in") + ".out", 'w')
        f.write("FILE: " + _f + '\n')
        f.write("Max Score: \n")
        f.write(str(p[1]) +'\n')
        f.write("KnapSack: \n")
        for i in p[2]:
            f.write(str(i) + '\n')