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


def creat_instance():
    k = 0
    st = []
    for i in range(11300*16): #11,500    
        st.append(str(i) + "; " + str(np.random.randint(100)) + "; " + str(np.random.randint(100)) + "; " + str(np.random.randint(100)) + "; " + str(np.random.randint(100)))
    return st
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

def _write(problems):
    for i in problems:
        _f = i
        if problems[i][0] == 0 and problems[i][1] == 0:
            continue
        try:
            if write_or_not(i.strip(".in") + ".out", problems[i][0]):
                write(i, problems[i], _f = _f) 
        except:
            print("File " + _f + " not found!")
            write(i, problems[i], _f = _f) 
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

def Master():
    problems = {}
    preserved = {}
    all_probs=["problem1.in", "problem2.in","problem3.in","problem4.in","problem5.in", "problem6.in","problem7.in", "problem8.in", "problem9.in", "problem10.in", "problem11.in", "problem12.in", "problem13.in", "problem14.in", "problem15.in", "problem16.in", "problem17.in", "problem18.in", "problem19.in", "problem20.in", "problem21.in"]
    all_probs_22=["problem3.in","problem4.in","problem8.in", "problem13.in", "problem14.in", "problem16.in", "problem17.in", "problem20.in", "problem21.in"]
    all_probs_17=["problem1.in","problem10.in"]
    all_probs_12=["problem6.in","problem12.in"]
    all_probs_8=["problem2.in","problem7.in"]
    all_probs_6 = ["problem18.in", "problem19.in", "problem5.in"]
    all_probs_3 = ["problem9.in","problem15.in", "problem11.in"]
    for problem in all_probs:
        problems[problem] = [0,0]
        preserved[problem] = [0,0]

    for i in range(10):
        problems = preserved
        for prob in all_probs_22:
            print("\"\"\"", prob, "\"\"\"")
            st = time.clock()
            try:          
                s, k = kp_R(_file=prob, num=22)
            except:
                continue
            max_s=max(s)
            max_k=k[s.index(max(s))]
            if problems[prob][0] < max_s:
                problems[prob][0] = max_s
                problems[prob][1] = max_k
            ed = time.clock()
            print("TIME = ", ed-st)
        for prob in all_probs_17:
            print("\"\"\"", prob, "\"\"\"")
            st = time.clock()
            try:          
                s, k = kp_R(_file=prob, num=17)
            except:
                continue
            max_s=max(s)
            max_k=k[s.index(max(s))]
            if problems[prob][0] < max_s:
                problems[prob][0] = max_s
                problems[prob][1] = max_k
            ed = time.clock()
            print("TIME = ", ed-st)
        for prob in all_probs_12:
            print("\"\"\"", prob, "\"\"\"")
            st = time.clock()
            try:          
                s, k = kp_R(_file=prob, num=12)
            except:
                continue
            max_s=max(s)
            max_k=k[s.index(max(s))]
            if problems[prob][0] < max_s:
                problems[prob][0] = max_s
                problems[prob][1] = max_k
            ed = time.clock()
            print("TIME = ", ed-st)
        for prob in all_probs_8:
            print("\"\"\"", prob, "\"\"\"")
            st = time.clock()            
            try:          
                s, k = kp_R(_file=prob, num=8)
            except:
                continue
            max_s=max(s)
            max_k=k[s.index(max(s))]
            if problems[prob][0] < max_s:
                problems[prob][0] = max_s
                problems[prob][1] = max_k
            ed = time.clock()
            print("TIME = ", ed-st)
        for prob in all_probs_6:
            print("\"\"\"", prob, "\"\"\"")
            st = time.clock()
            try:          
                s, k = kp_R(_file=prob, num=6)
            except:
                continue
            max_s=max(s)
            max_k=k[s.index(max(s))]
            if problems[prob][0] < max_s:
                problems[prob][0] = max_s
                problems[prob][1] = max_k
            ed = time.clock()
            print("TIME = ", ed-st)
        for prob in all_probs_3:
            print("\"\"\"", prob, "\"\"\"")
            st = time.clock()
            try:          
                s, k = kp_R(_file=prob, num=3)
            except:
                continue
            max_s=max(s)
            max_k=k[s.index(max(s))]
            if problems[prob][0] < max_s:
                problems[prob][0] = max_s
                problems[prob][1] = max_k
            ed = time.clock()
            print("TIME = ", ed-st)
    _write(problems)
    return problems
def _powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    # s = list([i for i in range(len(iterable))])
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

# A = np.array([[5,7,9],
#          [8,8,12],
#          [9,10,11],
#          [7,6,12],
#          [12,10,12]])

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

def check_valid(items, incomp_classes, _dict):
    classes = []
    for item in items:
        classes.append(item[1])
    for cls in classes:
        c = 0
        for _cls in classes:
            if _cls in _dict[str(int(cls))] and _cls != cls:
                c += 1
        if c >= 1:
            return False
    return True

def kp_R(p=0, m=0, n=0, items=[], c=0, incomp_classes=[], _file="problem1.in", num=10):
    if p == 0 and m==0 and n==0:
        params = get_params(_file)
        p, m, n, c, items, incomp_classes = params[0], params[1],  params[2],  params[3],\
        params[4],  params[5]
    scoress = []
    knapsackss = []
    _dict = make_dict(items, incomp_classes)

    for u in range(2):
        knapsack = []
        best_weight = 0
        best_value = 0
        best_cost = 0
        sample = random.sample(items, num)
        st = time.clock()
        while not check_valid(sample, incomp_classes, _dict):
            if (time.clock() - st) > 6:
                print(_file, " FAILED!!")
                return
                # print(_file, " FAILED!!")
                # sample = items[0]
                # break
            sample = random.sample(items, 17)

        ps = list(powerset(sample))
        for i in range(len(ps)):
            if ps[i] == () or ps[i] == []:
                if m > best_value:
                    best_weight, best_cost, best_value  = 0, 0, m
            _weight = sum(map(weight, ps[i]))
            _cost = sum(map(cost, ps[i]))
            if _weight > p or _cost > m:
                continue
            val = sum(map(value, ps[i])) + (m - _cost)
            if val > best_value:
                best_weight, best_cost, best_value, knapsack = _weight, _cost, val, [item[0] for item in ps[i]]

        scoress.append(best_value)
        knapsackss.append(knapsack)

    return scoress, knapsackss

# A = [i for i in range(30)]
# st = time.clock()
# u = list(powerset(A))
# ed = time.clock()
# print("time = ", ed-st)

st = time.clock()
# s, k = kp_R(_file="problem1.in")
pr = Master()
ed = time.clock()
print("time = ", ed-st)
# max(s)
# k[s.index(max(s))]

# params = get_params("problem11.in")
# d = make_dict(params[4], params[5])
# st = time.clock()
# ed = 0
# while ed - st < 5:
#     sample = random.sample(params[4], 8)
#     u = check_valid(sample,2,d)
#     check_valid(sample,2,d)
#     if u == True:
#         print("WINNEERRRR!!!")
#         print(ed - st)
#         break
#     ed = time.clock()
# def kp_G_A(p=0, m=0, n=0, items=[], c=0, incomp_classes=[], _file="problem1.in"):
#     if p == 0 and m==0 and n==0:
#         params = get_params(_file)
#         p, m, items = params[0], params[1], params[4]
#     lst1, lst2, lst3, lst4, lst5, lst6, sorted_lsts = [], [], [], [], [], [], []
#     for it in items:
#         try:
#             lst1.append((it, it[1]/it[0]))
#         except Exception:
#             lst1.append((it, 10000000000))
#         try:
#             lst2.append((it, it[2]/it[0]))
#         except Exception:
#             lst2.append((it, 10000000000))
#         try:
#             lst3.append((it, (it[2]-it[1])/it[0]))
#         except Exception:
#             lst3.append((it, 10000000000))
#         try:
#             lst4.append((it, (it[2]/it[1])/it[0]))
#         except Exception:
#             lst4.append((it, 10000000000))
#         try:
#             lst5.append((it, it[2]/it[1]))
#         except Exception:
#             lst5.append((it, 10000000000))
#         try:
#             lst6.append((it, it[2]/(it[1]+it[0])))
#         except Exception:
#             lst6.append((it, 10000000000))
#     for lst in [lst1, lst2, lst3, lst4, lst5, lst6]:
#         sorted_lsts.append(sorted(lst, key=lambda x: x[1]))
#     sols = []
#     u, leave = 1, []
#     for lst in sorted_lsts:
#         knapsack, tot_weight,tot_value,tot_cost = [],0,0,0
#         while len(lst) > 0:
#             item = lst.pop()[0]
#             if (weight(item) + tot_weight) <= p and (cost(item) + tot_cost) <= m:
#                 knapsack.append(item)
#                 tot_weight += weight(item)
#                 tot_value += value(item)
#                 tot_cost += cost(item)
#                 leave.append([knapsack, tot_value+(m-tot_cost)])
#             else:
#                 leave.append([knapsack, tot_value+(m-tot_cost)])
#                 break
#         sols.append([u, tot_value + (m-tot_cost), knapsack])
#         u+=1
#     leave.extend(sols)
#     sor = sorted(leave, key=lambda x: x[1])
#     print("sols=", sols)
#     return sor[-1], sols
# def 