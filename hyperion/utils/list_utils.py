"""
Utilities for lists.
"""
from __future__ import absolute_import
from __future__ import print_function
from six.moves import xrange

import numpy as np
from operator import itemgetter
from itertools import groupby


def list2ndarray(a):
    if(isinstance(a, list)):
        return np.asarray(a)
    assert(isinstance(a, np.ndarray))
    return a


def ismember(a, b):
    bad_idx = np.iinfo(np.int32).min
    d = {}
    for i, elt in enumerate(b):
        if elt not in d:
            d[elt] = i
    loc = np.array([d.get(x, bad_idx) for x in a], dtype='int32')
    f = loc != bad_idx
    return f, loc


def sort(a, reverse=False, return_index=False):
    if return_index:
        idx = np.argsort(a)
        if reverse:
            idx = idx[::-1]
        if not(isinstance(a, np.ndarray)):
            a = np.asarray(a)
        s_a = a[idx]
        return s_a, idx
    else:
        if reverse:
            return np.sort(a)[::-1]
        return np.sort(a)

    
def intersect(a, b, assume_unique=False, return_index = False):
    c = np.intersect1d(a, b, assume_unique)
    if return_index:
        _, ia = ismember(c, a)
        _, ib = ismember(c, b)
        return c, ia, ib
    else:
        return c



def split_list(a, idx, num_parts, key = None):
    if not(isinstance(a, np.ndarray)):
        a = np.asarray(a)
    if key is None:
        key = a
    _, ids=np.unique(key, return_inverse=True)
    n = float(ids.max()+1)
    idx_1 = int(np.floor((idx-1)*n/num_parts))
    idx_2 = int(np.floor(idx*n/num_parts))
    loc = np.empty(len(a), dtype='int64')
    k = 0
    for i in range(idx_1, idx_2):
        loc_i = (ids==i).nonzero()[0]
        loc[k:k+len(loc_i)] = loc_i
        k += len(loc_i)
    loc = loc[:k]
    return a[loc], loc


# def test_list_utils():

#     list1 = ['2', '2', '1', '2', '3', '3', '4', '4', '4', '4']
#     list2 = ['0', '1', '2', '20']
#     f, loc = ismember(list2, list1)
#     assert(np.all(loc == [np.iinfo(np.int32).min, 2, 0, np.iinfo(np.int32).min]))
#     assert(np.all(f == [False, True, True, False]))

#     list1_s, idx = sort(list1, return_index=True)
#     assert(np.all(list1_s == ['1', '2', '2', '2', '3', '3', '4', '4', '4', '4']))
    
#     list1_u, i_a, i_b, = np.unique(list1, True, True)
#     assert(np.all(list1_u == ['1', '2', '3', '4']))
#     assert(np.all(i_a == [2,  0, 4, 6]))
#     assert(np.all(i_b == [1, 1, 0, 1, 2, 2, 3, 3, 3, 3]))
    
#     list_i, i_a, i_b = intersect(list2, list1_u, return_index=True)
#     assert(np.all(list_i == ['1', '2']))
#     assert(np.all(i_a == [1, 2]))
#     assert(np.all(i_b == [0, 1]))

#     list1_d = np.setdiff1d(list1, list2)
#     assert(np.all(list1_d == ['3', '4']))
#     list2_d = np.setdiff1d(list2, list1)
#     assert(np.all(list2_d == ['0', '20']))

#     list_s, loc = split_list(list1, 1, 3)
#     assert(np.all(list_s == ['1']))
#     assert(np.all(loc == [2]))
    
#     list_s, loc = split_list(list1, 2, 3)
#     assert(np.all(list_s == ['2', '2', '2']))
#     assert(np.all(loc == [0, 1, 3]))

#     list_s, loc = split_list(list1, 3, 3)
#     assert(np.all(list_s == ['3', '3', '4', '4', '4', '4']))
#     assert(np.all(loc == [i for i in xrange(4, 10)]))

    
# def ismember(a, b):
#     d = {}
#     for i, elt in enumerate(b):
#         if elt not in d:
#             d[elt] = i
#     loc = [d.get(x, None) for x in a]
#     f = [x is not None for x in loc]
#     return f, loc

# def sort(a, reverse=False, return_index=False):
#     if return_index:
#         idx=sorted(range(len(a)), key=a.__getitem__, reverse=reverse)
#         s_a = [a[i] for i in idx]
#         return s_a, idx
#     else:
#         return sorted(a, reverse=reverse)
    


# def unique(a, return_index=False, return_inverse=False, return_counts=False):
#     r=np.unique(a, return_index, return_inverse, return_counts)
#     if isinstance(r, np.ndarray):
#         r = r.tolist()
#     else:
#         r = list(r)
#         r[0] = r[0].tolist()
#         r = tuple(r)
#     return r
#     # sort_list = sorted(set(list_in))
#     # list_out = map(itemgetter(0), groupby(sort_list))
#     # indx_in2out = [list_in.index(x) for x in list_out]
#     # indx_out2in = [list_out.index(x) for x in list_in]
#     # return list_out, indx_in2out, indx_out2in


# def intersect(a, b, assume_unique=False, return_index = False):
#     c = np.intersect1d(a, b, assume_unique).tolist()
#     if return_index:
#         _, ia = ismember(c, a)
#         _, ib = ismember(c, b)
#         return c, ia, ib
#     else:
#         return c
#     # f, ia = ismember(b, a)
#     # print(f)
#     # ia = ia[f]
#     # f, ib = ismember(a, b)
#     # ib = ib[f]
#     # c = a[ia]
#     # return c, ia, ib


# def setdiff(a, b,  assume_unique=False):
#     return np.setdiff1d(a, b, assume_unique).tolist()
#     #return list(set(a) - set(b))

# def union(a, b):
#     return np.union1d(a, b).tolist()


# def split_list(a, idx, num_parts, key = None):

#     if key is None:
#         key = a
#     _, _, ids=unique(key, return_index=True, return_inverse=True)
#     n = float(ids.max()+1)
#     idx_1 = int(np.floor((idx-1)*n/num_parts))
#     idx_2 = int(np.floor(idx*n/num_parts))
#     loc=np.empty(len(a), dtype='int32')
#     k=0
#     for i in range(idx_1, idx_2):
#         loc_i = (ids==i).nonzero()[0]
#         loc[k:k+len(loc_i)] = loc_i
#         k += len(loc_i)
#     loc = loc[:k]
#     return [a[j] for j in loc], loc.tolist()


# def test_list_utils():

#     list1 = ['2', '2', '1', '2', '3', '3', '4', '4', '4', '4']
#     list2 = ['0', '1', '2', '20']
#     f, loc = ismember(list2, list1)
#     assert(loc == [None, 2, 0, None])
#     assert(f == [False, True, True, False])

#     list1_s, idx = sort(list1, return_index=True)
#     assert(list1_s == ['1', '2', '2', '2', '3', '3', '4', '4', '4', '4'])
    
#     list1_u, i_a, i_b, = unique(list1, True, True)
#     assert(list1_u == ['1', '2', '3', '4'])
#     assert(np.all(i_a == [2,  0, 4, 6]))
#     assert(np.all(i_b == [1, 1, 0, 1, 2, 2, 3, 3, 3, 3]))
    
#     list_i, i_a, i_b = intersect(list2, list1_u, return_index=True)
#     assert(list_i == ['1', '2'])
#     assert(i_a == [1, 2])
#     assert(i_b == [0, 1])

#     list1_d = setdiff(list1, list2)
#     assert(list1_d == ['3', '4'])
#     list2_d = setdiff(list2, list1)
#     assert(list2_d == ['0', '20'])

#     list_s, loc = split_list(list1, 1, 3)
#     assert(list_s == ['1'])
#     assert(loc == [2])
#     list_s, loc = split_list(list1, 2, 3)
#     assert(list_s == ['2', '2', '2'])
#     assert(loc == [0, 1, 3])

#     list_s, loc = split_list(list1, 3, 3)
#     assert(list_s == ['3', '3', '4', '4', '4', '4'])
#     assert(loc == [i for i in xrange(4, 10)])

    

# class ListUtils:
#     @staticmethod
#     def parse_list(list_file,separator="="):
#         data=pd.read_csv(list_file,header=None,sep=separator)
#         n_columns=data.shape[1];
#         lists=[None]*n_columns
#         for i in range(n_columns):
#             lists[i]=data[i].values.tolist()
#         return lists

#     @staticmethod
#     def parse_scp(list_file,separator="="):
#         data=pd.read_csv(list_file,header=None,sep=separator)
#         assert(n_columns>=2,'File %s has n_columns=%d<2' % (list_file,n_columns))
#         key=data[0].values.tolist()
#         for i in xrange(2,n_columns):
#             data[1]+=data[i]
#         scp=data[1].values.tolist()
#         return key, scp

#     @staticmethod
#     def divide_lists(lists,part_indx,n_part,key_list=0):
#         n_lists=len(lists)
#         assert(key_list<n_lists,'key_list=%d but n_lists=%d' % (key_list,n_lists))
#         key=lists[key_list]
#         _,_,ids=unique(key_list)
#         n=ids.max()
#         indx1=floor((part_indx-1)*n/n_part);
#         indx2=floor(part_indx*n/n_part)-1;
#         loc=np.array([])
#         for i in range(indx1,indx2):
#             loc_i=(ids==i).nonzero()
#             loc=vstack((loc,loc_i))

#         sublists=[]
#         for i in xrange(n_lists):
#             sublist_i=[lists[i][j] for j in loc]
#             sublists.append(sublist_i)

#         return sublists
       
#     @staticmethod
#     def unique(list_in):
#         sort_list=sorted(set(list_in))
#         list_out=map(itemgetter(0), groupby(sort_list))
#         indx_in2out=[list_in.index(x) for x in list_out]
#         indx_out2in=[list_out.index(x) for x in list_in]
#         return list_out,indx_in2out,indx_out2in
    
#     @staticmethod
#     def ismember(a,b):
#         d = {}
#         for i, elt in enumerate(b):
#             if elt not in d:
#                 d[elt] = i
#         loc=[d.get(x,-1) for x in a]
#         f=[x != -1 for x in loc]
#         return f,loc

