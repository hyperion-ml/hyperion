"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange
#from six import string_types

#import os.path as path
import logging
from copy import deepcopy

import numpy as np
import pandas as pd

from .list_utils import *
from .segment_list import SegmentList


class RTTM(object):
    """Class to manipulate rttm files

    Attributes:
      df: Pandas dataframe.
      _index_by_file: if True the df is index by file name, if False by segment id.
      iter_idx: index of the current element for the iterator.
      unique_file_key: unique file names.
    """
    def __init__(self, segments, index_by_file=True):
        self.segments = segments
        self._index_by_file = index_by_file
        if index_by_file:
            self.segments.index = self.segments.file_id
        else:
            self.segments.index = [i for i in xrange(len(segments))]
        self.validate()
        self.unique_file_id = self.segments.file_id.unique()
        self.iter_idx = 0


        
    @classmethod
    def create(cls, segment_type, file_id, chnl=None, tbeg=None, tdur=None,
               ortho=None, stype=None, name=None, conf=None, slat=None,
               index_by_file=True):
        num_segments = len(segment_type)
        nans = ['<NA>' for i in xrange(num_segments)]
        if chnl is None:
            chnl = [1 for i in xrange(num_segments)]
        if tbeg is None:
            tbeg = nans
        if tdur is None:
            tend = nans
        if ortho is None:
            ortho = nans
        if stype is None:
            stype = nans
        if name is None:
            name = nans
        if conf is None:
            conf = nans
        if slat is None:
            slat = nans

        df = pd.DataFrame({'segment_type': segment_type,
                           'file_id': file_id,
                           'chnl': chnl,
                           'tbeg': tbeg,
                           'tdur': tdur,
                           'ortho': ortho,
                           'stype': stype,
                           'name': name,
                           'conf': conf,
                           'slat': slat})
        
        return cls(df, index_by_file)




    @classmethod
    def create_spkdiar(cls, file_id, tbeg, tdur, spk_id, conf=None, chnl=None,
                       index_by_file=True, prepend_file_id=False):
        segment_type = ['SPEAKER'] * len(file_id)
        spk_id = cls._make_spk_ids(spk_id, file_id, prepend_file_id)
        return cls.create(segment_type, file_id, chnl, tbeg, tdur,
                          name=spk_id, conf=conf, index_by_file=index_by_file)



    @classmethod
    def create_spkdiar_from_segments(cls, segments, spk_id, conf=None, chnl=None,
                                     index_by_file=True, prepend_file_id=False):
        assert len(segments) == len(spk_id)
        file_id = segments.file_id
        tbeg = segments.tbeg
        tdur = segments.tend - segments.tbeg
        segment_type = ['SPEAKER'] * len(file_id)
        spk_id = cls._make_spk_ids(spk_id, file_id, prepend_file_id)
        return cls.create(segment_type, file_id, chnl, tbeg, tdur,
                          name=spk_id, conf=conf,
                          index_by_file=index_by_file)


    @classmethod
    def create_spkdiar_from_ext_segments(cls, ext_segments, chnl=None, index_by_file=True, prepend_file_id=False):
        file_id = ext_segments.file_id
        tbeg = ext_segments.tbeg
        tdur = ext_segments.tend - ext_segments.tbeg
        segment_type = ['SPEAKER'] * len(file_id)
        name = ext_segments.segment_names
        conf = ext_segments.segment_score
        if prepend_file_id:
            name = cls._prepend_file_id(name, file_id)
            
        return cls.create(segment_type, file_id, chnl, tbeg, tdur,
                          name=name, conf=conf,
                          index_by_file=index_by_file)
        

    
    @staticmethod
    def _make_spk_ids(spk_ids, file_id, prepend_file_id):
        if prepend_file_id:
            return [ f + '_' + str(s) for f,s in zip(file_id,spk_ids)]
        return spk_ids #[str(s) for f,s in zip(file_id,spk_ids)]


    @staticmethod
    def _prepend_file_id(spk_ids, file_id):
        return [ f + '_' + str(s) for f,s in zip(file_id,spk_ids)]

    
    def validate(self):
        """Validates the attributes of the RTTM object.
        """
        assert np.all(np.logical_or(self.tbeg[1:]-self.tbeg[:-1]>=0,
                                    self.file_id[1:] != self.file_id[:-1]))


        
    @property
    def index_by_file(self):
        return self._index_by_file


    
    @index_by_file.setter
    def index_by_file(self, value):
        self._index_by_file = value
        if self._index_by_file:
            self.segments.index = self.segments.file_key
        else:
            self.segments.index = self.segments.segment
        
    
            
    @property
    def file_id(self):
        return np.asarray(self.segments['file_id'])


    
    @property
    def tbeg(self):
        return np.asarray(self.segments['tbeg'])


    
    @property
    def tdur(self):
        return np.asarray(self.segments['tdur'])

    
    
    @property
    def name(self):
        return np.asarray(self.segments['name'])


    
    def copy(self):
        """Makes a copy of the object."""
        return deepcopy(self)


    
    @property
    def num_files(self):
        return len(self.unique_file_id)


    
    @property
    def total_num_spks(self):
        return len(self.segments[self.segments['segment_type']=='SPEAKER'].name.unique())


    
    @property
    def num_spks_per_file(self):
        return {file_id: len(self.segments[(self.segments['segment_type']=='SPEAKER') &
                                           (self.segments['file_id']==file_id)].name.unique())
                for file_id in self.unique_file_id}
    

    
    @property
    def avg_num_spks_per_file(self):
        return np.mean([v for k, v in self.num_spks_per_file.items()])

    
    
    def __iter__(self):
        self.iter_idx=0
        return self


    
    def __next__(self):
        if self.index_by_file:
            if self.iter_idx < len(self.unique_file_id):
                r = self.__getitem__(self.unique_file_id[self.iter_idx])
            else:
                raise StopIteration()
        else:
            if self.iter_idx < len(self.segments):
                r = self.__getitem__(self.iter_idx)
            else:
                raise StopIteration()

        self.iter_idx += 1
        return r


    
    def __len__(self):
        """Returns the number of segments in the list."""
        return len(self.segments)
    
    
    def __contains__(self, key):
        """ Returns True if the segments contains the key"""
        return key in self.segments.segment_id


    

    def __getitem__(self, key):
        """It allows to acces the de segments by file_id or segment
           like in a ditionary, e.g.:
           If input is a string key:
               segmetns = SegmentList(...)
               segment, tbeg, tend = segments['file']
        Args:
          key: Segment or file key
        Returns:
          if index_by_file is True if returns segments of a given file_id 
          in SegmentsList format, else it returns DataFrame
        """
        if self.index_by_file:
            df = self.segments.loc[key]
            return RTTM(df, index_by_file=False)
        else:
            return self.segments.iloc[key]


    
    def save(self, file_path, sep=' '):
        """Saves segments to text file.

        Args:
          file_path: File to write the list.
          sep: Separator between the fields
        """
        self.segments[['segment_type', 'file_id', 'chnl',
                       'tbeg','tdur','ortho', 'stype',
                       'name', 'conf', 'slat']].to_csv(
                           file_path, sep=sep, float_format='%.3f',
                           index=False, header=False)


        
    @classmethod
    def load(cls, file_path, sep=' ', index_by_file=True):
        """Loads script list from text file.

        Args:
          file_path: File to read the list.
          sep: Separator between the key and file_path in the text file.

        Returns:
          SegmentList object.
        """
        df = pd.read_csv(file_path, sep=sep, header=None,
                         names=['segment_type','file_id','chnl','tbeg','tdur',
                                'ortho','stype','name','conf','slat'])
        return cls(df, index_by_file=index_by_file)


    
    def filter(self, filter_key, keep=True):
        if not keep :
            filter_key = np.setdiff1d(
                np.asrray(self.segments.index), filter_key)
        df = self.segments.loc[filter_key]
        return RTTM(df, index_by_file=self.index_by_file)
    

    
    def split(self, idx, num_parts):
        key, _ = split_list(self.index, idx, num_parts)
        df = self.segments.loc[key]
        return RTTM(df, index_by_file=self.index_by_file)



    @classmethod
    def merge(cls, rttm_list, index_by_file=True):
        dfs = []
        for rttm_i in rttm_list:
            dfs.append(rttm_i.segments)
        df = pd.concat(dfs)
        return cls(df, index_by_file=index_by_file)

    

    def merge_adjacent_segments(self, t_margin=0):
        segm = self.segments
        segm_1 = self.segments.shift(1)
        delta = segm.tbeg - segm_1.tbeg - segm_1.tdur
        index = ((segm.file_id == segm_1.file_id) &
                 (segm.segment_type == segm_1.segment_type) &
                 (segm.name == segm_1.name) & (delta <= t_margin))
        # ix=self.segments['name'] == 'DH_0006_1'
        # logging.debug(self.segments[ix])
        # logging.debug(index[ix])
        # logging.debug(segm[ix].tbeg)
        # logging.debug(segm[ix].tbeg+segm[ix].tdur)
        for i in xrange(len(self.segments)-1, 0, -1):
            if index.iloc[i]:
                tbeg = segm.iloc[i-1].tbeg
                tend = segm.iloc[i].tbeg + segm.iloc[i].tdur
                self.segments.iloc[i-1, self.segments.columns.get_loc('tdur')] = tend - tbeg
                self.segments.iloc[i, self.segments.columns.get_loc('segment_type')] = 'DROP'
        #logging.debug(self.segments[ix])
        self.segments = self.segments[self.segments.segment_type != 'DROP']

        # for i in xrange(len(self.segments)-1, 0, -1):
        #     row_i = self.segments.iloc[i]
        #     row_i_1 = self.segments.iloc[i-1]
        #     if (row_i.segment_type == row_i_1.segment_type and
        #         row_i.file_id == row_i_1.file_id and
        #         row_i.name == row_i_1.name and
        #         row_i.tbeg - row_i_1.tbeg - row_i_1.tdur <= t_margin):
        #         tend = row_i.tbeg + row_i.tdur
        #         self.segments.iloc[i-1, self.segments.columns.get_loc('tdur')] = tend - row_i_1.tbeg
        #         self.segments.iloc[i, self.segments.columns.get_loc('segment_type')] = 'DROP'
        # self.segments = self.segments[self.segments.segment_type != 'DROP']


    
    def __eq__(self, other):
        """Equal operator"""
        eq = self.segments.equals(other.segments)
        eq = eq and self.index_by_file == other.index_by_file
            
        return eq


    
    def __ne__(self, other):
        """Non-equal operator"""
        return not self.__eq__(other)


    
    def __cmp__(self, other):
        """Comparison operator"""
        if self.__eq__(other):
            return 0
        return 1


    
    def get_segment_names_slow(self, segment_list, sep='@', segment_type='SPEAKER'):
        num_segm = len(segment_list)
        names = []
        num_names = np.zeros((num_segm,), dtype=int)
        for i in xrange(num_segm):
            file_id = segment_list.file_id[i]
            tbeg = segment_list.tbeg[i]
            tend = segment_list.tend[i]
            segments_i = self.segments[(self.segments['segment_type'] == segment_type) &
                                       (self.segments['file_id'] == file_id) &
                                       ((self.segments['tbeg']<=tbeg) &
                                        (self.segments['tbeg'] + self.segments['tdur'] > tbeg) |
                                        (self.segments['tbeg']<tend) &
                                        (self.segments['tbeg'] + self.segments['tdur'] >=tend))]
            num_segm_i = len(segments_i)
            if num_segm_i == 0:
                num_names[i] = 0
                names.append('<NA>')
            else:
                names_i = list(segments_i['name'].unique())
                num_names[i] = len(names_i)
                names.append(sep.join(names_i))

        return np.asarray(names), num_names


    def get_segment_names(self, segment_list, sep='@', segment_type='SPEAKER'):
        num_segm = len(segment_list)
        names = []
        num_names = np.zeros((num_segm,), dtype=int)
        segments = self.segments[self.segments['segment_type'] == segment_type]
        prev_file_id = ''
        for i in xrange(num_segm):
            file_id = segment_list.file_id[i]
            tbeg = segment_list.tbeg[i]
            tend = segment_list.tend[i]
            if file_id != prev_file_id:
                segments_f = segments[segments['file_id'] == file_id]
                tbeg_f = segments_f['tbeg']
                tend_f = segments_f['tbeg'] + segments_f['tdur']
                prev_file_id = file_id

            segments_i = segments_f[((tbeg_f<=tbeg) & (tend_f > tbeg)) |
                                    ((tbeg_f < tend) & (tend_f >=tend))]

            num_segm_i = len(segments_i)
            if num_segm_i == 0:
                num_names[i] = 0
                names.append('<NA>')
            else:
                names_i = list(segments_i['name'].unique())
                num_names[i] = len(names_i)
                names.append(sep.join(names_i))

        return np.asarray(names), num_names


    
    def get_files_with_names_diff_to_file(self, file_id, segment_type='SPEAKER'):
        segments = self.segments[self.segments['segment_type'] == segment_type]
        names = segments[segments['file_id'] == file_id].name.unique()
        sel_files = segments[~segments.name.isin(names)].file_id.unique()
        return sel_files
                                        
            
    def prepend_file_id_to_name(self, segment_type='SPEAKER'):
        idx = self.segments['segment_type'] == segment_type
        self.segments.loc[idx, 'name'] = self.segments.loc[
            idx, ['file_id', 'name']].apply(lambda x: '_'.join(x), axis=1)


    def eliminate_overlaps(self):
        segm = self.segments
        segm_1 = self.segments.shift(1)
        tend_1 = segm_1.tbeg + segm_1.tdur
        tbeg = segm.tbeg
        tend = segm.tbeg + segm.tdur
        tavg = (tbeg + tend_1)/2
        index = ((segm.file_id == segm_1.file_id) &
                 (segm.segment_type == segm_1.segment_type) &
                 (tbeg < tend_1))
        # logging.debug(index)
        # logging.debug(self.segments.loc[index])
        # logging.debug(self.segments.loc[index])
        self.segments.loc[index, 'tbeg'] = tavg[index]
        # logging.debug(self.segments.loc[index])
        index_1 = index.shift(-1)
        index_1[index_1.isnull()] = False
        # logging.debug(self.segments.loc[index_1])
        self.segments.loc[index_1, 'tdur'] = tavg[index] - tbeg[index_1]
        # logging.debug(self.segments.loc[index_1])
        self.segments.loc[index, 'tdur'] = tend[index] - tavg[index]



    
    def to_matrix(self, file_id, frame_shift=0.001):
        if self.index_by_file:
            segments = self.segments[file_id]
        else:
            segments = self.segments[self.segments['file_id'] == file_id]

        u_names, name_ids = np.unique(segments['name'], return_inverse=True)
        tbeg = np.round(segments['tbeg']/frame_shift).astype(dtype=int)
        tend = np.round((segments['tbeg']+segments['tdur'])/frame_shift).astype(dtype=int)

        M = np.zeros((tend[-1], len(u_names)), dtype=int)
        for i in xrange(len(u_names)):
            M[tbeg[i]:tend[i], name_ids[i]] = 1


            
    def compute_stats(self, nbins_dur=None):

        # segment durations
        max_dur = self.segments['tdur'].max()
        min_dur = self.segments['tdur'].min()
        mean_dur = self.segments['tdur'].mean()
        std_dur = self.segments['tdur'].std()
        median_dur = self.segments['tdur'].median()
        mode_dur = self.segments['tdur'].mode()
        dur_info = pd.Series([mean_dur, std_dur, median_dur, mode_dur, min_dur, max_dur],
                             index=['mean','std','median','model','min','max'])

        if nbins_dur is None:
            nbins_dur = np.max(5, np.min(100, len(self.segments)/10))
        hist_dur = np.histogram(self.segments['tdur'], nbins_dur)

        
        # number of speakers
        total_spks = len(self.segments['name'].unique())

        # overlaps
