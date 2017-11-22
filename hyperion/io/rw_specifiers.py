"""
Functions to write and read kaldi files
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

from enum import Enum

class ArchiveType(Enum):
    H5 = 0
    ARK = 1


# Documentation for "wspecifier"
# "wspecifier" describes how we write a set of objects indexed by keys.
# The basic, unadorned wspecifiers are as follows:
#
#  h5:wxfilename
#  ark:wxfilename
#  scp:rxfilename
#  h5,scp:filename,wxfilename
#  ark,scp:filename,wxfilename
#
#
#  We also allow the following modifiers:
#  t means text mode.
#  b means binary mode.
#  f means flush the stream after writing each entry.
#   (nf means don't flush, and isn't very useful as the default is to flush).
#  p means permissive mode, when writing to an "scp" file only: will ignore
#     missing scp entries, i.e. won't write anything for those files but will
#     return success status).
#
#  So the following are valid wspecifiers:
#  ark,b,f:foo
#  "ark,b,b:| gzip -c > foo"
#  "ark,scp,t,nf:foo.ark,|gzip -c > foo.scp.gz"
#  ark,b:-
#
#  The meanings of rxfilename and wxfilename are as described in
#  kaldi-stream.h (they are filenames but include pipes, stdin/stdout
#  and so on; filename is a regular filename.
#

#  The ark:wxfilename type of wspecifier instructs the class to
#  write directly to an archive.  For small objects (e.g. lists of ints),
#  the text archive format will generally be human readable with one line
#  per entry in the archive.
#
#  The type "scp:xfilename" refers to an scp file which should
#  already exist on disk, and tells us where to write the data for
#  each key (usually an actual file); each line of the scp file
#  would be:
#   key xfilename
#
#  The type ark,scp:filename,wxfilename means
#  we write both an archive and an scp file that specifies offsets into the
#  archive, with lines like:
#    key filename:12407
#  where the number is the byte offset into the file.
#  In this case we restrict the archive-filename to be an actual filename,
#  as we can't see a situtation where an extended filename would make sense
#  for this (we can't fseek() in pipes).


class WSpecType(Enum):
    NO = 0
    ARCHIVE = 1
    SCRIPT = 2
    BOTH = 3


class WSpecifier(object):
    def __init__(self, spec_type, archive, script,
                 archive_type=ArchiveType.H5,
                 binary=True, flush=False, permissive=False):
        self.archive = archive
        self.script = script
        self.spec_type = spec_type
        self.archive_type = archive_type
        self.binary = binary
        self.flush = flush
        self.permissive = permissive

    @classmethod
    def create(cls, wspecifier):
        fields = wspecifier.strip().split(':')
        if len(fields) == 1:
            assert len(fields[0]) > 0
            return cls(WSpecType.ARCHIVE, fields[0], None)
        elif len(fields) == 2:
            options = fields[0].strip().split(',')
            archives = fields[1].strip().split(',')

            archive = None
            script = None
            archive_type = None
            binary = True
            flush = False
            permissive = False

            cur_archive = 0
            for option in options:
                if option == 'h5':
                    assert archive_type is None
                    assert archive is None, (
                        'Repeated h5, ark in wspecifier %s' % script)
                    assert len(archives) > cur_archive
                    archive_type = ArchiveType.H5
                    archive = archives[cur_archive]
                    cur_archive += 1
                elif option == 'ark':
                    assert archive_type is None
                    assert archive is None, (
                        'Repeated h5, ark in wspecifier %s' % script)
                    assert len(archives) > cur_archive
                    archive_type = ArchiveType.ARK
                    archive = archives[cur_archive]
                    cur_archive += 1
                elif option == 'scp':
                    assert script is None, (
                        'Repeated scp in wspecifier %s' % script)
                    assert len(archives) > cur_archive
                    script = archives[cur_archive]
                    cur_archive += 1
                elif option in ['b', 't', 'f', 'nf', 'p']:
                    pass
                else:
                    raise ValueError('Wrong wspecifier options %s'
                                     % fields[0])
                
            if archive is None:
                if script is not None:
                    spec_type = WSpecType.SCRIPT
                else:
                    raise ValueError('Wrong wspecifier %s ' % wspecifier)
            else:
                if script is None:
                    spec_type = WSpecType.ARCHIVE
                else:
                    spec_type = WSpecType.BOTH
                
            if archive_type == ArchiveType.ARK:
                for option in options:
                    if option == 't':
                        binary = False
                    elif option == 'f':
                        flush = True
                    elif option == 'p':
                        permissive = True
                
            return cls(spec_type, archive, script,
                       archive_type, binary, flush, permissive)
        else:
            raise ValueError('Two many fields (%d>2) in wspecifier %s'
                             % (len(fields), wspecifier))


        
    def __eq__(self, other):
        eq = self.archive == other.archive
        eq = eq and self.script == other.script
        eq = eq and self.spec_type == other.spec_type
        eq = eq and self.archive_type == other.archive_type
        eq = eq and self.binary == other.binary
        eq = eq and self.flush == other.flush
        eq = eq and self.permissive == other.permissive
        return eq
    

    def __ne__(self, other):
        return not self.__eq__(other)


    
    def __cmp__(self, other):
        if self.__eq__(other):
            return 0
        return 1



# Documentation for "rspecifier"
# "rspecifier" describes how we read a set of objects indexed by keys.
# The possibilities are:
#
# h5:rxfilename
# ark:rxfilename
# scp:rxfilename
#
# We also allow various modifiers:
#   o   means the program will only ask for each key once, which enables
#       the reader to discard already-asked-for values.
#   s   means the keys are sorted on input (means we don't have to read till
#       eof if someone asked for a key that wasn't there).
#   cs  means that it is called in sorted order (we are generally asserting
#       this based on knowledge of how the program works).
#   p   means "permissive", and causes it to skip over keys whose corresponding
#       scp-file entries cannot be read. [and to ignore errors in archives and
#       script files, and just consider the "good" entries].
#       We allow the negation of the options above, as in no, ns, np,
#       but these aren't currently very useful (just equivalent to omitting the
#       corresponding option).
#       [any of the above options can be prefixed by n to negate them, e.g. no,
#       ns, ncs, np; but these aren't currently useful as you could just omit
#       the option].
#   bg means "background".  It currently has no effect for random-access readers,
#       but for sequential readers it will cause it to "read ahead" to the next
#       value, in a background thread.  Recommended when reading larger objects
#       such as neural-net training examples, especially when you want to
#       maximize GPU usage.
#
#   b   is ignored [for scripting convenience]
#   t   is ignored [for scripting convenience]
#
#
#  So for instance the following would be a valid rspecifier:
#
#   "o, s, p, ark:gunzip -c foo.gz|"

        
class RSpecType(Enum):
    NO = 0
    ARCHIVE = 1
    SCRIPT = 2


class RSpecifier(object):
    def __init__(self, spec_type, archive,
                 archive_type=ArchiveType.H5,
                 once = False, is_sorted = False, called_sorted = False,
                 permissive=False, background = False):

        self.spec_type = spec_type
        self.archive = archive
        self.archive_type = archive_type
        self.once = once
        self.is_sorted = is_sorted
        self.called_sorted = called_sorted
        self.permissive = permissive
        self.background = background


    @property
    def script(self):
        return self.archive

    
    @classmethod
    def create(cls, rspecifier):
        fields = rspecifier.strip().split(':')
        if len(fields) == 1:
            assert len(fields[0]) > 0
            return cls(RSpecType.ARCHIVE, fields[0])
        elif len(fields) == 2:
            options = fields[0].strip().split(',')
            archives = fields[1].strip().split(',')
            assert len(archives) == 1

            spec_type = None
            archive = archives[0]
            archive_type = None
            once = False
            is_sorted = False
            called_sorted = False
            permissive = False
            background = False

            for option in options:
                if option == 'h5':
                    assert spec_type is None
                    spec_type = RSpecType.ARCHIVE
                    archive_type = ArchiveType.H5
                elif option == 'ark':
                    assert spec_type is None
                    spec_type = RSpecType.ARCHIVE
                    archive_type = ArchiveType.ARK
                elif option == 'scp':
                    assert spec_type is None
                    spec_type = RSpecType.SCRIPT
                elif option in ['o', 's', 'cs', 'bg', 'p']:
                    pass
                else:
                    raise ValueError('Wrong wspecifier options %s'
                                     % fields[0])

            assert spec_type is not None, ('Wrong wspecifier options %s'
                                           % fields[0])
            
            if spec_type == RSpecType.SCRIPT:
                with open(archive, 'r') as f:
                    scp = f.readline().strip().split(' ')[1].split(':')
                    if len(scp) == 1:
                        archive_type = ArchiveType.H5
                    else:
                        archive_type = ArchiveType.ARK
                
            if archive_type == ArchiveType.ARK:
                for option in options:
                    if option == 'o':
                        once = True
                    elif option == 's':
                        is_sorted = True
                    elif option == 'cs':
                        called_sorted = True
                    elif option == 'p':
                        permissive = True
                    elif option == 'bg':
                        background = True
                
            return cls(spec_type, archive, archive_type,
                       once, is_sorted, called_sorted,
                       permissive, background)
        else:
            raise ValueError('Two many fields (%d>2) in wspecifier %s'
                             % (len(fields), wspecifier))
        
        

    def __eq__(self, other):
        eq = self.spec_type == other.spec_type
        eq = eq and self.archive == other.archive
        eq = eq and self.archive_type == other.archive_type
        eq = eq and self.once == other.once
        eq = eq and self.is_sorted == other.is_sorted
        eq = eq and self.called_sorted == other.called_sorted
        eq = eq and self.permissive == other.permissive
        eq = eq and self.background == other.background
        return eq

    
    def __ne__(self, other):
        return not self.__eq__(other)
