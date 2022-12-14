import os
import shutil
import numpy as np
class Logs:
    log_path = os.getcwd() #Initial path to store log file
    def __init__(self):
        pass
    @classmethod
    def init_log_path(cls):
        return os.path.join(cls.log_path,'log.txt')
    @classmethod
    def log(cls, i_str):
        """Utility to saving string or dictionary to log file"""
        assert os.path.exists(cls.log_path)
        assert isinstance(i_str,(str,dict))
        with open(Logs.init_log_path(),'a+') as log:
            if isinstance(i_str, str):
                log.writelines('{}\n'.format(i_str))
                print('Logging: '+i_str)
            else:
                keys = i_str.keys()
                Logs.log(i_str='=' * 100)
                for key in keys:
                    Logs.log(i_str='(+) {:30} : {}'.format(key, i_str[key]))
                Logs.log(i_str='=' * 100)
    @classmethod
    def move_log(cls,i_dst_path):
        """Move the log file to another place"""
        assert isinstance(i_dst_path,str)
        src_path = os.path.join(cls.log_path,'log.txt')
        if i_dst_path.endswith('.txt'):
            dst_path = os.path.split(i_dst_path)[0]
        else:
            dst_path = i_dst_path
        assert os.path.exists(src_path)
        if os.path.exists(dst_path):
            pass
        else:
            os.makedirs(dst_path)
        shutil.move(src_path, dst_path)
        return dst_path
    @classmethod
    def rename_log(cls,i_new_name=None):
        """Rename the log file to new name at the same directory"""
        assert isinstance(i_new_name,str)
        src_path = os.path.join(cls.log_path, 'log.txt')
        if i_new_name.endswith('.txt'):
            dst_path = os.path.join(cls.log_path,i_new_name)
        else:
            dst_path = os.path.join(cls.log_path, i_new_name+'.txt')
        os.rename(src_path,dst_path)
        return dst_path
    @classmethod
    def move_log_with_new_name(cls,i_dst_path=None):
        assert isinstance(i_dst_path,str)
        src_path = os.path.join(cls.log_path, 'log.txt')
        if i_dst_path.endswith('.txt'):
            dst_path,dst_name = os.path.split(i_dst_path)
        else:
            dst_path,dst_name = i_dst_path,None
        assert os.path.exists(src_path)
        if os.path.exists(dst_path):
            pass
        else:
            os.makedirs(dst_path)
        """Rename the log file first"""
        if dst_name is None:
            shutil.move(src_path, dst_path)
        else:
            new_src_path = os.path.join(cls.log_path,dst_name)
            os.rename(src_path, new_src_path)
            shutil.move(new_src_path, dst_path)
    @classmethod
    def log_matrix(cls,i_str=None,i_matrix=None):
        """Print and log matrix for easy looking"""
        assert isinstance(i_matrix,np.ndarray)
        assert len(i_matrix.shape)==2
        height, width = i_matrix.shape
        dtype = i_matrix.dtype
        int_types = (np.int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)
        cls.log('-' * 50)
        cls.log(i_str)
        for h in range(height):
            row = None
            for w in range(width):
                if row is None:
                    if dtype in int_types:
                        row = '{}'.format(i_matrix[h,w])
                    else:
                        assert dtype in (np.float,np.float32,np.float64)
                        row = '{:3.3f}'.format(i_matrix[h,w])
                else:
                    if dtype in int_types:
                        row = '{}\t{}'.format(row, i_matrix[h,w])
                    else:
                        assert dtype in (np.float,np.float32,np.float64)
                        row = '{}\t{:3.3f}'.format(row, i_matrix[h,w])
            cls.log(row)
        cls.log('-'*50)
        return True
    @classmethod
    def log_cls_params(cls,i_cls):
        """i_cls is self or an instance of a class"""
        params = i_cls.__dict__
        return cls.log(params)
    @classmethod
    def read_log(cls,i_log_path=None):
        """i_log_path is a path to a log file."""
        """Return a list of lines (text) from the log file."""
        assert isinstance(i_log_path,str)
        assert os.path.exists(i_log_path)
        assert i_log_path.endswith('.txt')
        rtn_lines = []
        with open(i_log_path,'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.rstrip()
                rtn_lines.append(line)
        return rtn_lines
"""=================================================================================================================="""
if __name__ == "__main__":
    print('This module is to implement logging method during running program')
    Logs.log('Hello world@')
    Logs.log({'a':1,'b':2,'c':'Finished!'})
    imatrix = np.random.randint(0,10,(5,5),dtype=np.int)
    Logs.log_matrix('Matrix',i_matrix=imatrix.astype(np.float))
"""=================================================================================================================="""