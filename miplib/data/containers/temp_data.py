import datetime
import os
import tempfile

import miplib.data.io.tiffile


class TempData():

    def __init__(self, directory=None):
        if directory is None:
            self.dir = tempfile.mkdtemp('-miplib.temp.data')
        else:
            date_now = datetime.datetime.now().strftime("%y_%m_%d_")
            self.dir = '{}_supertomo_temp_data'.format(date_now)
            if not os.path.exists(self.dir):
                os.mkdir(self.dir)
        self.data_file = None

    def create_data_file(self, filename, col_names, append=False):
        data_file_name = os.path.join(self.dir, filename)
        self.data_file = RowFile(data_file_name,
                                 titles=col_names,
                                 append=append)

    def write_comment(self, comment):
        self.data_file.comment(comment)

    def write_row(self, data):
        self.data_file.write(data)

    def save_image(self, data, filename):
        image_path = os.path.join(self.dir, filename)
        miplib.data.io.tiffile.imsave(image_path, data)

    def close_data_file(self):
        self.data_file.close()

    def read_data_file(self):
        self.close_data_file()
        return self.data_file.read(with_titles=True)


class RowFile:
    """
    Represents a row file.

    The RowFile class is used for creating and reading row files.

    The format of the row file is the following:
    - row file may have a header line containg the titles of columns
    - lines starting with ``#`` are ignored as comment lines
    """

    def __init__(self, filename, titles = None, append=False):
        """
        Parameters
        ----------

        filename : str
          Path to a row file
        titles : {None, list}
          A list of column headers for writing mode.
        append : bool
          When True, new data will be appended to row file.
          Otherwise, the row file will be overwritten.
        """
        self.filename = filename
        dirname = os.path.dirname(self.filename)
        if not os.path.exists(dirname) and dirname:
            os.makedirs(dirname)
        self.file = None
        self.nof_cols = 0
        self.append = append
        self.extra_titles = ()
        if titles is not None:
            self.header(*titles)

        self.data_sep = ', '

    def __del__ (self):
        if self.file is not None:
            self.file.close()

    def header(self, *titles):
        """
        Write titles of columns to file.
        """
        data = None
        extra_titles = self.extra_titles
        if self.file is None:
            if os.path.isfile(self.filename) and self.append:
                data_file = RowFile(self.filename)
                data, data_titles = data_file.read(with_titles=True)
                data_file.close()
                if data_titles!=titles:
                    self.extra_titles = extra_titles = tuple([t for t in data_titles if t not in titles])
            self.file = open(self.filename, 'w')
            self.nof_cols = len(titles + extra_titles)
            self.comment('@,@'.join(titles + extra_titles))
            self.comment('To read data from this file, use ioc.microscope.data.RowFile(%r).read().' % (self.filename))

            if data is not None:
                for i in range(len(data[data_titles[0]])):
                    data_line = []
                    for t in titles + extra_titles:
                        if t in data_titles:
                            data_line.append(data[t][i])
                        else:
                            data_line.append(0)
                    self.write(*data_line)

    def comment (self, msg):
        """
        Write a comment to file.
        """
        if self.file is not None:
            self.file.write ('#%s\n' % msg)
            self.file.flush ()

    def write(self, *data):
        """
        Write a row of data to file.
        """
        if len (data) < self.nof_cols:
            data = data + (0, ) * (self.nof_cols - len (data))
        assert len(data) == self.nof_cols
        self.file.write(', '.join(str(i).strip('[]') for i in data) + '\n')
        self.file.flush()

    def _get_titles (self, line):
        if line.startswith('"'): # csv file header
            self.data_sep = '\t'
            return tuple([t[1:-1] for t in line.strip().split('\t')])
        return tuple([t.strip() for t in line[1:].split('@,@')])

    def read(self, with_titles = False):
        """
        Read data from a row file.

        Parameters
        ----------
        with_titles : bool
          When True, return also column titles.

        Returns
        -------
        data : dict
          A mapping of column values.
        titles : tuple
          Column titles.
        """
        f = open (self.filename, 'r')
        titles = None
        d = {}
        for line in f.readlines():
            if titles is None:
                titles = self._get_titles(line)
                for t in titles:
                    d[t] = []
                continue
            if line.startswith ('#'):
                continue
            data = line.strip().split(self.data_sep)
            for i, t in enumerate (titles):
                try:
                    v = float(data[i])
                except (IndexError,ValueError):
                    v = 0.0
                d[t].append(v)
        f.close()
        if with_titles:
            return d, titles
        return d

    def close (self):
        """
        Close row file.
        """
        if self.file is not None:
            print('Closing ',self.filename)
            self.file.close ()
            self.file = None
