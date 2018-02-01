class FSC_Writer(object):
    def __init__(self, output_dir, output_file_prefix):
        self.output_dir = output_dir
        self.output_prefix = output_file_prefix

    def write_table(self, data):
        filename = self.output_prefix + 'globalFSC.csv'
        full_path = os.path.join(self.output_dir, filename)