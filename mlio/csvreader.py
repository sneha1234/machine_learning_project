'''
@Author Emmanuel John

A simple CSV file reader.

Example usage:

	csv = CsvReader(filepath, delimeter)
	csv.read()

	print csv.getData()

'''


class CsvReader:

    def __init__(self, filename, delim=","):
        self.filename = filename
        self.data = []
        self.delim = delim
        self.title = None
        self.has_headers = True

    def read(self, headers=True):
        with open(self.filename, "r") as f:
            if headers:
                self.title = f.readline().strip().split(self.delim)
            else:
                self.has_headers = False

            for line in f:
                line = line.strip().split(self.delim)
                if not headers:
                    self.data.append(line)
                else:
                    self.data.append(dict(zip(self.title, line)))

    def getData(self):
        return self.data

    def getRow(self, lineno):
        return self.data[lineno]

    def getColumn(self, col_id):
        col = []
        for line in self.data:
            col.append(line[col_id])
        return col

    # e.g csv.getValueAt(0, "Dates")
    def getValueAt(self, row, col):
        # TODO: handle case where number is passed for col 
        return self.data[row][col]

