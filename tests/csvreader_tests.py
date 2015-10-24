from nose.tools import *
from mlio import csvreader

csv = None
def setup():
    pass

def teardown():
    print("TEAR DOWN!")

def test_basic():
    print("I RAN!")

def test_readData():
	csv = csvreader.CsvReader("/Users/emmanuj/projects/crime_classification/machine_learning_project/tests/sample_csv.csv",",")
	csv.read()
	dataset = csv.getData()
	assert_equals(10, len(dataset))

# TODO: Add more test cases for csv reader

