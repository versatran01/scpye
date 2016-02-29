# -*- coding: utf-8 -*-

from openpyxl import Workbook, load_workbook
import pandas as pd


xlsx_file = "/home/chao/Workspace/bag/apple/ground_truth_2015.xlsx"
csv_file = "/home/chao/Workspace/bag/apple/ground_truth_2015.csv"
#wb = load_workbook(xlsx_file, read_only=True)
#ws = wb['Ground Truth']
#
#for row in ws.rows:
#    for cell in row:
#        print(cell.value)
data = pd.read_csv(csv_file)