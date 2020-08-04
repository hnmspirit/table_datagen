'''
The code for generating 4 categories of tables consists of several small pieces e.g. types of borders,
irregular/regular headers and transformations.

Spanning headers:
+ 0: colspan headers only
+ 1: colspan + rowspan headers

We define border_categories with 4 possibilities:
1. border all
2. border none
3. border random
'''

import random
import numpy as np
from TableGeneration.Distribution import Distribution
import time

BORDER_CATS = ['border', 'border-top', 'border-bottom', 'border-left', 'border-right']


class Table:

    def __init__(self, no_of_rows, no_of_cols, images_path, ocr_path, gt_table_path, assigned_category, distributionfile, max_cel_len=40):

        #get distribution of data
        self.distribution = Distribution(images_path,ocr_path,gt_table_path,distributionfile)
        self.all_words, self.all_numbers, self.all_others = self.distribution.get_distribution()
        self.assigned_category = assigned_category
        self.max_cel_len = max_cel_len

        self.no_of_rows = no_of_rows
        self.no_of_cols = no_of_cols

        self.spanflag = False
        self.header_cat = np.random.choice([0,1])
        if self.assigned_category==1:
            self.border_cat = 'border'
        elif self.assigned_category==2:
            self.border_cat = 'none'
        elif self.assigned_category==3:
            self.spanflag = True
            self.border_cat = np.random.choice(BORDER_CATS)
        else:
            self.spanflag = random.choice([True, False])
            self.border_cat = np.random.choice(BORDER_CATS)

        self.idcounter = 0

        '''cell_types matrix have two possible values: 'n' and 'w' where 'w' means word and 'n' means number'''
        self.cell_types = np.chararray(shape=(self.no_of_rows,self.no_of_cols))

        '''headers matrix have two possible values: 's' and 'h' where 'h' means header and 's' means simple text'''
        self.headers = np.chararray(shape=(self.no_of_rows,self.no_of_cols))

        '''A positive value at a position in matrix shows the number of columns to span and -1 will show to skip that cell as part of spanned cols'''
        self.col_spans_matrix = np.zeros(shape=(self.no_of_rows,self.no_of_cols))

        '''A positive value at a position means number of rows to span and -1 will show to skip that cell as part of spanned rows'''
        self.row_spans_matrix = np.zeros(shape=(self.no_of_rows,self.no_of_cols))

        '''missing_cells will contain a list of (row,column) pairs where each pair would show a cell where no text should be written'''
        self.missing_cells = []

        #header_count will keep track of how many top rows and how many left columns are being considered as headers
        self.header_count = {'r':2,'c':0}

        '''This matrix is essential for generating same cell, same row and same col matrices. Because this
        matrix holds the list of word ids in each cell of the table'''
        self.data_matrix = np.empty(shape=(self.no_of_rows,self.no_of_cols),dtype=object)


    def define_col_types(self):
        '''
        We define the data type that will go in each column. We categorize data in three types:
        1. 'n': Numbers
        2. 'w': word
        3. 'r': other types (containing special characters)

        '''
        len_all_words = len(self.all_words)
        len_all_numbers = len(self.all_numbers)
        len_all_others = len(self.all_others)

        total = len_all_words+len_all_numbers+len_all_others

        prob_words = len_all_words / total
        prob_numbers = len_all_numbers / total
        prob_others =len_all_others / total

        for i, wtype in enumerate(np.random.choice(['n','w','r'], p=[prob_numbers,prob_words,prob_others], size=self.no_of_cols)):
            self.cell_types[:,i] = wtype

        '''The headers should be of type word'''
        self.cell_types[0:2,:] = 'w'

        '''All cells should have simple text but the headers'''
        self.headers[:] = 's'
        self.headers[0:2, :] = 'h'


    def generate_random_text(self, type):
        html = ''
        ids = []
        text_len = random.randrange(1,2)
        if(type == 'n'):
            out = random.sample(self.all_numbers, text_len)
        elif(type =='r'):
            out = random.sample(self.all_others, text_len)
        else:
            out = random.sample(self.all_words, text_len)

        cel_len = 0
        for i, e in enumerate(out):
            if i > 0:
                html += '&nbsp;'*random.randrange(2,4)
            html += '<span id=c{}>{}</span>'.format(self.idcounter, e)
            ids.append(self.idcounter)
            self.idcounter += 1
            cel_len += len(e)

            if cel_len > self.max_cel_len:
                break
        return html, ids


    def agnostic_span_indices(self, maxvalue):
        '''Spans indices. Can be used for row or col span
        Span indices store the starting indices of row or col spans while span_lengths will store
        the length of span (in terms of cells) starting from start index.'''
        span_indices = []
        span_lengths = []
        span_count = random.randint(1, 3)
        if span_count > maxvalue:
            return [],[]

        indices = sorted(random.sample(list(range(0, maxvalue)), span_count))

        starting_index = 0
        for i, index in enumerate(indices):
            if starting_index > index:
                continue

            max_lengths = maxvalue - index
            if max_lengths < 2:
                break
            len_span = random.randint(2, max_lengths)
            span_lengths.append(len_span)
            span_indices.append(index)
            starting_index = index + len_span

        return span_indices, span_lengths


    def make_colspan_headers(self, ratio=0.4):
        '''This function spans header cells'''
        if self.header_cat == 1:
            # except first and last col
            header_span_indices, header_span_lengths = self.agnostic_span_indices(self.no_of_cols - 2)
            header_span_indices = [x+1 for x in header_span_indices]
        else:
            # except last col
            header_span_indices, header_span_lengths = self.agnostic_span_indices(self.no_of_cols - 1)

        row_span_indices = []
        n = self.no_of_rows - 1
        m = int(ratio * n)
        for index,length in zip(header_span_indices,header_span_lengths):
            self.col_spans_matrix[0, index] = length
            self.col_spans_matrix[0, index+1:index+length] = -1
            row_span_indices += list(range(index,index+length))

            ids = np.random.choice(n, m, replace=False) + 1
            self.col_spans_matrix[ids, index] = length
            self.col_spans_matrix[ids, index+1:index+length] = -1

        b = [x for x in range(self.no_of_cols) if x not in row_span_indices]
        self.row_spans_matrix[0,b] = 2
        self.row_spans_matrix[1,b] = -1

        #If the table has irregular headers, then we can span some of the rows in those header cells
        if self.header_cat == 1:
            self.make_rowspan_headers()


    def make_rowspan_headers(self):
        '''To make some random row spans for headers on first col of each row'''
        colnumber = 0
        # except first 2 row
        span_indices, span_lengths = self.agnostic_span_indices(self.no_of_rows-3)
        span_indices = [x+2 for x in span_indices]

        for index, length in zip(span_indices, span_lengths):
            self.row_spans_matrix[index, colnumber] = length
            self.row_spans_matrix[index+1:index+length, colnumber] = -1
        self.headers[:,colnumber] = 'h'
        self.header_count['c'] += 1


    def generate_missing_cells(self, ratio=0.4):
        '''This is randomly select some cells to be empty (not containing any text)'''
        n = (self.no_of_rows-2) * (self.no_of_cols-1)
        m = int(ratio * n)
        ids = np.random.choice(n, m, replace=False)
        cols = ids % (self.no_of_cols-1) + 1
        rows = ids // (self.no_of_cols-1) + 2
        for i in range(len(ids)):
            self.missing_cells.append((rows[i], cols[i]))


    def create_style(self):
        '''This function will dynamically create stylesheet. This stylesheet essentially creates our specific
        border types in tables'''

        style = "<head><style>"
        style += "html{width:1366px;height:768px;background-color: white;} table{"

        # random center align
        if (random.random() < 0.5) or self.spanflag:
            style += "text-align:center;"

        style += """border-collapse:collapse;} td,th{padding:4px;padding-left:10px;padding-right:10px;"""

        if self.border_cat == 'none':
            style += """}"""
        else:
            style += self.border_cat + ':1px solid black;}'

        style += "</style></head>"
        return style


    def create_html(self):
        temparr = ['td', 'th']
        html = """<html>"""
        html += self.create_style()
        html += """<body><table>"""
        for r in range(self.no_of_rows):
            html += '<tr>'
            for c in range(self.no_of_cols):
                row_span_value = int(self.row_spans_matrix[r, c])
                col_span_value = int(self.col_spans_matrix[r, c])
                htmlcol = temparr[['s', 'h'].index(self.headers[r][c].decode('utf-8'))]

                if (row_span_value == -1):
                    self.data_matrix[r, c] = self.data_matrix[r - 1, c]
                    continue
                elif(row_span_value>0):
                    html += '<' + htmlcol + ' rowspan=\"' + str(row_span_value) + '"'
                else:
                    if(col_span_value==0):
                        if (r, c) in self.missing_cells:
                            html += """<td></td>"""
                            continue
                    if (col_span_value == -1):
                        self.data_matrix[r, c] = self.data_matrix[r, c - 1]
                        continue
                    html += '<' + htmlcol + """ colspan=""" + str(col_span_value)

                out,ids = self.generate_random_text(self.cell_types[r, c].decode('utf-8'))
                html+='>'+out+'</'+htmlcol+'>'

                self.data_matrix[r,c]=ids

            html += '</tr>'

        html+="""</table></body></html>"""
        return html


    def create_same_matrix(self,arr,ids):
        '''Given a list of lists with each list consisting of all ids considered same, this function
         generates a matrix '''
        matrix=np.zeros(shape=(ids,ids))
        for subarr in arr:
            for element in subarr:
                matrix[element,subarr]=1
        return matrix


    def create_same_col_matrix(self):
        '''This function will generate same column matrix from available matrices data'''
        all_cols=[]

        for col in range(self.no_of_cols):
            single_col = []
            for subarr in self.data_matrix[:,col]:
                if(subarr is not None):
                    single_col+=subarr
            all_cols.append(single_col)
        return self.create_same_matrix(all_cols,self.idcounter)


    def create_same_row_matrix(self):
        '''This function will generate same row matrix from available matrices data'''
        all_rows=[]

        for row in range(self.no_of_rows):
            single_row=[]
            for subarr in self.data_matrix[row,:]:
                if(subarr is not None):
                    single_row+=subarr
            all_rows.append(single_row)
        return self.create_same_matrix(all_rows,self.idcounter)


    def create_same_cell_matrix(self):
        '''This function will generate same cell matrix from available matrices data'''
        all_cells=[]
        for row in range(self.no_of_rows):
            for col in range(self.no_of_cols):
                if(self.data_matrix[row,col] is not None):
                    all_cells.append(self.data_matrix[row,col])
        return self.create_same_matrix(all_cells,self.idcounter)


    def create(self):
        '''This will create the complete table'''
        self.define_col_types()
        self.generate_missing_cells()
        if self.spanflag:
            self.make_colspan_headers()

        html = self.create_html()
        #create same row, col and cell matrices
        mat_cel = self.create_same_cell_matrix()
        mat_col = self.create_same_col_matrix()
        mat_row = self.create_same_row_matrix()

        return mat_cel, mat_col, mat_row, self.idcounter, html, self.assigned_category

