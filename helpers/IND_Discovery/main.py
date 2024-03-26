
def Convert(lst):
    res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}
    return res_dct

def read_file():
    name = 'tpcc'
    file1 = open(name, 'r')
    Lines = file1.readlines()
    line_num = 0
    column_it = 0
    ind_it = 0
    table=[]
    column=[]
    ind=[]
    for line in Lines:
        newline = line.replace("\n","")
        if newline == '# COLUMN':
            column_it = line_num
        if newline == '# RESULTS':
            ind_it = line_num
        if column_it == 0:
            if newline != '# TABLES':
                table.append(newline.split("\t"))
        elif (line_num>column_it ) & (ind_it == 0):
            y = newline.replace(".", "\t")
            column.append(y.split("\t"))
        elif line_num > ind_it:
            if newline != '# COLUMN':
                ind.append(newline.split("[="))
        line_num = line_num + 1

    file1.close()

    newCol=[]
    for t in table:
        for c in column:
            if c[0] in t:
                newCol.append([c[2],c[1],t[0]])
    pairs=[]
    for i in ind:
        for c in newCol:
            if i[0] == c[0]:
                fk=c[1:3]
                A = "http://kglids.org/resource/kaggle/"+name+"/dataResource/"+fk[1]+"/"+fk[0]
            if i[1] == c[0]:
                pk = c[1:3]
                B = "http://kglids.org/resource/kaggle/"+name+"/dataResource/"+pk[1]+"/"+pk[0]
        # pairs.append([fk[1].replace('.csv',''),fk[0],A,pk[1].replace('.csv',''),pk[0],B])
        pairs.append([fk[1].replace('.csv', ''), fk[0], A, pk[1].replace('.csv', ''), pk[0], B])


    f = open("sawfish-"+name+".csv", "w")
    f.write("Foreign_table,Foreign_key,A,Primary_table,Primary_key,B\n")
    for p in pairs:
        pClean=str(p)
        specialChars = "[]' "
        for specialChar in specialChars:
            pClean=pClean.replace(specialChar,"")

        f.write(pClean)
        f.write("\n")
    f.close()

if __name__ == '__main__':
    read_file()

