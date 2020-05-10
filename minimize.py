import csv

c = []
minix = []
miniy = []
with open('/Users/student/desktop/metnet/a.csv', 'rb') as f:
    reader = csv.reader(f)
    s_c = list(reader)
    for i in s_c:
        c.append([float(i[0]) - 38.0, float(i[1]) - 130.0])

for i in c:
    minix.append(i[0])
    miniy.append(i[1])

with open('/Users/student/desktop/metnet/zero.csv', 'wb') as f:
    wr = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    for i in range(len(c)):
        wr.writerow((minix[i], miniy[i]))
