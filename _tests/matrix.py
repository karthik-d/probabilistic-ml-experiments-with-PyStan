rows = int(input("Rows: "))
cols = 2**rows;
upper = [[0 for y in range(cols)] for x in range(rows)]
flags = [0 for x in range(rows)];
for i in range(1,rows+1):
    for j in range(1,cols+1):
        upper[i-1][j-1] = flags[i-1];
        if(j%(2**(i-1))==0):
            flags[i-1] = (flags[i-1]+1)%2;  # Toggle 0 and 1
for x in upper:
    print(x)

final = [[0 for y in range(cols)] for x in range(2*rows)]
lower = [[0 for y in range(cols)] for x in range(rows)]
ones =  [[1 for y in range(cols)] for x in range(rows)]

for i in range(1,rows+1):
    for j in range(1,cols+1):
        lower[i-1][j-1] = ones[i-1][j-1] - upper[i-1][j-1]
print()
for x in lower:
    print(x)
