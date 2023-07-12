step = 4
for processStep in range(0,24,step):
    limit = processStep+step if processStep+step < 24 else 24
    for i in range(processStep,limit):
        print(i)
    print('---')