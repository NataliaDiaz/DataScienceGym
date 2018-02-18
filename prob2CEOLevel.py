
def minLevel(levels, listOfLevels):
    maxLevel = -1
    nEmployeesLevelBelow = 0
    levels2employees = {}
    #print listOfLevels
    for pair in listOfLevels:
        amount, level = pair[0], pair[1]
        if level > maxLevel:
            maxLevel = level
        levels2employees[level] = amount
    belowCEOLevel = maxLevel
    while not levels2employees[belowCEOLevel]:
        belowCEOLevel -= 1
    if maxLevel + 1 < levels2employees[belowCEOLevel]:
        return levels2employees[maxLevel]
    else:
        return maxLevel+1

cases = int(raw_input())  # read a line with a single integer
#print 'cases: ',cases
for i in xrange(1, cases+1):
    nLevels = int(raw_input().split(" ")[0])
    listOfLevels = []
    #print 'nLevels: ',nLevels
    for level in range(nLevels): #      listOfLevels.append([n, experts]) #
        listOfLevels.append([int(s) for s in raw_input().split(" ")])  # read a list of integers, 2 in this case
    CEOLevel = minLevel(nLevels, listOfLevels)
    print "Case #{}: {} ".format(i, CEOLevel)
    # check out .format's specification for more formatting options
