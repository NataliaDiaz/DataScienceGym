import sys

def get_min_players_can_touch_ball(inputfile='./data/players.csv'):
    n = 0
    #m = create_adjacency_matrix(inputfile)
    m = []
    visibles = {}
    useful_players = set()
    with open(inputfile,'r') as f:
        output = f.read()
    # Create adjacency matrix within a dictionary
    for line in output.split('\n'):
        if len(line)>0:
            # use unpacking to have flexible nr of elements
            names =  line.split(',') # print names, type(names)
            names = [x.strip() for x in names]
            names = [x for x in names if len(x)>0]
            name, visible_players = names[0], names[1:]
            if name in visibles:
                print "Wrong format in input file: each player must appear at the beginning of each line only once"
                sys.exit(-1)
            if not ',' in line:
                print "Wrong format in input file: each line must be separated by commas if that player is seeing any other player"
                sys.exit(-1)
            visibles[name] = visible_players
    print "Finding maximum for players: \n",visibles
    # Finding pairs of
    for player in visibles.keys():
        for visible_player in visibles[player]:
            if visible_player in visibles and player in visibles[visible_player]:
                useful_players.add(player)
    n = len(useful_players)
    print n
    return n
print "\n\nTest 1"
get_min_players_can_touch_ball('./data/players.csv') # 3
print "\n\nTest 2"
get_min_players_can_touch_ball('./data/players1.csv') # 4
print "\n\nTest 3"
get_min_players_can_touch_ball('./data/players2.csv') # 6
print "\n\nTest 4"
get_min_players_can_touch_ball('./data/players3.csv') # 3
print "\n\nTest 5"
get_min_players_can_touch_ball('./data/players4.csv') # Error, wrong input

"""
Other possibility would be:
def create_adjacency_matrix(inputfile):
    # Reads a file and creates an adjacency table with each player in each row and each player in each column.
    # The value of m[i][j] is 0 if no bidirectional relation in the graph exists among player i and j, and 1 otherwise.
    m = []
    # Other alternative with a 2D array
    return m
"""
