import sys
"""
Football players in a field can throw a ball between them and we need to find the max nr of players that
can touch a ball after the ball is given to any player at the begining (each player can touch the ball
an unlimited nr of times). If each input in the file contains a comma-separated set of names where the
first field on each line is the name of a player (as a string), and the remaining fields are the names
of the players that that player can see:

player1,visible_player_1,visible_player_2,...
player2,visible_player_1,visible_player_2,...
...

For example, given the following input:
George,Beth, Sue
Rick,Anne
Anne,Beth
Beth, Anne ,George
Sue,Beth
The program should print 3 because George can see Beth and Beth can see George.
Additionally, Beth can see Anne, and Anne can see Beth. However, despite Rick being able to
see Anne, Anne cannot see Rick, and despite George being able to see Sue, Sue cannot see
George.

O(n*m) should be the worse case complexity (n is the nr of rows in the input file, m is the max number
of players that a single player can see).

ToDo: it is max, not total of players, therefore implement with tree traversal and class.
"""

def get_min_players_can_touch_ball(inputfile='./data/players.csv'):
    n = 0
    #m = create_adjacency_matrix(inputfile)
    m = []
    visibles = {}
    useful_players = set()
    options = []
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
                visibles[name].extend(visible_players)
            else:
                visibles[name] = visible_players
            if not ',' in line:
                print "Wrong format in input file: each line must be separated by commas if that player is seeing any other player"
                #print -1;return -1#sys.exit(-1)

    print "Finding maximum nr of players that can have the ball: \n",visibles
    # Finding pairs of players for each connected component (or subgraph of connected players 2-to-2)
    processed = []
    for player in visibles.keys():
        #if not player in processed: # TODO TRAVERSAL TREE
        processed.append(player)
        for visible_player in visibles[player]:
            if visible_player in visibles and player in visibles[visible_player]:
                useful_players.add(player)
        # we add the connected (nr of visible 2-to-2) players in this subgraph
        options.append(len(useful_players))
        useful_players.clear()
    n = max(options)
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
get_min_players_can_touch_ball('./data/players4.csv') # 3
print "\n\nTest 6"
get_min_players_can_touch_ball('./data/players5.csv') # 3


"""
Other possibility would be:
def create_adjacency_matrix(inputfile):
    # Reads a file and creates an adjacency table with each player in each row and each player in each column.
    # The value of m[i][j] is 0 if no bidirectional relation in the graph exists among player i and j, and 1 otherwise.
    m = []
    # Other alternative with a 2D array
    return m
"""
