# coding: utf-8

def get_current_hand_points(self, player):
    if player == HUMAN:
        cards = self.player_cards[:] 
        """ slicing needed to make the full copy of a list """
    elif player == DEALER:
        cards = self.dealer_cards[:]
    else:
        print "wrong player in get_points_sum: ",player
        sys.exit(-1)
    cards.sort(key=lambda x: x.number, reverse=True) 
    """ 
    if assigned to new var, cards becomes None (?) because this sort works 
    in place: cards = cards.sort(key=lambda x:....)
    """
    sum = 0

def copying_and_altering_list():
    originalFeatures = ['trip_id','shopper_id','fulfillment_model','store_id','shopping_started_at','shopping_ended_at']
    originalFeaturesButY = originalFeatures[:]
    originalFeaturesButY = originalFeaturesButY.remove('shopping_ended_at')

    print "originalFeatures ", originalFeatures #['trip_id', 'shopper_id', 'fulfillment_model', 'store_id', 'shopping_started_at', 'shopping_ended_at']
    print "originalFeaturesButY ", originalFeaturesButY  #Original features but Y  None -> Use Pandas for selection of rows of interest!


def on_multiple_permutations():
    import itertools
    #Permutation (order matters):

    print "permutations ",list(itertools.permutations([1,2,3,4], 2))
    [(1, 2), (1, 3), (1, 4),
    (2, 1), (2, 3), (2, 4),
    (3, 1), (3, 2), (3, 4),
    (4, 1), (4, 2), (4, 3)]

    #Combination (order does NOT matter):

    print list(itertools.combinations('123', 2))
    [('1', '2'), ('1', '3'), ('2', '3')]

    #Cartesian product (with several iterables):

    print list(itertools.product([1,2,3], [4,5,6]))
    [(1, 4), (1, 5), (1, 6),
    (2, 4), (2, 5), (2, 6),
    (3, 4), (3, 5), (3, 6)]

    #Cartesian product (with one iterable and itself):

    print list(itertools.product([1,2], repeat=3))
    [(1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 2),
    (2, 1, 1), (2, 1, 2), (2, 2, 1), (2, 2, 2)]




if __name__ == "__main__":
    copying_and_altering_list()
    on_multiple_permutations()

