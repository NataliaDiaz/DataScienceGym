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
  