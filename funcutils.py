from operator import itemgetter

__all__=["underscore"]

class Underscore:
    """ enable point-free functions """
    def __getattribute__(self, atr):
        """ _.t(a,b,c=d) == lambda x: x.t(a,b,c=d) """
        def _inner(*args, **kwargs):
            return lambda x: x.__getattribute__(atr)(*args, **kwargs)
            
        return _inner
    def __getitem__(self, item):
        """ _[t] == lambda x: x[t] """
        return itemgetter(item)

underscore = Underscore()