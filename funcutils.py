from operator import itemgetter

__all__=["underscore",'get']
# inspired by https://github.com/fnpy/fn.py/tree/master

class Underscore:
    """ enable point-free functions """
    def __getattribute__(self, atr):
        """ _.t(a,b,c=d) == lambda x: x.t(a,b,c=d) """
        def _inner(*args, **kwargs):
            return lambda x: x.__getattribute__(atr)(*args, **kwargs)
            
        return _inner

class Getter:
    def __getitem__(self, item):
        """ self[t] == lambda x: x[t] """
        return itemgetter(item)

    def __getattribute__(self, atr):
        """ self.t == lambda x: x['t'] """
            
        return itemgetter(atr)

underscore = Underscore()
get = Getter()