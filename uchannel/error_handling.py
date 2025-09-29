class FeasibilityError(Exception):
    """ Raised when a geometry is not feasible for creation. """
    pass

class FilletError(Exception):
    """ Raised when a fillet operation fails. """
    pass

class RemoveHelperEdgesError(Exception):
    """ Raised when a helper edge removal fails. """
    pass

class FragmentError(Exception):
    """ Raised when a fragment operation fails. """
    pass