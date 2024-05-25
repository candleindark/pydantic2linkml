class NameCollisionError(Exception):
    """
    Raised when there is a name collision
    """

    pass


class UserError(Exception):
    """
    Raised when an entity is not used correctly and other more precise exceptions
    are not appropriate
    """

    pass
