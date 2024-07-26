class NameCollisionError(Exception):
    """
    Raise when there is a name collision
    """

    pass


class UserError(Exception):
    """
    Raise when an entity is not used correctly and other more precise exceptions
    are not appropriate
    """

    pass
