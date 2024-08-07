class NameCollisionError(Exception):
    """
    Raise when there is a name collision
    """


class UserError(Exception):
    """
    Raise when an entity is not used correctly and other more precise exceptions
    are not appropriate
    """


class TranslationNotImplementedError(NotImplementedError):
    """
    Raise when the translation of a Pydantic core schema to LinkMK is not implemented

    Note: This is used to mark the translation methods of Pydantic core schemas that
      are deemed to be not necessary for use of this translation tool in general or
      against the targeted models expressed in Pydantic. File an issue if this error is
      encountered.
    """
