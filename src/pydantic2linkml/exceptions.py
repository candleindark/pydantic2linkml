from collections.abc import Iterable
from typing import Optional

# noinspection PyProtectedMember
from pydantic._internal._core_utils import CoreSchemaOrField


class NameCollisionError(Exception):
    """
    Raise when there is a name collision
    """


class UserError(Exception):
    """
    Raise when an entity is not used correctly and other more precise exceptions
    are not appropriate
    """


class GeneratorReuseError(UserError):
    """
    Raise when a generator object is reused
    """

    def __init__(self, generator):
        """
        :param generator: The generator object that is reused
        """
        super().__init__(
            f"{type(generator).__name__} generator object cannot be reused"
        )


class TranslationNotImplementedError(NotImplementedError):
    """
    Raise when the translation of a Pydantic core schema to LinkMK is not implemented

    Note: This is used to mark the translation methods of Pydantic core schemas that
      are deemed to be not necessary for use of this translation tool in general or
      against the targeted models expressed in Pydantic. File an issue if this error is
      encountered.
    """

    def __init__(self, schema: CoreSchemaOrField):
        """
        :param schema: The Pydantic core schema of which translation to LinkML is not
            implemented
        """
        super().__init__(
            f"Translation of Pydantic core schema, {schema['type']}, is not "
            "implemented. If you encounter this error in translating your models, "
            "consider filing an issue."
        )


class SlotUsageGenerationError(Exception):
    """
    Raise when a slot usage entry cannot be generated to make a given base slot
    definition function like a given target slot definition.

    A ``slot_usage`` entry can extend the base slot with new properties (meta
    slots), override the base slot's non-constraint properties (e.g.,
    ``title``, ``description``), and refine the base slot's constraint
    properties (those defined in ``SlotExpression``) in ways that are
    recognized as safe monotonic tightenings. The set of recognized
    refinements is defined by ``_is_allowed_constraint_refinement`` in
    ``pydantic2linkml.tools``. Any other change in a constraint property,
    or any meta slot that exists in the base but not in the target, makes
    the situation unrepresentable as a ``slot_usage`` entry and triggers
    this error.
    """

    def __init__(
        self,
        missing_meta_slots: Optional[Iterable[str]] = None,
        disallowed_varied_constraint_meta_slots: Optional[Iterable[str]] = None,
    ):
        """
        :param missing_meta_slots: The input for setting
            ``self.missing_meta_slots``, which is a list of the items
            provided in this input sorted case-insensitively. These items
            are the meta slots that exist in the base slot definition but
            not in the target slot definition. If None or not provided,
            an empty list is used.
        :param disallowed_varied_constraint_meta_slots: The input for
            setting ``self.disallowed_varied_constraint_meta_slots``,
            which is a list of the items provided in this input sorted
            case-insensitively. These items are the constraint meta slots
            (i.e., those defined in ``SlotExpression``) that exist in
            both the base and target slot definitions, have different
            values, and whose (base, target) change is not one of the
            allowed monotonic refinements that can be safely emitted as a
            ``slot_usage`` entry. (Constraint meta slots whose change
            is an allowed refinement, as defined by
            ``_is_allowed_constraint_refinement`` in
            ``pydantic2linkml.tools``, do not appear here.) If None or
            not provided, an empty list is used.
        :raises ValueError: If both `missing_meta_slots` and
            `disallowed_varied_constraint_meta_slots` are empty
        """

        def _sort(items: Optional[Iterable[str]]) -> list[str]:
            return sorted(items, key=str.casefold) if items is not None else []

        sorted_missing: list[str] = _sort(missing_meta_slots)
        sorted_disallowed_varied_constraint: list[str] = _sort(
            disallowed_varied_constraint_meta_slots
        )

        if len(sorted_missing) + len(sorted_disallowed_varied_constraint) == 0:
            error_msg = (
                "At least one of `missing_meta_slots` and "
                "`disallowed_varied_constraint_meta_slots` must be non-empty."
            )
            raise ValueError(error_msg)

        super().__init__()

        self.missing_meta_slots: list[str] = sorted_missing
        self.disallowed_varied_constraint_meta_slots: list[str] = (
            sorted_disallowed_varied_constraint
        )

    def __str__(self):
        return (
            f"Target slot definition has missing meta slots, "
            f"{self.missing_meta_slots}, and disallowed varied constraint "
            f"meta slots, {self.disallowed_varied_constraint_meta_slots}"
        )

    def __repr__(self):
        return (
            f"{type(self).__name__}"
            f"(missing_meta_slots={self.missing_meta_slots!r}, "
            f"disallowed_varied_constraint_meta_slots="
            f"{self.disallowed_varied_constraint_meta_slots!r})"
        )


class YAMLContentError(ValueError):
    """
    Raise when the content of a YAML file is not what is expected
    """


class InvalidLinkMLSchemaError(ValueError):
    """
    Raised when a YAML string does not conform to the LinkML meta schema
    (e.g. unknown field names or wrong-type values)
    """
