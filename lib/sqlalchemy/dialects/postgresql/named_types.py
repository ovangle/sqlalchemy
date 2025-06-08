# dialects/postgresql/named_types.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors
from __future__ import annotations

from types import ModuleType
from typing import Any
from typing import cast
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

from ... import util
from ... import engine
from ...sql import coercions
from ...sql import ddl
from ...sql import elements
from ...sql import roles
from ...sql import schema
from ...sql import sqltypes
from ...sql import type_api
from ...sql.base import _NoArg
from ...sql.ddl import SchemaGenerator
from ...sql.ddl import SchemaDropper
from ...sql.schema import Column
from ...sql.schema import MetaData
from ...sql.schema import Table

if TYPE_CHECKING:
    from ...sql._typing import _CreateDropBind
    from ...sql._typing import _TypeEngineArgument

    from .base import PGDialect


class NamedType(
    sqltypes.SchemaType, schema.SchemaVisitable, type_api.TypeEngineMixin
):
    """Base for named types."""

    __abstract__ = True

    table: Table | None
    column: Column | None

    def create(
        self, bind: _CreateDropBind, checkfirst: bool = True, **kw: Any
    ) -> None:
        """Emit ``CREATE`` DDL for this type.

        :param bind: a connectable :class:`_engine.Engine`,
         :class:`_engine.Connection`, or similar object to emit
         SQL.
        :param checkfirst: if ``True``, a query against
         the PG catalog will be first performed to see
         if the type does not exist already before
         creating.

        """
        bind._run_ddl_visitor("create", self, checkfirst=checkfirst)

    def drop(
        self, bind: _CreateDropBind, checkfirst: bool = True, **kw: Any
    ) -> None:
        """Emit ``DROP`` DDL for this type.

        :param bind: a connectable :class:`_engine.Engine`,
         :class:`_engine.Connection`, or similar object to emit
         SQL.
        :param checkfirst: if ``True``, a query against
         the PG catalog will be first performed to see
         if the type actually exists before dropping.

        """
        bind._run_ddl_visitor("drop", self, checkfirst=checkfirst)


class PGDDLGenerator(SchemaGenerator):
    @property
    def is_metadata_operation(self):
        return isinstance(self.target, MetaData)

    @property
    def is_table_operation(self):
        return isinstance(self.target, Table)

    @property
    def is_type_operation(self):
        return isinstance(self.target, type_api.TypeEngine)

    def _can_create_type(self, typ: NamedType):
        effective_schema = self.connection.schema_for_object(typ)
        return (
            (self.is_type_operation or typ.create_type)
            and (
                not self.checkfirst
                # Prefer raising a compilation error later
                or typ.name is None
                or not cast("PGDialect", self.dialect).has_type(
                    self.connection,
                    typ.name,
                    schema=effective_schema
                )
            )
        )

    def visit_metadata(self, metadata: MetaData):
        schema_types = {
            resolved_t._key: resolved_t
            for t in metadata._types.values()
            if (
                t.column is None and
                (
                    resolved_t := resolve_named_type(t, self.connection.dialect)
                ) is not None
                and self._can_create_type(resolved_t)
            )
        }

        for t in schema_types.values():
            self.traverse_single(t, create_ok=True)

        return super().visit_metadata(metadata)

    def visit_table(self, table: Table, create_ok=False, **kwargs):
        if not create_ok and not self._can_create_table(table):
            return

        metadata = table.metadata
        assert metadata is not None

        table_types = {}

        for col in table.columns:
            t = resolve_named_type(col.type, self.connection.dialect)

            if t is None or t._key in table_types:
                continue

            try:
                metadata_type = metadata._types[t._key]
            except KeyError:
                continue

            should_include = (
                # If the registered type was first used by a
                # column of this table and is never used on a
                # different table, then the type is considered internal
                # to the table and is generated here.
                metadata_type.table is table
            ) or (
                # Otherwise, the type is declared as part of
                # multiple table definitions. If we are running
                # via Table.create and can check for existence
                # of the type first, then it is safe to create
                self.checkfirst and self.is_table_operation
            )

            if should_include and self._can_create_type(t):
                table_types[t._key] = t

        for t in table_types.values():
            self.traverse_single(t, create_ok=True)

        return super().visit_table(table, create_ok=create_ok, **kwargs)

    def visit_enum(self, enum, create_ok=False):
        if not create_ok and not self._can_create_type(enum):
            return

        if not self.dialect.supports_native_enum:
            return

        with self.with_ddl_events(enum):
            CreateEnumType(enum)._invoke_with(self.connection)

    def visit_DOMAIN(self, domain, create_ok=False):
        if not create_ok and not self._can_create_type(domain):
            return

        with self.with_ddl_events(domain):
            CreateDomainType(domain)._invoke_with(self.connection)


class PGDDLDropper(SchemaDropper):
    def __init__(self, connection, target, **kwargs):
        super().__init__(connection, target, **kwargs)
        self._table_ignore_types = set()

    @property
    def is_metadata_operation(self):
        return isinstance(self.target, MetaData)

    @property
    def is_table_operation(self):
        return isinstance(self.target, Table)

    @property
    def is_type_operation(self):
        return isinstance(self.target, type_api.TypeEngine)

    def _can_drop_type(self, typ):
        effective_schema = self.connection.schema_for_object(typ)

        return (
            (self.is_type_operation or typ.create_type)
            and typ.name is not None
            and (
                not self.checkfirst
                or cast("PGDialect", self.dialect).has_type(
                    self.connection,
                    typ.name,
                    schema=effective_schema
                )
            )
        )

    def visit_metadata(self, metadata: MetaData):
        def resolve_type(t: Any):
            return resolve_named_type(t, self.connection.dialect)

        super().visit_metadata(metadata)

        schema_types = {
            resolved_t._key: resolved_t
            for t in metadata._types.values()
            if (
                (resolved_t := resolve_type(t)) is not None
                and self._can_drop_type(resolved_t)
            )
        }
        assert not self._table_ignore_types
        self._table_ignore_types = schema_types

        for t in schema_types.values():
            self.traverse_single(t, drop_ok=True)

    def visit_table(self, table: Table, drop_ok: bool = False, **kwargs):
        if not drop_ok and not self._can_drop_table(table):
            return

        super().visit_table(table, drop_ok=drop_ok, **kwargs)

        metadata = table.metadata
        assert metadata is not None

        table_types = {}

        for col in table.columns:
            t = resolve_named_type(col.type, self.connection.dialect)

            if (
                t is None
                or t._key in table_types
                or t._key in self._table_ignore_types
            ):
                continue

            try:
                metadata_type = metadata._types[t._key]
            except KeyError:
                continue

            # Unlike generate, we only ever drop types
            # which are strictly internal to the table
            # and we only ever drop them if invoked via
            # `Table.create` as we do not know whether
            # there are still tables in the schema which
            # depend on the type.
            if (
                self.is_table_operation
                and cast(NamedType, metadata_type).table is table
                and self._can_drop_type(t)
            ):
                table_types[t._key] = t

        for t in table_types.values():
            self.traverse_single(t, drop_ok=True)

    def visit_DOMAIN(self, domain: DOMAIN, drop_ok=False):
        if not drop_ok and not self._can_drop_type(domain):
            return

        with self.with_ddl_events(domain):
            DropDomainType(domain)._invoke_with(self.connection)

    def visit_enum(self, enum: ENUM, drop_ok=False):
        if not drop_ok and not self._can_drop_type(enum):
            return

        with self.with_ddl_events(enum):
            DropEnumType(enum)._invoke_with(self.connection)


class ENUM(type_api.NativeForEmulated, sqltypes.Enum, NamedType):
    """PostgreSQL ENUM type.

    This is a subclass of :class:`_types.Enum` which includes
    support for PG's ``CREATE TYPE`` and ``DROP TYPE``.

    When the builtin type :class:`_types.Enum` is used and the
    :paramref:`.Enum.native_enum` flag is left at its default of
    True, the PostgreSQL backend will use a :class:`_postgresql.ENUM`
    type as the implementation, so the special create/drop rules
    will be used.

    The create/drop behavior of ENUM is necessarily intricate, due to the
    awkward relationship the ENUM type has in relationship to the
    parent table, in that it may be "owned" by just a single table, or
    may be shared among many tables.

    When using :class:`_types.Enum` or :class:`_postgresql.ENUM`
    in an "inline" fashion, the ``CREATE TYPE`` and ``DROP TYPE`` is emitted
    corresponding to when the :meth:`_schema.Table.create` and
    :meth:`_schema.Table.drop`
    methods are called::

        table = Table(
            "sometable",
            metadata,
            Column("some_enum", ENUM("a", "b", "c", name="myenum")),
        )

        table.create(engine)  # will emit CREATE ENUM and CREATE TABLE
        table.drop(engine)  # will emit DROP TABLE and DROP ENUM

    To use a common enumerated type between multiple tables, the best
    practice is to declare the :class:`_types.Enum` or
    :class:`_postgresql.ENUM` independently, and associate it with the
    :class:`_schema.MetaData` object itself::

        my_enum = ENUM("a", "b", "c", name="myenum", metadata=metadata)

        t1 = Table("sometable_one", metadata, Column("some_enum", myenum))

        t2 = Table("sometable_two", metadata, Column("some_enum", myenum))

    When this pattern is used, care must still be taken at the level
    of individual table creates.  Emitting CREATE TABLE without also
    specifying ``checkfirst=True`` will still cause issues::

        t1.create(engine)  # will fail: no such type 'myenum'

    If we specify ``checkfirst=True``, the individual table-level create
    operation will check for the ``ENUM`` and create if not exists::

        # will check if enum exists, and emit CREATE TYPE if not
        t1.create(engine, checkfirst=True)

    When using a metadata-level ENUM type, the type will always be created
    and dropped if either the metadata-wide create/drop is called::

        metadata.create_all(engine)  # will emit CREATE TYPE
        metadata.drop_all(engine)  # will emit DROP TYPE

    The type can also be created and dropped directly::

        my_enum.create(engine)
        my_enum.drop(engine)

    """

    native_enum = True

    def __init__(
        self,
        *enums,
        name: Union[str, _NoArg, None] = _NoArg.NO_ARG,
        create_type: bool = True,
        **kw,
    ):
        """Construct an :class:`_postgresql.ENUM`.

        Arguments are the same as that of
        :class:`_types.Enum`, but also including
        the following parameters.

        :param create_type: Defaults to True.
         Indicates that ``CREATE TYPE`` should be
         emitted, after optionally checking for the
         presence of the type, when the parent
         table is being created; and additionally
         that ``DROP TYPE`` is called when the table
         is dropped.    When ``False``, no check
         will be performed and no ``CREATE TYPE``
         or ``DROP TYPE`` is emitted, unless
         :meth:`~.postgresql.ENUM.create`
         or :meth:`~.postgresql.ENUM.drop`
         are called directly.
         Setting to ``False`` is helpful
         when invoking a creation scheme to a SQL file
         without access to the actual database -
         the :meth:`~.postgresql.ENUM.create` and
         :meth:`~.postgresql.ENUM.drop` methods can
         be used to emit SQL to a target bind.

        """
        native_enum = kw.pop("native_enum", None)
        if native_enum is False:
            util.warn(
                "the native_enum flag does not apply to the "
                "sqlalchemy.dialects.postgresql.ENUM datatype; this type "
                "always refers to ENUM.   Use sqlalchemy.types.Enum for "
                "non-native enum."
            )
        if name is not _NoArg.NO_ARG:
            kw["name"] = name
        super().__init__(*enums, create_type=create_type, **kw)

    def coerce_compared_value(self, op, value):
        super_coerced_type = super().coerce_compared_value(op, value)
        if (
            super_coerced_type._type_affinity
            is type_api.STRINGTYPE._type_affinity
        ):
            return self
        else:
            return super_coerced_type

    @classmethod
    def __test_init__(cls):
        return cls(name="name")

    @classmethod
    def adapt_emulated_to_native(cls, impl, **kw):
        """Produce a PostgreSQL native :class:`_postgresql.ENUM` from plain
        :class:`.Enum`.

        """
        kw.setdefault("validate_strings", impl.validate_strings)
        kw.setdefault("name", impl.name)
        kw.setdefault("schema", impl.schema)
        kw.setdefault("inherit_schema", impl.inherit_schema)
        kw.setdefault("metadata", impl.metadata)
        kw.setdefault("values_callable", impl.values_callable)
        kw.setdefault("omit_aliases", impl._omit_aliases)
        kw.setdefault("_adapted_from", impl)
        if type_api._is_native_for_emulated(impl.__class__):
            kw.setdefault("create_type", impl.create_type)

        return cls(**kw)

    def get_dbapi_type(self, dbapi: ModuleType) -> None:
        """dont return dbapi.STRING for ENUM in PostgreSQL, since that's
        a different type"""

        return None


class DOMAIN(NamedType, type_api.TypeEngine[str]):
    r"""Represent the DOMAIN PostgreSQL type.

    A domain is essentially a data type with optional constraints
    that restrict the allowed set of values. E.g.::

        PositiveInt = DOMAIN("pos_int", Integer, check="VALUE > 0", not_null=True)

        UsPostalCode = DOMAIN(
            "us_postal_code",
            Text,
            check="VALUE ~ '^\d{5}$' OR VALUE ~ '^\d{5}-\d{4}$'",
        )

    See the `PostgreSQL documentation`__ for additional details

    __ https://www.postgresql.org/docs/current/sql-createdomain.html

    .. versionadded:: 2.0

    """  # noqa: E501

    __visit_name__ = "DOMAIN"

    def __init__(
        self,
        name: str,
        data_type: _TypeEngineArgument[Any],
        *,
        collation: Optional[str] = None,
        default: Union[elements.TextClause, str, None] = None,
        constraint_name: Optional[str] = None,
        not_null: Optional[bool] = None,
        check: Union[elements.TextClause, str, None] = None,
        create_type: bool = True,
        **kw: Any,
    ):
        """
        Construct a DOMAIN.

        :param name: the name of the domain
        :param data_type: The underlying data type of the domain.
          This can include array specifiers.
        :param collation: An optional collation for the domain.
          If no collation is specified, the underlying data type's default
          collation is used. The underlying type must be collatable if
          ``collation`` is specified.
        :param default: The DEFAULT clause specifies a default value for
          columns of the domain data type. The default should be a string
          or a :func:`_expression.text` value.
          If no default value is specified, then the default value is
          the null value.
        :param constraint_name: An optional name for a constraint.
          If not specified, the backend generates a name.
        :param not_null: Values of this domain are prevented from being null.
          By default domain are allowed to be null. If not specified
          no nullability clause will be emitted.
        :param check: CHECK clause specify integrity constraint or test
          which values of the domain must satisfy. A constraint must be
          an expression producing a Boolean result that can use the key
          word VALUE to refer to the value being tested.
          Differently from PostgreSQL, only a single check clause is
          currently allowed in SQLAlchemy.
        :param schema: optional schema name
        :param metadata: optional :class:`_schema.MetaData` object which
         this :class:`_postgresql.DOMAIN` will be directly associated
        :param create_type: Defaults to True.
         Indicates that ``CREATE TYPE`` should be emitted, after optionally
         checking for the presence of the type, when the parent table is
         being created; and additionally that ``DROP TYPE`` is called
         when the table is dropped.

        """
        self.data_type = type_api.to_instance(data_type)
        self.default = default
        self.collation = collation
        self.constraint_name = constraint_name
        self.not_null = bool(not_null)
        if check is not None:
            check = coercions.expect(roles.DDLExpressionRole, check)
        self.check = check
        super().__init__(name=name, create_type=create_type, **kw)

    @classmethod
    def __test_init__(cls):
        return cls("name", sqltypes.Integer)

    def adapt(self, cls, **kw):
        kw["check"] = self.check
        kw["not_null"] = self.not_null
        kw["default"] = self.default
        kw["collation"] = self.collation
        return super().adapt(cls, **kw)


class CreateEnumType(ddl._CreateDropBase):
    __visit_name__ = "create_enum_type"


class DropEnumType(ddl._CreateDropBase):
    __visit_name__ = "drop_enum_type"


class CreateDomainType(ddl._CreateDropBase):
    """Represent a CREATE DOMAIN statement."""

    __visit_name__ = "create_domain_type"


class DropDomainType(ddl._CreateDropBase):
    """Represent a DROP DOMAIN statement."""

    __visit_name__ = "drop_domain_type"


def resolve_named_type(
    typ: type_api.TypeEngine[Any], dialect: engine.Dialect
) -> NamedType | None:
    """
    Attempts to associate a `NamedType` with the specified
    type. Associations can be either:
        - The NamedType is an implementation of typ in the current dialect
        - The provided type is an array of a named type
        - The type is a custom type which uses a `NamedType` as an implementation

    If there is no associated type, returns `None`
    """

    if isinstance(typ, type_api.TypeDecorator):
        return resolve_named_type(typ.impl_instance, dialect)

    if isinstance(typ, sqltypes.ARRAY):
        return resolve_named_type(typ.item_type, dialect)

    typ = typ.dialect_impl(dialect)
    return typ if isinstance(typ, NamedType) else None
