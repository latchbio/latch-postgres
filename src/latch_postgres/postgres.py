import asyncio
import functools
import random
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from textwrap import dedent
from typing import Any, Concatenate, ParamSpec, TypeVar, cast

import psycopg.sql as sql
from latch_config.config import PostgresConnectionConfig
from latch_data_validation.data_validation import JsonObject, validate
from latch_o11y.o11y import dict_to_attrs, trace_function
from opentelemetry.sdk.resources import Attributes
from opentelemetry.trace import SpanKind, get_tracer
from psycopg import AsyncConnection, AsyncCursor, IsolationLevel
from psycopg.abc import AdaptContext, Params, Query
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.errors import (
    AdminShutdown,
    CannotConnectNow,
    ConnectionDoesNotExist,
    ConnectionException,
    ConnectionFailure,
    CrashShutdown,
    DatabaseError,
    DeadlockDetected,
    DiskFull,
    DuplicateFile,
    IdleSessionTimeout,
    InsufficientResources,
    IoError,
    OperationalError,
    OperatorIntervention,
    OutOfMemory,
    ProtocolViolation,
    QueryCanceled,
    SerializationFailure,
    SqlclientUnableToEstablishSqlconnection,
    SqlserverRejectedEstablishmentOfSqlconnection,
    SystemError,
    TooManyConnections,
    UndefinedFile,
)
from psycopg.errors import Error as PGError
from psycopg.rows import AsyncRowFactory, Row, dict_row, kwargs_row
from psycopg.types.composite import CompositeInfo, register_composite
from psycopg.types.enum import EnumInfo, register_enum
from psycopg_pool import AsyncConnectionPool
from typing_extensions import Self

from latch_postgres.retries import CABackoff

YT = TypeVar("YT")
ST = TypeVar("ST")
T = TypeVar("T")

tracer = get_tracer(__name__)


# todo(maximsmol): switch all the tracing attributes to otel spec
# del span.type
# in event names postgres -> postgresql (or to psycopg)?
# out.host -> net.peer.name
# out.port -> net.peer.port
# sql.query -> db.statement
# del resource.name
def conninfo_attributes(x: str) -> Attributes:
    data = conninfo_to_dict(x)

    res: Attributes = {
        # todo(maximsmol): is this service name override a datadog idiosyncrasy?
        # todo(maximsmol): allow for multiple databases
        "service.name": "vacuole",
        "db.system": "postgresql",
        "span.type": "sql",
        "db.connection_string": make_conninfo(x, password="[REDACTED]"),
    }
    if data["host"] is not None:
        res["out.host"] = data["host"]
    if data["port"] is not None:
        res["out.port"] = data["port"]
    if data["dbname"] is not None:
        res["db.name"] = data["dbname"]
    if data["user"] is not None:
        res["db.user"] = data["user"]
    if data["application_name"] is not None:
        res["db.application"] = data["application_name"]

    return res


def query_to_string(x: Query, ctx: AdaptContext):
    if isinstance(x, str):
        return x

    if isinstance(x, bytes):
        return x

    return x.as_string(ctx)


# todo(maximsmol): switch tracing to use decorators from o11y
class TracedAsyncCursor(AsyncCursor[Row]):
    trace_attributes: Attributes

    async def execute(
        self,
        query: Query,
        params: Params | None = None,
        *,
        prepare: bool | None = None,
        binary: bool | None = None,
    ) -> Self:
        with tracer.start_as_current_span(
            "postgres.query",
            kind=SpanKind.CLIENT,
            attributes={
                **self.trace_attributes,
                "sql.query": query_to_string(query, self),
                "resource.name": query_to_string(query, self),
            },
        ) as span:
            try:
                return await super().execute(
                    query, params, prepare=prepare, binary=binary
                )
            finally:
                span.set_attributes({"db.rowcount": self.rowcount})

    async def executemany(self, query: Query, params_seq: Iterable[Params]) -> None:
        with tracer.start_as_current_span(
            "postgres.query.many",
            kind=SpanKind.CLIENT,
            attributes={
                **self.trace_attributes,
                "sql.query": query_to_string(query, self),
                "resource.name": query_to_string(query, self),
            },
        ) as span:
            try:
                return await super().executemany(query, params_seq)
            finally:
                span.set_attributes({"db.rowcount": self.rowcount})


class LatchAsyncConnection(AsyncConnection[Row]):
    trace_attributes: Attributes

    def cursor(
        self, row_factory: AsyncRowFactory[Any], *, binary: bool = True
    ) -> AsyncCursor[Any]:
        res = super().cursor(row_factory=row_factory, binary=binary)
        assert isinstance(res, TracedAsyncCursor)
        res.trace_attributes = self.trace_attributes
        return res

    @asynccontextmanager
    async def _query(
        self, model: type[T], query: sql.SQL, **kwargs: Any
    ) -> AsyncGenerator[AsyncCursor[T], None]:
        def model_(**kwargs: JsonObject) -> T:
            return validate(kwargs, model)

        async with self.cursor(kwargs_row(model_)) as curs:
            curs = cast(AsyncCursor[T], curs)
            await curs.execute(query, params=kwargs)

            yield curs

    @asynccontextmanager
    async def _query_no_validate(
        self, query: sql.SQL, **kwargs: Any
    ) -> AsyncGenerator[AsyncCursor[Any], None]:
        async with self.cursor(dict_row) as curs:
            await curs.execute(query, params=kwargs)

            yield curs

    async def queryn(self, model: type[T], query: sql.SQL, **kwargs: Any) -> list[T]:
        async with self._query(model, query, **kwargs) as curs:
            if curs.description is None:
                return []

            return await curs.fetchall()

    async def query1(self, model: type[T], query: sql.SQL, **kwargs: Any) -> T:
        results = await self.queryn(model, query, **kwargs)

        if len(results) == 0:
            raise RuntimeError(f"received no rows: '{len(results)}' < 1")

        if len(results) > 1:
            raise RuntimeError(f"received too many rows: '{len(results)}' > 1")

        return results[0]

    async def query_opt(
        self, model: type[T], query: sql.SQL, **kwargs: Any
    ) -> T | None:
        results = await self.queryn(model, query, **kwargs)

        if len(results) < 1:
            return None

        if len(results) > 1:
            raise RuntimeError(f"received too many rows: '{len(results)}' > 1")

        return results[0]

    async def query_void(self, query: sql.SQL, **kwargs: Any) -> None:
        await self.queryn(type(None), query, **kwargs)

    async def query_unknown(self, query: sql.SQL, **kwargs: Any) -> Any:
        async with self._query_no_validate(query, **kwargs) as curs:
            if curs.description is None:
                return None

            return await curs.fetchall()


@dataclass(frozen=True)
class EnumInfoQueryResponse:
    nspname: str
    name: str
    oid: int
    array_oid: int
    labels: list[str]


class TracedAsyncConnectionPool(AsyncConnectionPool):
    def __init__(
        self,
        conninfo: str = "",
        *,
        open: bool = True,
        connection_class: type[AsyncConnection[Any]] = ...,
        configure: Callable[[AsyncConnection[Any]], Awaitable[None]] | None = None,
        reset: Callable[[AsyncConnection[Any]], Awaitable[None]] | None = None,
        **kwargs: Any,
    ):
        self._real_configure = configure

        self.setup_commands: list[sql.SQL] = []
        self.enum_map: dict[str, type] = {}
        self.composite_type_map: dict[str, Callable[..., Any] | None] = {}

        self._trace_attributes = conninfo_attributes(conninfo)

        super().__init__(
            conninfo,
            open=open,
            connection_class=connection_class,
            configure=self._configure_wrapper,
            reset=reset,
            **kwargs,
        )

    async def _configure_wrapper(self, conn: AsyncConnection[object]):
        with tracer.start_as_current_span("configure connection"):
            assert isinstance(conn, LatchAsyncConnection)

            conn.cursor_factory = TracedAsyncCursor
            conn.trace_attributes = self._trace_attributes

            if (
                len(self.setup_commands) > 0
                or len(self.enum_map) > 0
                or len(self.composite_type_map) > 0
            ):
                old_read_only = conn.read_only
                old_autocommit = conn.autocommit
                await asyncio.gather(
                    conn.set_read_only(False), conn.set_autocommit(True)
                )

                async def run_setup(cmd: sql.SQL):
                    with tracer.start_as_current_span("setup command"):
                        await conn.query_opt(dict, cmd)

                async def run_composite_setup():
                    with tracer.start_as_current_span("composite type setup"):
                        with tracer.start_as_current_span("register composite types"):
                            for db_type, f in self.composite_type_map.items():
                                type_info = await CompositeInfo.fetch(conn, db_type)
                                if type_info is None:
                                    raise RuntimeError(
                                        f"failed to fetch composite info for {db_type}"
                                    )
                                register_composite(type_info, conn, f)

                async def run_enum_setup():
                    with tracer.start_as_current_span("enum setup"):
                        with tracer.start_as_current_span("fetch type info"):
                            # query from
                            # https://github.com/psycopg/psycopg/blob/fd1659118e96a48f22b4e67ff17c2cdab8bd0e84/psycopg/psycopg/_typeinfo.py#L148
                            raw_responses = await conn.queryn(
                                EnumInfoQueryResponse,
                                sqlq(
                                    """
                                select
                                    nspname,
                                    name,
                                    oid,
                                    array_oid,
                                    array_agg(label)
                                        as labels
                                from
                                    (
                                        select
                                            n.nspname,
                                            t.typname
                                                as name,
                                            t.oid
                                                as oid,
                                            t.typarray
                                                as array_oid,
                                            e.enumlabel
                                                as label
                                        from
                                            pg_type t
                                        inner join
                                            pg_enum e
                                            on e.enumtypid = t.oid
                                        inner join
                                            pg_namespace n
                                            on n.oid = t.typnamespace
                                        inner join
                                            unnest(%(enum_names)s::text[]) en
                                            on to_regtype(en) = t.oid
                                        order by
                                            e.enumsortorder
                                    ) x
                                group by
                                    nspname,
                                    name,
                                    oid,
                                    array_oid
                                """
                                ),
                                enum_names=list(self.enum_map.keys()),
                            )

                            info_responses = {
                                f"{r.nspname}.{r.name}": r for r in raw_responses
                            }

                        for dbname, native in self.enum_map.items():
                            with tracer.start_as_current_span(
                                "register enum",
                                attributes={
                                    "dbname": dbname,
                                    "native": native.__qualname__,
                                },
                            ):
                                r = info_responses[dbname]
                                enum_info = EnumInfo(
                                    r.name, r.oid, r.array_oid, r.labels
                                )
                                register_enum(enum_info, conn, native)

                setup_tasks = [
                    asyncio.create_task(run_setup(cmd)) for cmd in self.setup_commands
                ]
                if len(self.enum_map) > 0:
                    setup_tasks.append(asyncio.create_task(run_enum_setup()))

                if len(self.composite_type_map) > 0:
                    setup_tasks.append(asyncio.create_task(run_composite_setup()))

                await asyncio.gather(*setup_tasks)

                await asyncio.gather(
                    conn.set_read_only(old_read_only),
                    conn.set_autocommit(old_autocommit),
                )

            if self._real_configure is not None:
                await self._real_configure(conn)

    async def getconn(self, timeout: float | None = None) -> AsyncConnection[object]:
        with tracer.start_as_current_span(
            "postgres.connect", kind=SpanKind.CLIENT, attributes=self._trace_attributes
        ):
            return await super().getconn(timeout)

    # todo(maximsmol): somehow track progress per-connection
    async def open(self, wait: bool = False, timeout: float = 30) -> None:
        with tracer.start_as_current_span(
            "open db pool", attributes=conninfo_attributes(self.conninfo)
        ):
            return await super().open(wait, timeout)

    async def close(self, timeout: float = 5) -> None:
        with tracer.start_as_current_span(
            "close db pool", attributes=conninfo_attributes(self.conninfo)
        ):
            return await super().close(timeout)


P = ParamSpec("P")


def mixin_dict(a: dict[str, object], b: dict[str, object]):
    for k, v in b.items():
        if k in a:
            a_val = a[k]
            if isinstance(a_val, dict) and isinstance(v, dict):
                mixin_dict(a_val, v)
                continue

        a[k] = v


def pg_error_to_dict(x: PGError, *, short: bool = False):
    diagnostic_obj = {
        "severity": x.diag.severity,
        "message": {"detail": x.diag.message_detail},
    }

    if not short:
        mixin_dict(
            diagnostic_obj,
            {
                "message": {
                    # These are most likely the same for each sqlstate
                    "primary": x.diag.message_primary,
                    "hint": x.diag.message_hint,
                },
                "context": x.diag.context,
                # todo(maximsmol): not sure which of these are actually useful
                "statement_position": x.diag.statement_position,
                "internal": {
                    "position": x.diag.internal_position,
                    "query": x.diag.internal_query,
                },
                "schema_name": x.diag.schema_name,
                "table_name": x.diag.table_name,
                "column_name": x.diag.column_name,
                "datatype-name": x.diag.datatype_name,
                "constraint_name": x.diag.constraint_name,
                # this is C library code so basically useless
                # "source": {
                #     "file": x.diag.source_file,
                #     "line": x.diag.source_line,
                #     "function": x.diag.source_function,
                # },
            },
        )

    return {
        "sqlstate": x.sqlstate,
        "diagnostic": diagnostic_obj,
        "type": type(x).__name__,
    }


def with_conn_retry(
    f: Callable[Concatenate[LatchAsyncConnection[Any], P], Coroutine[YT, ST, T]],
    pool: AsyncConnectionPool,
    db_config: PostgresConnectionConfig,
) -> Callable[P, Coroutine[YT, ST, T]]:
    @functools.wraps(f)
    async def inner(*args: P.args, **kwargs: P.kwargs):
        with tracer.start_as_current_span("database session") as s:
            try:
                retries = 0
                accum_retry_time = 0

                while True:
                    try:
                        async with pool.connection() as conn:
                            with tracer.start_as_current_span("connection acquired"):
                                assert isinstance(conn, LatchAsyncConnection)

                                backoff = CABackoff(
                                    db_config.tx_retries.delay_quant,
                                    db_config.tx_retries.max_wait_time,
                                )
                                while True:
                                    try:
                                        res = await f(conn, *args, **kwargs)
                                        # Commit here so we can retry if it fails
                                        # Otherwise the context manager will commit and fail
                                        await conn.commit()
                                        return res
                                    except (
                                        # Class 40 - Transaction Rollback
                                        SerializationFailure,
                                        DeadlockDetected,
                                    ) as e:
                                        # Retry with the same connection
                                        delay = backoff.retry()

                                        if delay is None:
                                            raise e

                                        s.add_event(
                                            "transaction retry",
                                            {
                                                "db.retry.count": retries,
                                                "db.retry.accum_retry_time": str(
                                                    backoff.acc_wait_time
                                                ),
                                                "db.retry.delay": str(delay),
                                            }
                                            | dict_to_attrs(
                                                pg_error_to_dict(e, short=True),
                                                "db.retry.reason",
                                            ),
                                        )

                                        await conn.rollback()
                                        await asyncio.sleep(delay)
                    except (SerializationFailure, DeadlockDetected):
                        # todo(maximsmol): should be unnecessary if the list below is precise enough
                        raise
                    except (
                        # Class 08 - Connection Exception
                        ConnectionException,
                        ConnectionDoesNotExist,
                        ConnectionFailure,
                        SqlclientUnableToEstablishSqlconnection,
                        SqlserverRejectedEstablishmentOfSqlconnection,
                        ProtocolViolation,
                        # Class 53 - Insufficient Resources
                        InsufficientResources,
                        DiskFull,
                        OutOfMemory,
                        TooManyConnections,
                        # Class 57 - Operator Intervention
                        OperatorIntervention,
                        QueryCanceled,
                        AdminShutdown,
                        CrashShutdown,
                        CannotConnectNow,
                        IdleSessionTimeout,
                        # Class 58 - System Error (errors external to PostgreSQL itself)
                        SystemError,
                        IoError,
                        UndefinedFile,
                        DuplicateFile,
                        # todo(maximsmol): narrow this down. had to include it for connection loss recovery because subclasses probably do not work
                        # I've improved the error logging so we should know exactly what to expect in practice now
                        OperationalError,
                    ) as e:
                        # Connection is dead. Get a new one and retry

                        # todo(maximsmol): add metrics
                        retries += 1

                        if accum_retry_time > db_config.conn_retries.max_wait_time:
                            raise

                        if retries == 1:
                            delay = 0
                        else:
                            delay = random.uniform(
                                db_config.conn_retries.min_retry_time,
                                db_config.conn_retries.max_retry_time,
                            )

                        s.add_event(
                            "connection retry",
                            {
                                "db.retry.count": retries,
                                "db.retry.accum_retry_time": str(accum_retry_time),
                                "db.retry.delay": str(delay),
                            }
                            | dict_to_attrs(
                                pg_error_to_dict(e, short=True), "db.retry.reason"
                            ),
                        )

                        if delay != 0:
                            await asyncio.sleep(delay)
                            accum_retry_time += delay
            except DatabaseError as e:
                s.set_attributes(dict_to_attrs(pg_error_to_dict(e), "db.error"))

                raise e

    return inner


def get_with_conn_retry(
    pool: AsyncConnectionPool, db_config: PostgresConnectionConfig
) -> Callable[
    [Callable[Concatenate[LatchAsyncConnection[Any], P], Coroutine[YT, ST, T]]],
    Callable[P, Coroutine[YT, ST, T]],
]:
    return functools.partial(with_conn_retry, pool=pool, db_config=db_config)


def sqlq(x: str):
    return sql.SQL(dedent(x))


# todo(maximsmol): conn resets appear in the startup span and make it last forever
@trace_function(tracer)
async def reset_conn(
    x: AsyncConnection[object],
    read_only: bool = False,
    isolation_level: IsolationLevel = IsolationLevel.SERIALIZABLE,
):
    x.prepare_threshold = 0

    if x.read_only != read_only:
        await x.set_read_only(read_only)

    if x.isolation_level != isolation_level:
        await x.set_isolation_level(isolation_level)


# fixme(maximsmol): use autocommit transactions
def get_pool(
    config: PostgresConnectionConfig,
    *,
    application_name: str,
    read_only: bool = True,
    isolation_level: IsolationLevel = IsolationLevel.SERIALIZABLE,
    configure: Callable[[AsyncConnection[object]], Awaitable[None]] | None = None,
    reset: Callable[[AsyncConnection[object]], Awaitable[None]] | None = None,
) -> TracedAsyncConnectionPool:
    conn_str = make_conninfo(
        host=config.host,
        port=config.port,
        dbname=config.dbname,
        user=config.user,
        password=config.password,
        application_name=application_name,
    )

    async def configure_impl(x: AsyncConnection[object]) -> None:
        await reset_conn(x, read_only=read_only, isolation_level=isolation_level)
        if configure is not None:
            await configure(x)

    async def reset_impl(x: AsyncConnection[object]) -> None:
        await reset_conn(x, read_only=read_only, isolation_level=isolation_level)
        if reset is not None:
            await reset(x)

    return TracedAsyncConnectionPool(
        conn_str,
        min_size=1,
        max_size=config.pool_size,
        timeout=timedelta(seconds=5) / timedelta(seconds=1),
        open=False,
        configure=configure_impl,
        reset=reset_impl,
        connection_class=LatchAsyncConnection,
    )
