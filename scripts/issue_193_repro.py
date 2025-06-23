from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
import argparse
import sys
from pgvector.sqlalchemy import Vector
import sqlalchemy as sa
import numpy as np
import asyncio

metadata = sa.MetaData()

Embeddings = sa.Table(
    "embeddings",
    metadata,
    sa.Column("id", sa.BigInteger, primary_key=True),
    sa.Column("embedding", Vector(1024)),
)

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=100)
parser.add_argument("--batches", type=int, default=1000)
parser.add_argument("--parallelism", type=int, default=1)

async def _main(args):
    # Adapted connection string for local PGRX development
    db_url = "postgresql+asyncpg://tjg@localhost:28817/postgres"

    engine = create_async_engine(
        db_url,
        connect_args={"server_settings": {"application_name": "corruption-test"}},
    )
    await _init_db(engine)

    if args.parallelism == 1:
        for _ in range(args.batches):
            await _insert_embeddings(engine, args.batch_size)
        return

    todo = args.batches

    async def worker():
        nonlocal todo
        while todo >= 0:
            try:
                await _insert_embeddings(engine, args.batch_size)
            except Exception as e:
                print(e)
                print(f"error after {args.batches - todo} inserts")
                sys.exit(1)
            todo -= 1

    await asyncio.gather(*(worker() for _ in range(args.parallelism)))

async def _init_db(engine) -> None:
    tbl = """CREATE TABLE IF NOT EXISTS embeddings (
 id BIGSERIAL PRIMARY KEY,
 embedding vector(1024)
 )
 """

    idx = """CREATE INDEX IF NOT EXISTS embeddings_embedding_diskann
 ON embeddings USING diskann (embedding vector_cosine_ops)
 """
    session = AsyncSession(engine)
    async with session.begin():
        await session.execute(sa.text(tbl))
        await session.execute(sa.text(idx))

async def _insert_embeddings(engine, n: int):
    session = AsyncSession(engine)
    async with session.begin():
        await session.execute(
            Embeddings.insert(), [{"embedding": e} for e in _random_embeddings(n)]
        )

def _random_embeddings(n: int):
    return [np.random.rand(1024) for _ in range(n)]

if __name__ == "__main__":
    args = parser.parse_args()
    asyncio.run(_main(args))