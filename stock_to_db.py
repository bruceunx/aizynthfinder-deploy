from sqlalchemy import MetaData, create_engine, Table, Column, String, insert
import pandas as pd

df = pd.read_hdf("./zinc_stock.hdf5", key='table')

url = "sqlite:///test.sql"  # change url to mysql url

engine = create_engine(url)

metaObject = MetaData()

chemicals = Table(
    "chemcials",
    metaObject,
    Column("inchi_key", String(27), nullable=False, index=True, unique=True),
)

metaObject.create_all(engine)

with engine.begin() as conn:
    for inchi_key in df['inchi_key']:
        stmt = insert(chemicals).values(inchi_key=inchi_key)
        conn.execute(stmt)

# sqlite 1.25g
