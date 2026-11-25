import sqlite3
import pandas as pd
import glob


if __name__ == "__main__":
    night_max = 730

    db_files = glob.glob("*10yrs.db")

    for filename in db_files:

        con = sqlite3.connect(filename)
        df = pd.read_sql("select * from observations where night < %i;" % night_max, con)
        con.close()

        outfile = filename.replace("10yrs", "2yrs")
        con = sqlite3.connect(outfile)
        df.to_sql(observations, con)
        con.close()

        