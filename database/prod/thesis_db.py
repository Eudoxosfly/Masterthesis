import os

import matplotlib.pyplot as plt
import pandas as pd

from db_schemas import *


class ThesisDB:
    """A class to handle a database for the thesis project.

    The database is a collection of tables, each with a specific schema.
    The tables are stored in a .pkl file in the specified directory.
    The database has the following tables:
        - patterns: The patterns used in the tests.
        - laser_settings: The laser settings used in the tests.
        - batches: The batches used in the tests.
        - tests: The tests performed.
        - uCT_scans: The uCT scans performed.
        - raw_Al_batches: The raw Al batches.
        - structured_Al_batches: The structured Al batches.
        - process: The process for the structured Al batches.
        - fixture: The fixture used in the tests.

    Attributes:
        dir (str): The directory where the database file is located.
        file (str): The name of the database file.
        auto_save_on (bool): If True, the database is saved automatically after each change.
        schemas (dict): A dictionary containing the schemas for each table.
    """

    # Schemas
    schemas = {
        "patterns": patterns_schema,
        "laser_settings": laser_settings_schema,
        "batches": batches_schema,
        "tests": tests_schema,
        "uCT_scans": uCT_scans_schema,
        "raw_Al_batches": raw_Al_batches_schema,
        "structured_Al_batches": structured_Al_batches_schema,
        "process": process_schema,
        "fixture": fixture_schema,
    }

    def __init__(self, path="data", create_new=False, auto_save=True):
        """Initialize the ThesisDB instance.

        Args:
            path (str): The directory where the database file is located.
            create_new (bool): If True, the database is created anew, even if a file exists in the directory.
            auto_save (bool): If True, the database is saved automatically after each change.
        """
        self.dir = path
        self.file = "thesis_db"
        self.auto_save_on = auto_save

        self.__startup(create_new)

        not self.auto_save_on and print("Auto save is off. Remember to save the database manually.")

    # Public methods
    def print_table(self, table):
        """Print the table to the console."""
        print("---", table, "---")
        print(self.__dict__[table])

    def get_table(self, table):
        """Returns a copy of the table as a pd.DataFrame.

        The table is a copy, so changes to the returned DataFrame will not affect the database."""
        return self.__dict__[table].copy()

    def save(self):
        """Save the database to a .pkl file."""
        file_path = os.path.join(self.dir, self.file)
        db = {table: self.__dict__[table] for table in self.schemas.keys()}
        pd.to_pickle(db, file_path + ".pkl")
        not self.auto_save_on and print("Database saved to", file_path + ".pkl")

    def add_entries(self, table, entries: dict[str, list]):
        """Add entries to a table.

        The entries must be a dictionary with the column names as keys and lists of values as values.
        The lists must have the same length.
        The id column must be included in the entries.

        Example:
            db.add_entries("patterns", {"id": [1, 2, 3],
            "path": ["path1", "path2", "path3"],
            "description": ["desc1", "desc2", "desc3"]})
        """
        if not isinstance(entries, dict) or not all(isinstance(v, list) for v in entries.values()):
            raise TypeError("Entry must be a dictionary containing lists. Got", type(entries))
        if table not in self.schemas:
            raise KeyError("There is no table named", table)

        schema = self.schemas[table]
        entry = pd.DataFrame(entries)
        new_table = pd.concat([self.__dict__[table], entry], ignore_index=True)
        try:
            self.__dict__[table] = schema.validate(new_table)
        except pa.errors.SchemaError as e:
            print(e)
            print("Entries not added.")
            return
        self.__auto_save()

    def delete_entries(self, table, entry_ids: int | list[int]):
        """Delete entries from a table by their id(s)."""
        if isinstance(entry_ids, int):
            entry_ids = [entry_ids]
        idx_to_drop = self.__dict__[table].id.isin(entry_ids)
        entry_id = self.__dict__[table].loc[idx_to_drop].id.values
        self.__dict__[table] = self.__dict__[table].loc[~idx_to_drop]

        print("Deleted entries with id(s):", entry_id)
        self.__auto_save()

    # Private methods
    def __startup(self, create_new):
        """Initialize the database.

        If the database file exists, load it. Otherwise, create empty tables from the schemas.
        If create_new is True, the database is always created anew, even if the file exists."""
        if self.__is_db_file() and not create_new:
            self.__load()
            print("Database loaded from", "/" + self.dir + "/" + self.file + ".pkl")
        else:
            os.makedirs(self.dir, exist_ok=True)
            self.__create_tables()
            self.__auto_save()
            print("Database created in", "/" + self.dir)

    def __auto_save(self):
        """Automatically save the database to file if auto_save is on."""
        if self.auto_save_on:
            self.save()

    def __is_db_file(self):
        """Check if the database file exists."""
        file_path = os.path.join(self.dir, self.file + ".pkl")
        return os.path.exists(file_path)

    def __load(self):
        """Load the database from a .pkl file.

        The file is loaded into the ThesisDB instance, creating a table for each schema in the ThesisDB.schemas
        dictionary."""
        file_path = os.path.join(self.dir, self.file)
        db = pd.read_pickle(file_path + ".pkl")
        for table in self.schemas.keys():
            self.__dict__[table] = db[table]

    def __create_tables(self):
        """Create empty tables from the schemas.

        Creates a table for each schema in the ThesisDB.schemas dictionary."""
        tables = self.schemas.keys()
        for table in tables:
            self.__dict__[table] = self.__empty_dataframe_from_schema(self.schemas[table])

    # Static methods (utility)

    # private
    @staticmethod
    def __empty_dataframe_from_schema(schema) -> pd.DataFrame:
        """Create an empty pd.DataFrame from a pandera DataFrameSchema."""
        df = (pd.DataFrame(columns=schema.dtypes.keys()).astype(
            {col: str(dtype) for col, dtype in schema.dtypes.items()}
        ))
        return df


    # public
    @staticmethod
    def show_database_structure():
        """Show the database structure as a diagram.

        May not be the current version."""
        fig, axs = plt.subplots(1, 1, figsize=(20, 10))
        axs.imshow(plt.imread("data/db_diagram.png"))
        axs.axis("off")
        plt.title("This diagram may not represent the current version of the database.")
        plt.show()
