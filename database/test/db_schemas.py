import pandera as pa
from pandera import DataFrameSchema, Column

standard_args = {
    "nullable": True,
    "coerce": True,
    "required": True,
}

id_args = {
    "nullable": False,
    "coerce": True,
    "required": True,
    "unique": True,
}

patterns_schema = DataFrameSchema({
    "id": Column(pa.Int, **id_args),
    "path": Column(pa.String, **standard_args),
    "description": Column(pa.String, **standard_args),
}, strict=True, unique=["id"])

laser_settings_schema = DataFrameSchema({
    "id": Column(pa.Int, **id_args),
    "pattern": Column(pa.Int, **standard_args),
    "speed": Column(pa.Float, **standard_args),
    "power": Column(pa.Float, pa.Check.between(0, 100), **standard_args),
    "passes": Column(pa.Int, **standard_args),
    "distance": Column(pa.Float, **standard_args),
    "line_spacing": Column(pa.Float, **standard_args),
}, strict=True, unique=["id"])

batches_schema = DataFrameSchema({
    "id": Column(pa.Int, **id_args),
    "laser_settings": Column(pa.Int, **standard_args),
    "specimen": Column(pa.Int, **standard_args),
    "fixture": Column(pa.Int, **standard_args),
    "Al_batch": Column(pa.Int, **standard_args),
    "polymer_lower": Column(pa.String, **standard_args),
    "polymer_upper": Column(pa.String, **standard_args),
    "Al_mass": Column(pa.Float, **standard_args),
    "production_date": Column(pa.DateTime, **standard_args),
    "notes": Column(pa.String, **standard_args),
}, strict=True, unique=["id"])

tests_schema = DataFrameSchema({
    "id": Column(pa.Int, **id_args),
    "batch": Column(pa.Int, **standard_args),
    "file_path": Column(pa.String, **standard_args),
    "ultimate_tensile_strength": Column(pa.Float, **standard_args),
    "strain_at_break": Column(pa.Float, **standard_args),
    "test_date": Column(pa.DateTime, **standard_args),
    "force": Column(pa.Object, **standard_args),
    "distance": Column(pa.Object, **standard_args),
    "speed": Column(pa.Float, **standard_args),
    "state": Column(pa.String, pa.Check.isin(["alpha", "beta", "gold"]), **standard_args),
    "notes": Column(pa.String, **standard_args),
}, strict=True, unique=["id"])

uCT_scans_schema = DataFrameSchema({
    "id": Column(pa.Int, **id_args),
    "test": Column(pa.Int, **standard_args),
    "file_path": Column(pa.String, **standard_args),
    "voxel_size": Column(pa.Float, **standard_args),
    "SOD": Column(pa.Float, **standard_args),
    "SDD": Column(pa.Float, **standard_args),
    "voltage": Column(pa.Float, **standard_args),
    "current": Column(pa.Float, **standard_args),
    "fps": Column(pa.Int, **standard_args),
    "averaging": Column(pa.Int, **standard_args),
    "notes": Column(pa.String, **standard_args),
}, strict=True, unique=["id"])

raw_Al_batches_schema = DataFrameSchema({
    "id": Column(pa.Int, **id_args),
    "diameter_mean": Column(pa.Float, **standard_args),
    "diameter_std": Column(pa.Float, **standard_args),
    "aspect_ratio": Column(pa.Float, **standard_args),
    "supplier": Column(pa.String, **standard_args),
    "images": Column(pa.String, **standard_args),
    "notes": Column(pa.String, **standard_args),
}, strict=True, unique=["id"])

structured_Al_batches_schema = DataFrameSchema({
    "id": Column(pa.Int, **id_args),
    "raw_batch": Column(pa.Int, **standard_args),
    "process": Column(pa.Int, **standard_args),
    "diameter_mean": Column(pa.Float, **standard_args),
    "diameter_std": Column(pa.Float, **standard_args),
    "start_weight": Column(pa.Float, **standard_args),
    "end_weight": Column(pa.Float, **standard_args),
    "images": Column(pa.String, **standard_args),
    "notes": Column(pa.String, **standard_args),
}, strict=True, unique=["id"])

process_schema = DataFrameSchema({
    "id": Column(pa.Int, **id_args),
    "highTemp": Column(pa.Float, **standard_args),
    "highTemp_time": Column(pa.Float, **standard_args),
    "lowTemp": Column(pa.Float, **standard_args),
    "lowTemp_time": Column(pa.Float, **standard_args),
    "notes": Column(pa.String, **standard_args),
}, strict=True, unique=["id"])

fixture_schema = DataFrameSchema({
    "id": Column(pa.Int, **id_args),
    "file_path": Column(pa.String, **standard_args),
    "description": Column(pa.String, **standard_args),
}, strict=True, unique=["id"])
