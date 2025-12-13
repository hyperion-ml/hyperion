#!/usr/bin/env python
"""
Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ActionYesNo,
    ArgumentParser,
    namespace_to_dict,
)

from hyperion.hyp_defs import config_logger
from hyperion.utils import (
    ClassInfo,
    DiarizationSet,
    EnrollmentMap,
    FeatureSet,
    InfoTable,
    PathLike,
    RecordingSet,
    SegmentSet,
    VADSet,
)

subcommand_list = [
    "cat",
    "filter",
    "filter_by_predicate",
    "make_class_file_from_column",
    "drop_columns",
    "add_columns",
    "replace_columns",
    "average_results",
    "harmonize_columns_by_majority_vote",
    "harmonize_columns_by_average",
    "histogram",
    "scatter2d",
    "scatter3d",
]
table_dict = {
    "segments": SegmentSet,
    "recordings": RecordingSet,
    "features": FeatureSet,
    "vads": VADSet,
    "diarizations": DiarizationSet,
    "classes": ClassInfo,
    "enrollments": EnrollmentMap,
    "generic": InfoTable,
}


def add_common_args(parser):
    """Add shared table type and verbosity options to an argument parser."""
    parser.add_argument(
        "--table-type",
        default="generic",
        choices=list(table_dict.keys()),
        help=f"Type of table in {list(table_dict.keys())}",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
        help="verbosity level for logging output",
    )


def make_cat_parser():
    """Build parser for concatenating multiple tables into one."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="configuration file in YAML format"
    )
    parser.add_argument(
        "--input-files", default=None, nargs="+", help="optional list of input files"
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="""output file, if input-files is None, input files names are derived from it""",
    )
    parser.add_argument(
        "--num-tables",
        default=0,
        type=int,
        help="""number of jobs we used to create the individual tables""",
    )
    parser.add_argument(
        "--base-idx",
        default=1,
        type=int,
        help="""index of the first job, typically 0 or 1""",
    )
    parser.add_argument(
        "--skip-missing",
        default=False,
        action=ActionYesNo,
        help="skip missing input files instead of raising an error",
    )

    add_common_args(parser)
    return parser


def cat(
    table_type: str,
    input_files: Union[List[PathLike], None],
    output_file: PathLike,
    num_tables: int,
    base_idx: int = 1,
    skip_missing: bool = False,
):
    """Concatenate a list of tables into a single file."""
    assert input_files is not None or num_tables != 0
    output_file = Path(output_file)
    if input_files is None:
        ext = output_file.suffix
        input_file_base = output_file.with_suffix("")
        input_files = []
        for i in range(num_tables):
            idx = base_idx + i
            input_file_i = input_file_base.with_suffix(f".{idx}{ext}")
            input_files.append(input_file_i)

    logging.info(f"Concatenating {len(input_files)} files into {output_file}")
    table_class = table_dict[table_type]
    tables = []
    for file_path in input_files:
        file_path = Path(file_path)
        if not file_path.is_file():
            if skip_missing:
                logging.warning(f"Skipping missing file {file_path}")
                continue
            raise FileNotFoundError(f"Input file not found: {file_path}")
        tables.append(table_class.load(file_path))

    if not tables:
        raise ValueError("No input tables found to concatenate")

    output_table = table_class.cat(tables)
    output_table.save(output_file)


def make_filter_parser():
    """Build parser for filtering a table using another table as filter."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="configuration file in YAML format"
    )
    parser.add_argument("--input-file", required=True, help="input table file")
    parser.add_argument(
        "--filter-file", required=True, help="table file that we use as filter"
    )
    parser.add_argument(
        "--filter-by", default="id", help="column that we use to filter "
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="""output table file""",
    )
    parser.add_argument(
        "--raise-if-missing",
        default=True,
        action=ActionYesNo,
        help="raise exception if filter values are not in input file",
    )
    parser.add_argument(
        "--keep",
        default=True,
        action=ActionYesNo,
        help="whether to keep or remove the filtered items",
    )
    add_common_args(parser)
    return parser


def filter(
    table_type: str,
    input_file: PathLike,
    filter_file: PathLike,
    output_file: PathLike,
    filter_by: str,
    keep: bool,
    raise_if_missing: bool,
):
    """Filter rows by matching values from a second table."""
    logging.info(
        f"Filtering {input_file} by {filter_file} on {filter_by} into {output_file}"
    )

    input_file = Path(input_file)
    filter_file = Path(filter_file)
    output_file = Path(output_file)

    table_class = table_dict[table_type]
    input_table = table_class.load(input_file)
    filter_table = table_class.load(filter_file)
    output_table = input_table.filter(
        items=filter_table[filter_by],
        by=filter_by,
        keep=keep,
        raise_if_missing=raise_if_missing,
    )
    output_table.save(output_file)


def make_filter_by_predicate_parser():
    """Build parser for filtering a table using a predicate expression."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="configuration file in YAML format"
    )
    parser.add_argument("--input-file", required=True, help="input table file")
    parser.add_argument(
        "--predicate", required=True, help="predicate to use for filtering"
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="""output table file""",
    )
    parser.add_argument(
        "--keep",
        default=True,
        action=ActionYesNo,
        help="whether to keep or remove the filtered items",
    )
    add_common_args(parser)
    return parser


def filter_by_predicate(
    table_type: str,
    input_file: PathLike,
    predicate: str,
    keep: bool,
    output_file: PathLike,
):
    """Filter rows using a boolean predicate on the table columns."""
    logging.info(f"Filtering {input_file} by {predicate} into {output_file}")

    input_file = Path(input_file)
    output_file = Path(output_file)

    table_class = table_dict[table_type]
    input_table = table_class.load(input_file)
    output_table = input_table.filter(
        predicate=predicate,
        keep=keep,
    )
    output_table.save(output_file)


def make_make_class_file_from_column_parser():
    """Build parser for creating a class info table from a column."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="configuration file in YAML format"
    )
    parser.add_argument("--input-file", required=True, help="input table file")

    parser.add_argument(
        "--column",
        required=True,
        help="column that we want to use to create a class-file",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="""output class-file table""",
    )

    add_common_args(parser)
    return parser


def make_class_file_from_column(
    table_type: str,
    input_file: PathLike,
    output_file: PathLike,
    column: str,
):
    """Generate a class-info table from unique values of a column."""
    logging.info(
        f"Creating class file from {input_file} column {column} into {output_file}"
    )
    input_file = Path(input_file)
    output_file = Path(output_file)

    table_class = table_dict[table_type]
    input_table = table_class.load(input_file)
    class_ids = np.unique(input_table[column])
    df = pd.DataFrame({"id": class_ids})
    output_table = ClassInfo(df)
    output_table.save(output_file)


def make_drop_columns_parser():
    """Build parser for dropping or keeping columns from a table."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="configuration file in YAML format"
    )
    parser.add_argument("--input-file", required=True, help="input table file")
    parser.add_argument(
        "--columns", required=True, nargs="+", help="columns to keep or drop"
    )
    parser.add_argument(
        "--keep",
        default=False,
        action=ActionYesNo,
        help="whether to keep or drop columns",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="""output table file, If None, it overwrites input""",
    )

    add_common_args(parser)
    return parser


def drop_columns(
    table_type: str,
    input_file: PathLike,
    columns: List[str],
    output_file: Optional[PathLike] = None,
    keep: bool = False,
):
    """Drop or keep selected columns, backing up the input if overwriting."""

    input_file = Path(input_file)
    if output_file is None:
        bk_file = input_file.with_suffix(input_file.suffix + ".bk")
        if not bk_file.is_file():
            import shutil

            shutil.copy2(input_file, bk_file)
        output_file = input_file

    logging.info(f"Dropping columns {columns} from {input_file} into {output_file}")
    table_class = table_dict[table_type]
    input_table = table_class.load(input_file)
    output_table = input_table.filter(columns=columns, keep=keep)
    output_table.save(output_file)


def make_add_columns_parser():
    """Build parser for merging columns from a secondary table."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="configuration file in YAML format"
    )
    parser.add_argument("--input-file", required=True, help="input table file")
    parser.add_argument("--right-table-file", required=True, help="table file to add")
    parser.add_argument("--columns", default=None, nargs="+", help="columns to add")

    parser.add_argument("--on", default="id", help="column to join on")
    parser.add_argument(
        "--right-on", default=None, help="column to join on in right table"
    )
    parser.add_argument(
        "--replace-overlapping",
        default=False,
        action=ActionYesNo,
        help="replace overlapping columns if True",
    )
    parser.add_argument(
        "--ignore-overlapping",
        default=False,
        action=ActionYesNo,
        help="ignore overlapping columns if True",
    )
    parser.add_argument(
        "--remove-missing",
        default=False,
        action=ActionYesNo,
        help="remove rows with missing values in right table",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="""output table file, If None, it overwrites input""",
    )

    add_common_args(parser)
    return parser


def add_columns(
    table_type: str,
    input_file: PathLike,
    right_table_file: PathLike,
    columns: List[str],
    on: str = "id",
    right_on: Optional[str] = None,
    replace_overlapping: bool = False,
    ignore_overlapping: bool = False,
    remove_missing: bool = False,
    output_file: Optional[PathLike] = None,
):
    """Join columns from another table into the input table."""

    input_file = Path(input_file)
    if output_file is None:
        bk_file = input_file.with_suffix(input_file.suffix + ".bk")
        if not bk_file.is_file():
            import shutil

            shutil.copy2(input_file, bk_file)
        output_file = input_file

    logging.info(
        f"Adding columns from {right_table_file} to {input_file} on {on} into {output_file}"
    )
    table_class = table_dict[table_type]
    input_table = table_class.load(input_file)
    right_table = table_class.load(right_table_file)
    input_table.add_columns(
        right_table,
        column_names=columns,
        on=on,
        right_on=right_on,
        replace_overlapping=replace_overlapping,
        ignore_overlapping=ignore_overlapping,
        remove_missing=remove_missing,
    )
    input_table.save(output_file)


def make_replace_columns_parser():
    """Build parser for replacing columns in a table with another table."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="configuration file in YAML format"
    )
    parser.add_argument("--input-file", required=True, help="input table file")
    parser.add_argument(
        "--replacement-file",
        required=True,
        help="table whose rows we are going to copy to the original table",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="""output table file, If None, it overwrites input""",
    )
    parser.add_argument(
        "--columns",
        default=None,
        nargs="+",
        help="columns to replace, if None, all are replaced",
    )

    add_common_args(parser)
    return parser


def replace_columns(
    table_type: str,
    input_file: PathLike,
    replacement_file: PathLike,
    output_file: Optional[PathLike] = None,
    columns: Optional[List[str]] = None,
):
    """Replace columns in a table with values from another table."""
    logging.info(
        f"Replacing columns in {input_file} with {replacement_file} into {output_file}"
    )
    input_file = Path(input_file)
    if output_file is None:
        bk_file = input_file.with_suffix(input_file.suffix + ".bk")
        if not bk_file.is_file():
            import shutil

            shutil.copy2(input_file, bk_file)
        output_file = input_file

    table_class = table_dict[table_type]
    input_table = table_class.load(input_file)
    replacement_table = table_class.load(replacement_file)
    input_table.replace_columns(replacement_table, column_names=columns)
    input_table.save(output_file)


def make_average_results_parser():
    """Build parser for averaging columns across several result tables."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="configuration file in YAML format"
    )
    parser.add_argument(
        "--input-files", default=None, nargs="+", help="optional list of input files"
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="""output file, if input-files is None, input files names are derived from it""",
    )
    parser.add_argument(
        "--num-tables",
        default=0,
        type=int,
        help="""number of jobs we used to create the individual tables""",
    )
    parser.add_argument(
        "--base-idx",
        default=1,
        type=int,
        help="""index of the first job, typically 0 or 1""",
    )

    add_common_args(parser)
    return parser


def average_results(
    table_type: str,
    input_files: Union[List[PathLike], None],
    output_file: PathLike,
    num_tables: int,
    base_idx: int = 1,
):
    """Average numeric result columns across multiple result tables."""
    assert input_files is not None or num_tables != 0
    output_file = Path(output_file)
    if input_files is None:
        ext = output_file.suffix
        input_file_base = output_file.with_suffix("")
        input_files = []
        for i in range(num_tables):
            idx = base_idx + i
            input_file_i = input_file_base.with_suffix(f".{idx}{ext}")
            input_files.append(input_file_i)

    logging.info(f"Averaging {len(input_files)} files into {output_file}")
    output_table = None
    index = None
    columns = None
    for file_path in input_files:
        file_path = Path(file_path)
        if file_path.suffix == ".tsv":
            sep = "\t"
        else:
            sep = ","
        df = pd.read_csv(file_path, sep=sep)
        if index is None:
            if "scores" in df and "key" in df:
                index = ["scores", "key"]
                columns = df.columns[2:]
            else:
                raise ValueError("don't know what to use as index for table")

        df.set_index(index, inplace=True)
        for column in columns:
            if df[column].dtype == "object":
                df[column] = pd.to_numeric(df[column])

        if output_table is None:
            output_table = df
        else:
            output_table = pd.merge(
                left=output_table,
                right=df,
                how="inner",
                left_index=True,
                right_index=True,
                suffixes=(None, "_y"),
            )
            for column in columns:
                output_table[column] += output_table[f"{column}_y"]
                output_table.drop(columns=[f"{column}_y"], inplace=True)

    for column in columns:
        output_table[column] /= len(input_files)

    if output_file.suffix == ".tsv":
        sep = "\t"
    else:
        sep = ","

    output_dir = output_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_table.reset_index(inplace=True)
    output_table.to_csv(output_file, sep=sep, index=False, float_format="{:.4f}".format)


def make_harmonize_columns_by_majority_vote_parser():
    """Build parser for harmonizing categorical columns via majority vote."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="configuration file in YAML format"
    )
    parser.add_argument("--input-file", required=True, help="input table file")
    parser.add_argument(
        "--voter-columns",
        required=True,
        nargs="+",
        help="columns that define the group for voting",
    )
    parser.add_argument(
        "--target-columns",
        required=True,
        nargs="+",
        help="columns to harmonize using majority vote",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="""output table file, If None, it overwrites input""",
    )

    add_common_args(parser)
    return parser


def harmonize_columns_by_majority_vote(
    table_type: str,
    input_file: PathLike,
    voter_columns: List[str],
    target_columns: List[str],
    output_file: Optional[PathLike] = None,
):
    """Harmonize categorical columns so each group agrees via majority vote."""
    input_file = Path(input_file)
    if output_file is None:
        bk_file = input_file.with_suffix(input_file.suffix + ".bk")
        if not bk_file.is_file():
            import shutil

            shutil.copy2(input_file, bk_file)
        output_file = input_file

    logging.info(
        "Harmonizing columns %s by majority vote grouped by %s from %s into %s",
        target_columns,
        voter_columns,
        input_file,
        output_file,
    )
    table_class = table_dict[table_type]
    table = table_class.load(input_file)
    table.harmonize_columns_by_majority_vote(
        voter_columns=voter_columns, target_columns=target_columns
    )
    table.save(output_file)


def make_harmonize_columns_by_average_parser():
    """Build parser for harmonizing numeric columns via averaging."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="configuration file in YAML format"
    )
    parser.add_argument("--input-file", required=True, help="input table file")
    parser.add_argument(
        "--voter-columns",
        required=True,
        nargs="+",
        help="columns that define the group for averaging",
    )
    parser.add_argument(
        "--target-columns",
        required=True,
        nargs="+",
        help="numeric columns to harmonize via mean",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="""output table file, If None, it overwrites input""",
    )

    add_common_args(parser)
    return parser


def harmonize_columns_by_average(
    table_type: str,
    input_file: PathLike,
    voter_columns: List[str],
    target_columns: List[str],
    output_file: Optional[PathLike] = None,
):
    """Harmonize numeric columns so each group shares the same mean value."""
    input_file = Path(input_file)
    if output_file is None:
        bk_file = input_file.with_suffix(input_file.suffix + ".bk")
        if not bk_file.is_file():
            import shutil

            shutil.copy2(input_file, bk_file)
        output_file = input_file

    logging.info(
        "Harmonizing columns %s by averaging grouped by %s from %s into %s",
        target_columns,
        voter_columns,
        input_file,
        output_file,
    )
    table_class = table_dict[table_type]
    table = table_class.load(input_file)
    table.harmonize_columns_by_average(
        voter_columns=voter_columns, target_columns=target_columns
    )
    table.save(output_file)


def make_histogram_parser():
    """Build parser for plotting a histogram of a column."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="configuration file in YAML format"
    )
    parser.add_argument("--input-file", required=True, help="input table file")
    parser.add_argument("--column", required=True, help="column to plot")
    parser.add_argument(
        "--bins",
        default=None,
        type=int,
        help="number of bins for numeric columns (ignored for strings)",
    )
    parser.add_argument(
        "--density",
        default=True,
        action=ActionYesNo,
        help="plot density/relative frequency instead of counts",
    )
    parser.add_argument(
        "--kind",
        default="bar",
        choices=["bar", "line"],
        help="histogram style",
    )
    parser.add_argument(
        "--color",
        default="C0",
        choices=[f"C{i}" for i in range(10)] + ["r", "g", "b", "c", "m", "y", "k"],
        help="matplotlib color for the histogram",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="output image file; if not set, the plot is shown",
    )
    parser.add_argument(
        "--dropna",
        default=True,
        action=ActionYesNo,
        help="drop NA values before plotting",
    )

    add_common_args(parser)
    return parser


def histogram(
    table_type: str,
    input_file: PathLike,
    column: str,
    bins: Optional[int],
    density: bool,
    kind: str,
    color: str,
    output_file: Optional[PathLike],
    dropna: bool,
):
    """Plot or save a histogram for a column in the table."""
    input_file = Path(input_file)
    logging.info(f"Plotting histogram for column {column} from {input_file}")

    table_class = table_dict[table_type]
    table = table_class.load(input_file)
    table.histogram(
        column=column,
        bins=bins,
        density=density,
        kind=kind,
        color=color,
        output_file=output_file,
        dropna=dropna,
    )


def make_scatter2d_parser():
    """Build parser for plotting a 2D scattergram."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="configuration file in YAML format"
    )
    parser.add_argument("--input-file", required=True, help="input table file")
    parser.add_argument("--x-column", required=True, help="column for x-axis")
    parser.add_argument("--y-column", required=True, help="column for y-axis")
    parser.add_argument(
        "--color",
        default="C0",
        choices=[f"C{i}" for i in range(10)] + ["r", "g", "b", "c", "m", "y", "k"],
        help="matplotlib color for points",
    )
    parser.add_argument(
        "--marker",
        default="o",
        choices=["o", "s", "^", "v", "x", "+", "*", "d", "."],
        help="matplotlib marker style for points",
    )
    parser.add_argument(
        "--sample-frac",
        default=1.0,
        type=float,
        help="fraction of points to sample randomly (0,1]",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="output image file; if not set, the plot is shown",
    )
    parser.add_argument(
        "--dropna",
        default=True,
        action=ActionYesNo,
        help="drop NA values before plotting",
    )

    add_common_args(parser)
    return parser


def scatter2d(
    table_type: str,
    input_file: PathLike,
    x_column: str,
    y_column: str,
    color: str,
    marker: str,
    sample_frac: float,
    output_file: Optional[PathLike],
    dropna: bool,
):
    """Plot or save a 2D scattergram for two columns."""
    input_file = Path(input_file)
    logging.info(f"Plotting 2D scatter for {x_column} vs {y_column} from {input_file}")

    table_class = table_dict[table_type]
    table = table_class.load(input_file)
    table.scatter2d(
        x_column=x_column,
        y_column=y_column,
        color=color,
        marker=marker,
        sample_frac=sample_frac,
        output_file=output_file,
        dropna=dropna,
    )


def make_scatter3d_parser():
    """Build parser for plotting a 3D scattergram."""
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="configuration file in YAML format"
    )
    parser.add_argument("--input-file", required=True, help="input table file")
    parser.add_argument("--x-column", required=True, help="column for x-axis")
    parser.add_argument("--y-column", required=True, help="column for y-axis")
    parser.add_argument("--z-column", required=True, help="column for z-axis")
    parser.add_argument(
        "--color",
        default="C0",
        choices=[f"C{i}" for i in range(10)] + ["r", "g", "b", "c", "m", "y", "k"],
        help="matplotlib color for points",
    )
    parser.add_argument(
        "--marker",
        default="o",
        choices=["o", "s", "^", "v", "x", "+", "*", "d", "."],
        help="matplotlib marker style for points",
    )
    parser.add_argument(
        "--sample-frac",
        default=1.0,
        type=float,
        help="fraction of points to sample randomly (0,1]",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="output image file; if not set, the plot is shown",
    )
    parser.add_argument(
        "--dropna",
        default=True,
        action=ActionYesNo,
        help="drop NA values before plotting",
    )

    add_common_args(parser)
    return parser


def scatter3d(
    table_type: str,
    input_file: PathLike,
    x_column: str,
    y_column: str,
    z_column: str,
    color: str,
    marker: str,
    sample_frac: float,
    output_file: Optional[PathLike],
    dropna: bool,
):
    """Plot or save a 3D scattergram for three columns."""
    input_file = Path(input_file)
    logging.info(
        f"Plotting 3D scatter for {x_column}, {y_column}, {z_column} from {input_file}"
    )

    table_class = table_dict[table_type]
    table = table_class.load(input_file)
    table.scatter3d(
        x_column=x_column,
        y_column=y_column,
        z_column=z_column,
        color=color,
        marker=marker,
        sample_frac=sample_frac,
        output_file=output_file,
        dropna=dropna,
    )


def main():
    """Parse CLI arguments and dispatch to the selected table utility."""
    parser = ArgumentParser(description="Tool to manipulates the Hyperion data tables")
    parser.add_argument(
        "--cfg", action=ActionConfigFile, help="configuration file in YAML format"
    )

    subcommands = parser.add_subcommands()
    for subcommand in subcommand_list:
        parser_func = f"make_{subcommand}_parser"
        subparser = globals()[parser_func]()
        subcommands.add_subcommand(subcommand, subparser)

    args = parser.parse_args()
    subcommand = args.subcommand
    kwargs = namespace_to_dict(args)[args.subcommand]
    config_logger(kwargs["verbose"])
    del kwargs["verbose"]
    del kwargs["cfg"]
    globals()[subcommand](**kwargs)


if __name__ == "__main__":
    main()
