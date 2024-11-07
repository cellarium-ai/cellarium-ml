"""Helper functions for finding and localizing data from a Terra workspace.

An example workflow...
Create a new notebook runtime VM (using the terra-jupyter-pcl image of course)

Open up a new notebook.

From within the notebook, do

```
from terra_data import get_terra_data_table, localize_terra_data

samples = get_terra_data_table(table='sample')
# possibly take a subset: samples = samples[my_samples_of_interest]
localize_terra_data(
    terra_data_table=sample_subset,
    columns_to_localize=['count_matrix_h5',
                         'cellbender_v2_output_dir',
                         'scrinvex_out'],
    map_column_to_local_folder={'count_matrix_h5': '/home/jupyter-user/data/cellranger',
                                'cellbender_v2_output_dir': '/home/jupyter-user/data/cellbender',
                                'scrinvex_out': '/home/jupyter-user/data/scrinvex'},
    file_renaming_fcns={'count_matrix_h5': (lambda file, sample: sample + '.h5')},
    map_column_to_gs_path={'cellbender_v2_output_dir': (lambda d: os.path.join(d, '*_filtered.h5'))},
)
```

"""

try:
    import firecloud.api as fapi
except ImportError as e:
    print('You need to install the firecloud toolkit.  Try `pip install firecloud`')
    raise e
import pandas as pd

import subprocess
from subprocess import Popen, PIPE
import os
import io
from typing import List, Dict, Callable, Optional


def get_terra_data_table(billing_project: Optional[str] = None,
                         workspace: Optional[str] = None,
                         bucket: Optional[str] = None,
                         table: str = 'sample',
                         verbose: bool = True) -> Optional[pd.DataFrame]:
    """Get a data table from a Terra workspace.
    
    Args:
        billing_project: The Terra billing project name (part of the full
            workspace name before the slash '/')
        workspace: The Terra billing project name (part of the full
            workspace name after the slash '/')
        bucket: The Google bucket associated with the workspace, for example
            gs://fc-01234...../
        table: Name of the Terra data table, e.g. 'sample' or 'sample_set', etc.
            (This is also referred to by Terra as the table "entity" type)
        verbose: True to print out information

    NOTE: None values are allowed as inputs because from within a Jupyter
    notebook running on Terra, there are environment variables set where we can
    read these values automatically, making it easier.
    
    Returns:
        df: Pandas version of the Terra data table
        
    """

    # inputs
    if billing_project is None:
        billing_project = os.environ['WORKSPACE_NAMESPACE']
    if workspace is None:
        workspace = os.environ['WORKSPACE_NAME']
    if bucket is None:
        bucket = os.environ['WORKSPACE_BUCKET'] + "/"
    else:
        assert bucket.startswith('gs://fc-'), 'All Terra workspace buckets ' \
                                              'should start with "gs://fc-"'

    # workspace information
    if verbose:
        print('Billing project: ' + billing_project)
        print('Workspace: ' + workspace)
        print('Workspace storage bucket: ' + bucket)

    # take a look at all the entities in our workspace
    firecloud_return = fapi.list_entity_types(billing_project, workspace)
    if str(firecloud_return) == '<Response [401]>':
        print('Probable permissions issue: <Response [401]>')
        print('Try\ngcloud auth login --update-adc\n'
              'https://support.terra.bio/hc/en-us/articles/360042259232?page=1#comment_360012283152')
        return None
    ent_types = firecloud_return.json()
    try:
        if verbose:
            for t in ent_types.keys():
                print(t, "count:", ent_types[t]['count'])
    except Exception as e:
        print(ent_types)
        raise e

    # ensure that the table we are requesting exists
    assert table in ent_types.keys(), \
        f'The specified table "{table}" is not one of the tables that appears ' \
        f'in this workspace.  Choose from {ent_types.keys()}'

    # request the table and format as a dataframe
    if table == 'sample_set':
        raise NotImplementedError('Not yet implemented for "sample_set" data (complicated)')
    else:
        df = pd.read_csv(io.StringIO(fapi.get_entities_tsv(billing_project, workspace, table,
                                                           model='flexible').text), sep='\t')

    return df


def localize_terra_data(terra_data_table: pd.DataFrame,
                        columns_to_localize: List[str],
                        map_column_to_local_folder: Dict[str, str],
                        file_renaming_fcns: Optional[Dict[str, Optional[Callable[[str, str], str]]]] = None,
                        map_column_to_gs_path: Optional[Dict[str, Optional[Callable[[str], str]]]] = None):
    """Download data from Google Cloud to the local disk using `gsutil cp`.
    The location of cloud data is input in the form of a Terra data table, which
    has columns that hold information about the location of files.

    Args:
        terra_data_table: Dataframe containing file information, as obtained
            from `get_terra_data_table()`. (Probably a subset of full dataframe)
        columns_to_localize: The columns of the dataframe that contain file
            locations for data which should be localized.
        map_column_to_local_folder: Specify where should the data go on the
            local disk. A dictionary with keys as column names and values as
            local directory names.
            Example:
                {'cellranger_h5': '/home/jupyter-user/data/cellranger',
                 'cellbender_v2': '/home/jupyter-user/data/cellbender'}
        file_renaming_fcns: Specify whether the downloaded file name should be
            different from the cloud file name. If so, provide a dictionary of
            lambda functions (one for each column) that constructs the local
            destination filename from (cloud_file, entity/sample).
            Example:
                {'cellranger_h5': (lambda file, sample: f'{sample}.h5')}
                Here the file is named 'raw_feature_bc_matrix.h5' for every sample.
                That's not very useful as a local filename, since it will be the
                same for all samples and can lead to confusion. The above input
                specifies that the local copy of the file should ignore the
                original file name and instead be called (sample + '.h5')
        map_column_to_gs_path: If the table entry is not the full file path, but
            instead a directory or something you can use to construct the full
            path, then you can specify the function that constructs the full
            path here, as a dictionary of lambda functions (one for each column).
            Example:
                If you have cellranger directories but not the full h5 paths:
                {'cellranger_dir': (lambda folder: os.path.join(folder, 'raw_feature_bc_matrix.h5'))}

    Returns:
        None

    """

    # input checks
    for col in columns_to_localize:
        assert col in terra_data_table.columns, \
            f'Specified column "{col}" is not one of the columns of the input ' \
            f'terra_data_table: {terra_data_table.columns}'
        assert col in map_column_to_local_folder.keys(), \
            f'Specified column "{col}" is not one of the keys of the input ' \
            f'map_column_to_local_folder dict: {map_column_to_local_folder.keys()}. ' \
            f'In map_column_to_local_folder, include "{col}" as a dictionary ' \
            f'key whose value is the local destination file path. For example: ' \
            '{' + f'"{col}": "/home/jupyter-user/data/{col}"' + '}'

        # ensure destination folders exist on the local disk and are writeable
        path = map_column_to_local_folder[col].rstrip("/")
        os.makedirs(path, exist_ok=True)  # create full path if it doesn't exist
        assert os.access(path, os.W_OK), f'Cannot write to specified directory {path}, ' \
            f'as specified in the input map_column_to_local_folder'

    def _ignore_nulls(fcn):
        """wrap a function to return empty string if input is not a string (nan)"""
        def fun(*args):
            if type(args[0]) == str:
                return fcn(*args)
            else:
                print(f'WARNING: missing value in {fcn}({args})')
                return ''
        return fun

    def cloud_path_fcn(column: str) -> Callable[[str], str]:
        if ((map_column_to_gs_path is None)
                or (column not in map_column_to_gs_path.keys())
                or (map_column_to_gs_path[column] is None)):
            return lambda s: s
        else:
            return _ignore_nulls(map_column_to_gs_path[column])

    def file_rename_fcn(column: str) -> Callable[[str, str], str]:
        if ((file_renaming_fcns is None)
                or (column not in file_renaming_fcns.keys())
                or (file_renaming_fcns[column] is None)):
            return lambda file, entity: file
        else:
            return _ignore_nulls(file_renaming_fcns[column])

    # figure out the "entity" column and the unique IDs
    all_cols = terra_data_table.columns
    entity_col = all_cols[[c.startswith('entity:') for c in all_cols]].item()
    ids = terra_data_table[entity_col].values.tolist()

    # one column at a time
    for col in columns_to_localize:

        # file paths
        get_path = cloud_path_fcn(col)
        paths = [get_path(i) for i in terra_data_table[col].values.tolist()]

        if ((file_renaming_fcns is None)
                or (col not in file_renaming_fcns.keys())
                or (file_renaming_fcns[col] is None)):

            # file renaming is not necessary so we can do a parallel download
            print(f'Parallel download for "{col}" ======================\n')
            print('stdin:')
            print('\n'.join(paths), end='\n\n')
            cmd = ['gsutil', '-m', 'cp', '-I', map_column_to_local_folder[col]]
            print(' '.join(cmd))
            p = Popen(cmd, stdout=PIPE, stdin=PIPE, stderr=PIPE)
            result = p.communicate(input='\n'.join(paths).encode())
            print(result[1].decode())

        else:

            # we must download one file at a time
            print(f'Serial downloads for "{col}" =======================\n')

            # handle the renaming of individual files
            rename_file = file_rename_fcn(col)

            for index, file in zip(ids, paths):

                cmd = ['gsutil', 'cp', file,
                       os.path.join(map_column_to_local_folder[col], rename_file(file, index))]
                print(' '.join(cmd))
                result = subprocess.run(cmd, stdout=PIPE, stderr=PIPE)
                print(result.stderr.decode())


def create_tsv_terra_data_table(column_path_dict: Dict[str, str],
                                sample_name_constructor: Dict[str, Callable[[str], str]]) -> pd.DataFrame:
    """Iff your files are located in a Google bucket in such a way that they can
    be located using `gsutil ls some_path_with_wildcards`, then you can use this
    function to construct a data table which can be uploaded to Terra.

    Args:
        column_path_dict: Dictionary of column names and how to find the
            associated data.
            Example:
                {'column_name1': 'gs://path/with/wildcards/*/raw_feature_bc_matrix.h5',
                 'column_name2': 'gs://path/with/wildcards/*/possorted_genome_bam.bam',
                 'column_name3': 'gs://path/with/wildcards/*/*_out_filtered.h5'}
        sample_name_constructor: Specifies how to construct the sample name
            from one of the column filepaths.
            Example:
                {'column_name3': lambda s: s.split('.')[-1].split('_out_filtered.h5')[0]}

    Returns:
        df: Pandas dataframe which can be saved as a TSV for upload to Terra.

    """

    raise NotImplementedError
