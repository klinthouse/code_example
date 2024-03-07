import os
import re
from collections import Counter, defaultdict
from functools import reduce
from itertools import product
import time
import random

import awswrangler as wr
import boto3
import hashlib
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import warnings


from ..align import align as al
from ..endpoints import master_function as mf
from ..query import query as q
from ..read import read_pipe as read


PARENT_DIR = os.path.join('' + '../' * os.getcwd().split('/')[::-1].index('ds_dl_organization'))
DRUG_DICT_PATH = os.path.join(PARENT_DIR, 'data', 'drug_dictionary.csv')
pd.options.mode.chained_assignment = None

drug_dict = pd.read_csv(DRUG_DICT_PATH)
for column in ['Brand', 'Generic']:
    drug_dict[column] = drug_dict[column].str.lower()

GENERIC_TO_BRAND = drug_dict.dropna(subset=['Generic', 'Brand']).set_index('Generic')['Brand'].to_dict()
BRAND_TO_MOA = drug_dict.dropna(subset=['Brand', 'MOA']).set_index('Brand')['MOA'].to_dict()
GENERIC_TO_BRAND['certolizumab-pegol'] = 'cimzia'

def do_pivot(table, specimens, features, metric_, destination, random_delay):
    glue = boto3.client("glue", region_name="us-east-1")
    job = "test_job_gene_pivot_RC_1"
    try:
        if random_delay == "yes":
            time.sleep(random.randint(0, 30))
        response = glue.start_job_run(
            JobName=job,
            Arguments={
                "--table": table,
                "--specimens": specimens,
                "--features": features,
                "--metric_": metric_,
                "--destination": destination,
            },
        )
    except:
        if random_delay == "yes":
            time.sleep(random.randint(0, 300))
        response = glue.start_job_run(
            JobName=job,
            Arguments={
                "--table": table,
                "--specimens": specimens,
                "--features": features,
                "--metric_": metric_,
                "--destination": destination,
            },
        )
    status_detail = glue.get_job_run(JobName=job, RunId=response.get("JobRunId"))
    status = status_detail.get("JobRun").get("JobRunState")
    while status == "RUNNING":
        status_detail = glue.get_job_run(JobName=job, RunId=response.get("JobRunId"))
        status = status_detail.get("JobRun").get("JobRunState")
        print(status)
        time.sleep(20)
    print("Finished")

def prepare_biologic_data(df, parameter):
    """Extracts and prepares biologic data from the given dataframe based on the provided parameter."""
    df_history = df.sort_values(['patient_id', 'data_date'])
    data_biologic = df_history[df_history.parameter == parameter][['patient_id', 'data_date', 'value']]
    return data_biologic.dropna().drop_duplicates()

def process_reverse_date_order(np_biologic):
    """Processes biologic data rows in reverse date order to generate upper time bounds."""
    future_id, future_value, future_date = None, None, None
    upper_timebound = []
    misordered_record = []

    for current_id, current_date, current_value in np_biologic[::-1]:

        # if new sample, reset future id, date and values
        if current_id != future_id:
            future_id, future_value, future_date = None, None, pd.Timestamp('2200-01-01')

        # if there are multiple treatments on a same day, reorder the drug timeline weighing the continuation of the treatment (TNFi, [])
        if (current_date == future_date) & (len(upper_timebound) > 1):
            if ((current_value != future_value)&
                (current_value == upper_timebound[-2][2])&
                (current_id == upper_timebound[-2][0])):
                misordered_record = upper_timebound.pop()
                misordered_record[-1] = current_date
                future_id, future_date, future_value, enddate = upper_timebound[-1]

        # if current parameter value is new, set already processed date (or reset date) as the "ceil date"
        if current_value != future_value:
            enddate = future_date
        
        # assign already processed row as the "future" values
        future_id, future_date, future_value = current_id, current_date, current_value

        # store the already processed row with the ceil date
        upper_timebound.append([current_id, current_date, current_value, enddate])

        # add correected misorder corrections into the append
        if len(misordered_record) > 0:
            upper_timebound.append(misordered_record)
            misordered_record = []

    return upper_timebound[::-1]

def construct_upper_timebound_df(upper_timebound):
    """Constructs a DataFrame from the given upper time bounds."""
    df_all_upbound = pd.DataFrame(
        upper_timebound,
        columns=['patient_id', 'init_date', 'value', 'upper_timebound']
    )
    return df_all_upbound.sort_values('init_date').groupby(
        ['patient_id', 'upper_timebound', 'value']).first().reset_index(
        ).sort_values(['patient_id', 'init_date'])

def get_init_upbound(df, parameter='biologic_grp'):
    """create lower and upper timebounds for each epochs

    Parameters
    ----------
    df : DataFrame
        longformat - must have dates, parameter and values.
    parameter : str
        determining factors for the epochs
        
    Returns
    -------
    df_init_upbound : DataFrame
        longformat with initial date and end date of each epoch attached.


    Examples
    --------
    This example code will find init date (lower bound) and ceil date (upper bound) of each
    biologic treatment.

    >>> session = boto3.Session(region_name='us-east-1')
    >>> df = wr.athena.read_sql_query(sql="SELECT * FROM clinical_edc_biobank_longformat;",
                                database="smdsinput", boto3_session=session)
    >>> df.data_date = pd.to_datetime(df.data_date, errors='coerce')
    >>> df.patient_id = [patient_id.upper() for patient_id in df.patient_id]
    >>> df_init_upbound = get_init_upbound(df)

    """
    data_biologic = prepare_biologic_data(df, parameter)
    upper_timebound = process_reverse_date_order(data_biologic.to_numpy())
    return construct_upper_timebound_df(upper_timebound)

def scan_unordered_sequence(subsequence, sequence):
    """check if the elements in subsequence exist in a given sequence

    Parameters
    ----------
    subsequence : array
    sequence : array
        
    Returns
    -------
    exist : bool
        if True, all the subsequence elements exist in the sequence
        if False, sequence missing element(s) in the subsequence

    Examples
    --------

    >>> scan_unordered_sequence(['a', 'b', 'c', 'c'], ['a', 'b', 'c', 'c', 'd', 'e'])
    >>> scan_unordered_sequence(['a', 'b', 'c', 'c'], ['a', 'b', 'c', 'd', 'e'])
    >>> scan_unordered_sequence([], [])

    """
    count_subsequence = Counter(subsequence)
    count_sequence = Counter(sequence)

    for key in count_subsequence:
        if key in count_sequence:
            if count_subsequence[key] > count_sequence[key]:
                return False
        else:
            return False
            
    return True

def locate_ordered_sequence(info, subsequence):
    """check if the subsequence with its exact order exist in a given sequence

    Parameters
    ----------
    info : array(any, str, ...)
        sequence of array that contain list of information which index 1 must contain 
        information related to the subsequence
    subsequence : array
        ordered subsequence to search for in the sequence
        
    Returns
    -------
    subseq_info : array or nan
        if subsequence with its exact order exist in the sequence, returns the matching 
            segment of the info array
        else, returns nan value.

    Examples
    --------

    >>> locate_ordered_sequence(
        [[2023-1-23, 'b'], [2023-1-24, 'c'], [2023-1-25, 'd']], 
        ['b', 'c']
        )

    """
    sequence = np.append('naive', info[:, 1])

    #TODO: remove to higher level, when we do df.replace
    # in case subsequence has generic name, replace with the brand
    
    subsequence = [GENERIC_TO_BRAND[drug] if drug in GENERIC_TO_BRAND else drug for drug in subsequence]

    if 'exposed' in subsequence[0]:
        sequence = list(sequence)
        #TODO: impute values from exposed to target from sequence and just pass it on 
        try:
            i = (sequence).index(subsequence[0].split('_')[0])
            j = sequence[(i + 1):].index(subsequence[-1]) + i + 1

            return info[i - 1 : j]
        except:
            return np.nan

    if scan_unordered_sequence(subsequence, sequence):
        pass
    else:
        return np.nan

    i, j, k = 0, 0, 0

    while j < len(sequence) and i < len(subsequence):

        if sequence[j] == subsequence[i]:
            if i == 0:
                k = j
            i += 1
        else:
            if i > 0:
                j = k
            i = 0

        j += 1
    
    if i == len(subsequence):
        if k == 0:
            return info[k : j - 1]
        return info[k - 1 : j - 1]
    else:
        return np.nan

def get_cohort_init_upbound(df, g_patient_bound, condition):
    """filter longformat by the cohorts that meets the given condition. 

    Parameters
    ----------
    df : DataFrame
        longformat mainly to provide additional information to the processed data of cohorts to
        maintain the information from the original longformat. The additional information is any 
        columns other than patient_id, value, init_date, upper_timebound
    g_patient_bound : pd.DataFrame.groupby
        longformat with init_date and upper_timebound that is grouped by patient id
    condition : array
        subsequence of the array 
        
    Returns
    -------
    cohort_data : DataFrame
        longformat data of patients that meet the given condition

    """

    cohort_init_upbound_list = []

    for _, df_patient in g_patient_bound:

        value_history = df_patient[['patient_id', 'value', 'init_date', 'upper_timebound']].to_numpy()

        # ADD for loop for condition 
        condition_history = locate_ordered_sequence(value_history, condition)
        if isinstance(condition_history, np.ndarray):
            cohort_init_upbound_list.append(condition_history[-1])

    if cohort_init_upbound_list:
        cohort_init_upbound_list = np.vstack(cohort_init_upbound_list)

    df_cohort_init_upbound = pd.DataFrame(
        cohort_init_upbound_list,
        columns=['patient_id', 'value', 'init_date', 'ceil_date']).drop(columns='value')

    cohort_data = pd.merge(
        df.sort_values(by=["data_date"]),
        df_cohort_init_upbound,
        how='right',
        on=['patient_id']
    )
    
    return cohort_data

def reset_init_date(df, num_repeats):


    # Get specimen information
    df_specimen = df[df.parameter == 'specimen_name']
    specimen_mask = (((df_specimen.data_date - df_specimen.init_date).dt.days > 30) &
        ((df_specimen.ceil_date - df_specimen.data_date).dt.days > 0))

    if len(df_specimen[specimen_mask]['data_date']) >= num_repeats:
        new_init_date = df_specimen[specimen_mask]['data_date'].iloc[num_repeats - 1]
        df['init_date'] = new_init_date
        return df
    else:
        return None

def patch_cohort_init(cohort_data, boundary=False):
    """distill the longformat by selecting parameter values that are closest to and less than the 
    initialization date. If the boundary is given, pick the value with the constratins that it is
    not earlier than the given boundary.

    Parameters
    ----------
    cohort_data : DataFrame
        longformat with selected cohort of patients with init date and ceil date
    boundary : int, default: False
        boundary in the unit of month that puts constraint to the scope of initial parameter
        value selection
        
    Returns
    -------
    df_cohort_init : DataFrame
        longformat with selected cohort of patients with init date and ceil date with selected
        initial parameter values
    """

    cohort_data_uplimited = cohort_data[
        cohort_data['data_date'] <= cohort_data['ceil_date']
        ]

    # NOTE: here we have sort_values by `value` only to make the consistent query. 
    # (e.g. patient with multiple specimen name with same datadate results in inconsistency)
    if boundary == False:
        baseline_data = cohort_data_uplimited[
            (cohort_data_uplimited['data_date'] <= cohort_data_uplimited['init_date'])
            ].dropna(subset=['value']).sort_values(by=['data_date', 'value']).groupby([
                "patient_id", "parameter"
                ]).last() 
    else:
        baseline_data = cohort_data_uplimited[
            (cohort_data_uplimited['data_date'] <= cohort_data_uplimited['init_date'])
            &
            (cohort_data_uplimited['init_date'] - pd.to_timedelta(boundary*30, 'days') <= cohort_data_uplimited['data_date'])
            ].dropna(subset=['value']).sort_values(by=['data_date', 'value']).groupby([
                "patient_id", "parameter"
                ]).last()

    baseline_data['data_date_og'] = baseline_data['data_date']
    baseline_data['data_date'] = baseline_data['init_date']
    baseline_data['visit'] = 0

    df_cohort_init = baseline_data.sort_index().reset_index()
    df_cohort_init['epoch'] = 0 #technically, not necessary anymore
    
    df_cohort_init.drop(
        columns=['init_date', 'ceil_date'],
        inplace=True
        )

    return df_cohort_init


def patch_cohort_target(cohort_data, month, boundary_window=False):
    """distill the longformat by selecting parameter values that are closest to the interested month
    after the treatment and less than the treatment end date. If the boundary_window is given, pick
    the value with the constratins that it is within the given boundary window.

    Parameters
    ----------
    cohort_data : DataFrame
        longformat with selected cohort of patients with init date and ceil date
    month : int
        month of the interested endpoint
    boundary_window : int, default False
        boundary in the unit of month that puts constraint to the scope of endpoint parameter 
        value selection
        
    Returns
    -------
    df_cohort_init : DataFrame
        longformat with selected cohort of patients with init date and ceil date with selected
        initial parameter values
    """
    cohort_data_uplimited = cohort_data[
        cohort_data['data_date'] <= cohort_data['ceil_date']
        ]

    def get_timediff_without_limit(row, month, signed=True):
        try:
            diff = (row['init_date'] + pd.Timedelta(days=30*month) - row['data_date'])
        except:
            diff = pd.Timedelta(days=2**16 + 5000)
            print(row)
        if signed:
            return diff
        return abs(diff)
    
    cohort_data_uplimited['target_date_diff'] = cohort_data_uplimited.apply(get_timediff_without_limit, month=month, signed=False, axis=1)
    cohort_data_uplimited['signed_target_date_diff'] = cohort_data_uplimited.apply(get_timediff_without_limit, month=month, axis=1)
    # abs(
    #     (cohort_data_uplimited['init_date'] + pd.Timedelta(days=30*month))
    #     - cohort_data_uplimited['data_date'])#.replace(pd.Timestamp('1700-01-01'), pd.Timestamp('1700-01-01')))

    if boundary_window == False:
        target_data = cohort_data_uplimited[
            (cohort_data_uplimited['data_date'] > cohort_data_uplimited['init_date'] + pd.to_timedelta(30, 'days'))
        ].dropna(subset=['value']).sort_values('target_date_diff').groupby([
            "patient_id", "parameter"
            ]).first()
    else:
        target_data = cohort_data_uplimited[
            (cohort_data_uplimited['data_date'] > cohort_data_uplimited['init_date'] + pd.to_timedelta(30, 'days'))
            &
            (cohort_data_uplimited['signed_target_date_diff'] >= -pd.Timedelta(days=30*boundary_window[-1]))
            &
            (cohort_data_uplimited['signed_target_date_diff'] <= pd.Timedelta(days=30*boundary_window[0]))
        ].dropna(subset=['value']).sort_values('target_date_diff').groupby([
            "patient_id", "parameter"
            ]).first()
    
    target_data['data_date_og'] = target_data['data_date']
    target_data['data_date'] = target_data['init_date'] + pd.Timedelta(days=30*month)
    target_data['visit'] = month

    df_cohort_target = target_data.sort_index().reset_index()
    df_cohort_target['epoch'] = 0 #technically, not necessary anymore
    
    df_cohort_target.drop(
        columns=['init_date', 'ceil_date', 'target_date_diff', 'signed_target_date_diff'],
        inplace=True
        )

    return df_cohort_target


def aggregate_multisource(df, params, datatype, agg_fn='mean', replace_dict=dict()):
    """aggregates the duplicated parameter values from multiple data source (i.e., 
    LIMS, EDC).

    Parameters
    ----------
    df : DataFrame
        longformat
    params : str
        parameters with multiple data sources
    datatype : float, int or str
        parameter value type
    agg_fn: str
        aggregation method
    replace_dict: dict
        `key` of values that need to be renamed or replaced with `value` of values
        
    Returns
    -------
    df : DataFrame
        longformat with aggregated duplicated values 
    """

    for param in params:
        df_param = df[df.parameter.isin([param, f'{param}_lims'])].replace(replace_dict)
        if datatype in ['float', 'int']:
            df_param.value = pd.to_numeric(df_param.value, errors='coerce')
        df_param.dropna(subset=['value'], inplace=True)
        #TODO: get rid of initdate when it is plugged into the query_v2 pipeline
        if 'data_source' in df_param:
            df_param_agg = df_param.groupby(['patient_id', 'data_date']).agg({
                    'value': 'mean',
                    'data_source': lambda x: list(x)
                }).reset_index()
        else:
            df_param_agg = df_param.groupby(['patient_id', 'data_date']).agg({
                'value': 'mean'
            }).reset_index()
        df_param_agg['parameter'] = param
        df =  pd.concat([df[~df.parameter.isin([param, f'{param}_lims'])], df_param_agg])

    return df

def reorder_strings(strings):
    """
    Reorders the list of strings, moving strings with the 'ex_' prefix to the back.
    """
    # Separate strings based on whether they start with 'ex_'
    prefixed = [s for s in strings if s.startswith("ex_")]
    non_prefixed = [s for s in strings if not s.startswith("ex_")]

    # Combine the lists, with non-prefixed strings first
    return non_prefixed + prefixed


class CohortCurator():
    """
        DataFrame of variouscohorts in dictionary.
        Parameters
        ----------
        data : ndarray (structured or homogeneous), Iterable, dict, or DataFrame
            Dict can contain Series, arrays, constants, dataclass or list-like objects. If
            data is a dict, column order follows insertion-order. If a dict contains Series
            which have an index defined, it is aligned by its index.
            .. versionchanged:: 0.25.0
            If data is a list of dicts, column order follows insertion-order.
        index : Index or array-like
            Index to use for resulting frame. Will default to RangeIndex if
            no indexing information part of input data and no index provided.
        columns : Index or array-like
            Column labels to use for resulting frame when data does not have them,
            defaulting to RangeIndex(0, 1, 2, ..., n). If data contains column labels,
            will perform column selection instead.
        dtype : dtype, default None
            Data type to force. Only a single dtype is allowed. If None, infer.
        copy : bool or None, default None
            Copy data from inputs.
            For dict data, the default of None behaves like ``copy=True``.  For DataFrame
            or 2d ndarray input, the default of None behaves like ``copy=False``.
            If data is a dict containing one or more Series (possibly of different dtypes),
            ``copy=False`` will ensure that these inputs are not copied.
            .. versionchanged:: 1.3.0
    """
    def __init__(
        self,
        table,
        database,
        conditions_list,
        month,
        feature_list,
        endpoint_type,
        parameter='biologic_grp',
        debug_mode=False,
        temp=False,
        **kwarg
        ):
        
        self.table = table
        self.database = database
        self.conditions_list = conditions_list
        self.month = month
        self.feature_list = feature_list
        self.endpoint_type = endpoint_type
        self.parameter = parameter
        self.debug_mode = debug_mode
        self.temp = temp
        self.kwarg = defaultdict(bool)
        self.kwarg.update(kwarg)


    def _create_hash(self, table, database, condition, feature_set=None):

        parent_hash = hashlib.md5((table+'-'+database).encode()).hexdigest()
        condition_hash = hashlib.md5(str(tuple(condition)).encode()).hexdigest()
        if feature_set:
            feature_hash = hashlib.md5((feature_set).encode()).hexdigest()
            hash_str = parent_hash + '-' + condition_hash + '-' + feature_hash
        else:
            hash_str = parent_hash + '-' + condition_hash

        return hash_str


    def _recall_cohort(self, conditions_list=None):

        table = self.table
        database = self.database
        new_conditions = []
        all_cohorts = {}
        conditions_list = self.conditions_list

        # if len(conditions_list) == 1: 
        #     conditions_list = ['exposed*'] + conditions_list
        for i, condition in enumerate(conditions_list):
            if isinstance(condition, list):
                if 'exposed' in condition:
                    raise Exception('Have "exposed" on its own and not in a list of prior condition')
            else:
                if 'exposed' == condition.lower():
                    if conditions_list[-1] in set(drug_dict['MOA']):
                        conditions_list[i] = list(set(drug_dict['MOA']) - set([conditions_list[-1]])) 
                    if conditions_list[-1] in BRAND_TO_MOA:
                        conditions_list[i] = list(set(BRAND_TO_MOA.keys()) - set([conditions_list[-1]]))
                    if conditions_list[-1] in GENERIC_TO_BRAND:
                        conditions_list[i] = list(set(GENERIC_TO_BRAND.keys()) - set([conditions_list[-1]]))
                # if '*' in condition.lower():
                #     conditions_list[i].append('naive')


        if len(np.hstack(conditions_list)) > len(conditions_list):
            conditions_list = [event if isinstance(event, list) else [event] for event in conditions_list]
            std_conditions_list = reduce(lambda pre_event, post_event: product(pre_event, post_event), conditions_list)
            std_conditions_list = list(std_conditions_list)
        else:
            if isinstance(conditions_list[-1], str):
                std_conditions_list = [conditions_list]
            std_conditions_list = conditions_list
    
        for one_condition in std_conditions_list:
            one_condition = np.hstack(one_condition)
            hash_str = self._create_hash(table, database, str(tuple(one_condition)))
            if 'tag' in self.kwarg:
                hash_str += ('-' + self.kwarg['tag'])
            not_exist, query_info = read.check_query_info(hash_str)
            
            if self.debug_mode:
                not_exist = True
            if not not_exist:
                if query_info['status'][0] == 'complete':
                    print(f"found {str(tuple(one_condition))} cohort!!!")
                    print(hash_str)
                    cohort_data = read.retrieve_query_results(hash_str)
                    all_cohorts[tuple(one_condition)] = cohort_data
            else:
                print(f"didn't find {str(tuple(one_condition))} cohort...")
                print(hash_str)
                new_conditions.append(one_condition)

        self.new_conditions = new_conditions
        self.all_cohorts = all_cohorts
        return all_cohorts, new_conditions


    def prep_clinical(self, table=None, database=None):

        table, database = self.table, self.database 

        #START: TEMP for og aims table
        if 'aims' in table:
            session = boto3.Session(region_name='us-east-1')
            df = wr.athena.read_sql_query(sql=f"SELECT * FROM {table}_etl;",
                                        database=database, boto3_session=session)
            df = df[~df.parameter.isin(['drug_type', 'drugname'])]
            # df['parameter'] = df['parameter'].replace({'kit_id': 'specimen_name'})

            session = boto3.Session(region_name='us-east-1')
            df_med = wr.athena.read_sql_query(sql=f"SELECT * FROM {table}_megamed;",
                                        database=database, boto3_session=session)

            aims_moa = df_med[['patient_id', 'startdate', 'moa']].rename(columns={'moa': 'value', 'startdate':'data_date'}).dropna()
            aims_moa['parameter'] = 'biologic_grp'

            clinical_long = pd.concat([df[~df.parameter.isin(['drug_type', 'drugname'])], pd.concat([aims_moa])])
            clinical_long.patient_id = [patient_id.upper() for patient_id in clinical_long.patient_id]
            clinical_long.replace({'':None, 'NA':None}, inplace=True)
            clinical_long['parameter'] = clinical_long['parameter'].str.lower()

            session = boto3.Session(region_name='us-east-1')
            df_kit_id_map = wr.athena.read_sql_query(sql="SELECT * FROM specimen_index;",
                                            database="smdsinput", boto3_session=session)
            df_kit_map_clean = df_kit_id_map.loc[
                df_kit_id_map['processed'] == True, 
                ['export_date', 'kit_id', 'specimen_name']
                ].sort_values(['kit_id', 'export_date']).drop_duplicates(
                subset='kit_id', 
                keep='last'
                )
            
            df_specimen = clinical_long[clinical_long.parameter == 'kit_id'].merge(
                df_kit_map_clean, left_on='value', right_on='kit_id')[['patient_id', 'data_date', 'specimen_name']]
            
            df_specimen['parameter'] = 'specimen_name'
            df_specimen.rename(columns={'specimen_name':'value'}, inplace=True)

            df_prior_tnfi = clinical_long[clinical_long['parameter'] == 'prior_tnfi']
            df_prior_tnfi_exposed = df_prior_tnfi[df_prior_tnfi.value == '1']
            df_prior_tnfi_exposed['parameter'] = 'biologic_grp'
            df_prior_tnfi_exposed['data_date'] = pd.Timestamp('1900-01-01')
            df_prior_tnfi_exposed['value'] = 'TNFi'

            clinical_long = pd.concat([clinical_long, df_specimen, df_prior_tnfi_exposed])
        #END: TEMP for og aims table
                  
        else:
              clinical_long = al.load_clinical_data(table, database)           
        clinical_long = clinical_long.rename(columns={"patient": "patient_id"})

        longformat_columns = set(clinical_long) & set(['patient_id', 'data_date', 'parameter','value','data_type','data_source'])

        clinical_long = clinical_long[list(longformat_columns)]

        clinical_long = clinical_long.dropna(subset=["data_date"])
        clinical_long = clinical_long.loc[clinical_long["data_date"] != "NA"]
        clinical_long = al.format_string_into_datetime(clinical_long, "data_date")
        
        clinical_long = aggregate_multisource(
            clinical_long,
            ['crp', 'pt_global_assess'],
            'float',
            replace_dict={"<3.6":'3.6'}
            )

        parameter_rename = {
            'drugname': 'biologic_grp',
            'drug_name': 'biologic_grp', ### Patch
            'simponi-aria': 'simponi aria',
            'ccp':'anticcp_binary',
            'how_much_pain_have_you_had_because_of_your_condition_over_the_past_week_please_indicate_below_how_18015':'pt_pain'
            }
        parameter_rename.update(GENERIC_TO_BRAND)
        clinical_long.replace(parameter_rename, inplace=True)

        exposed_mask = (clinical_long['data_date'] == pd.Timestamp('1900-01-01')) 
        clinical_long.loc[exposed_mask, 'value'] = clinical_long.loc[exposed_mask, 'value'] + 'exposed_'

        session = boto3.Session(region_name='us-east-1')
        df_kit_id_map = wr.athena.read_sql_query(sql="SELECT * FROM specimen_index;",
                                        database="smdsinput", boto3_session=session)
        df_kit_map_clean = df_kit_id_map.loc[
            (df_kit_id_map['processed'] == True) & (df_kit_id_map['rine'].astype(float) >= 4), 
            ['export_date', 'kit_id', 'specimen_name']
            ].sort_values(['kit_id', 'export_date']).drop_duplicates(
            subset='kit_id', 
            keep='last'
            )

        df_specimen = clinical_long[clinical_long.parameter == 'specimen_name'].merge(
            df_kit_map_clean, left_on='value', right_on='specimen_name')[['patient_id', 'data_date', 'specimen_name']]
        clinical_long = clinical_long[clinical_long.parameter != 'specimen_name']

        df_specimen['parameter'] = 'specimen_name'
        df_specimen.rename(columns={'specimen_name':'value'}, inplace=True)

        clinical_long = pd.concat([clinical_long, df_specimen])

        self.clinical_long = clinical_long
        return clinical_long

    
    def generate_cohorts(self, new_conditions=None):

        clinical_long = self.prep_clinical()
        table = self.table
        database = self.database
        all_cohorts = {}
        new_conditions = self.new_conditions
        
        #TODO: have separate get_init_upbound when there are multiple queries which one does not have samepiror_exposure
        i = len(new_conditions[0]) - 2
        num_same_exposure = 0
        while i >= 0:
            if new_conditions[0][-1] == new_conditions[0][i]:
                num_same_exposure += 1
            else:
                break

            i -= 1

        if new_conditions[-1][-1] in set(drug_dict['MOA']):
            clinical_long.replace(BRAND_TO_MOA, inplace=True)

        clinical_long = clinical_long[~((clinical_long['value'] == 'DMARD') 
            & (clinical_long['parameter'] == 'biologic_grp'))]
        
        self.df_init_upbound = get_init_upbound(clinical_long, self.parameter)

        if self.parameter in ['specimen_name', 'kit_id']:
            clinical_long_bound = clinical_long.merge(self.df_init_upbound.drop(columns='value'), how='right', on='patient_id')
            clinical_long_bounded = clinical_long_bound[
                (clinical_long_bound.data_date >= clinical_long_bound.init_date)
                & (clinical_long_bound.data_date < clinical_long_bound.upper_timebound)
                ]
            self.df_init_upbound = get_init_upbound(clinical_long_bounded, 'biologic_grp')
            #### if inclusion exclusion is sensitive to blood draw, 
            #### replace the specimen date as specimen with self.df_init_upbound

        self.g_patient_bound = self.df_init_upbound.groupby('patient_id')

        for one_condition in new_conditions:

            hash_str = self._create_hash(table, database, str(tuple(one_condition)))
            if 'tag' in self.kwarg:
                hash_str += ('-' + self.kwarg['tag'])
            item = {
                'database':database,
                'condition':str(tuple(one_condition)),
                'parameter':self.parameter,
            }

            read.insert_query_info(item, table, one_condition[-1], hash_str)

            cohort_data = get_cohort_init_upbound(
                clinical_long,
                self.g_patient_bound,
                one_condition[:len(one_condition)-num_same_exposure]
                )

            if num_same_exposure > 0:
                cohort_data_list = []
                for _, df_patient in cohort_data.groupby('patient_id'):
                    same_moa_patient = reset_init_date(df_patient, num_same_exposure)
                    if same_moa_patient is not None:
                        cohort_data_list.append(same_moa_patient)
                cohort_data = pd.concat(cohort_data_list)
            all_cohorts[tuple(one_condition)] = cohort_data
            
            try:
                read.dump_query_results(cohort_data, hash_str, 'complete')
            except Exception as e:
                err = ["failed job", "type error: " + str(e), traceback.format_exc()]
                read.dump_query_results(cohort_data, hash_str, 'failed', err)
                # data = 'none'
                raise Exception('Error in making the Query', "type error: " + str(e), traceback.format_exc())
        
        self.all_cohorts.update(all_cohorts)
        return all_cohorts


    def get_cohort_features(self, condition, feature_list=None):

        table = self.table
        database = self.database
        feature_list = self.feature_list
        cohort_data = self.all_cohorts[condition]

        if len(cohort_data) == 0:
            return None
        
        # cohort_data = aggregate_multisource(
        #     cohort_data,
        #     ['crp', 'pt_global_assess'],
        #     'float',
        #     replace_dict={"<3.6":'3.6'}
        #     )
        
        self.clinical_init = patch_cohort_init(cohort_data, self.kwarg['boundary'])
        
        df_clinical = self.clinical_init[
            self.clinical_init.parameter.isin(['specimen_name']+feature_list['clinical']) 
            ].set_index('patient_id').pivot(columns='parameter', values='value')

        if len(df_clinical) == 0:
            return None
        # NOTE: currently no specimen name is causing the pipeline to break
        s_specimen = df_clinical['specimen_name']
        self.s_specimen = s_specimen

        parent_hash_str = self._create_hash(table, database, str(tuple(condition)))

        all_genes = []
        for feature in feature_list['genes']:

            if isinstance(feature_list['genes'], list):
                metric_ = 'expected_count'
            else:
                metric_ = feature_list['genes'][feature]

            if isinstance(feature, tuple):
                features = feature
                feature = self.kwarg['tuple_feature_name']
            else:
                ft_path = PARENT_DIR + 'parameters/features.csv'
                features = pd.read_csv(ft_path)[feature].dropna()

            feature_hash = hashlib.md5((feature + '_' + metric_).encode()).hexdigest()
            hash_str = parent_hash_str + '-' + feature_hash            
            if 'tag' in self.kwarg:
                hash_str += ('-' + self.kwarg['tag'])
            not_exist, query_info = read.check_query_info(hash_str)
            destination = 'genes_pl_test_' + hash_str

            if self.debug_mode:
                not_exist = True
            if not not_exist:
                if query_info['status'][0] == 'complete':
                    print(f'found {str(tuple(condition))}-{feature}_{metric_} glue job!!!')
                    print(hash_str)
                    df_genes = wr.s3.read_parquet(
                                path="s3://patient-dl-processed-clinical/ml-outputs/tm-temp/"
                                    + destination
                                    + "/"
                            )           
            else:
                print(f"didn't find {str(tuple(condition))}-{feature}_{metric_}glue job...")
                print(hash_str)
                item = {
                    'database':database,
                    'condition':str(tuple(condition)),
                    'parameter':self.parameter,
                    'feature':feature+'_'+metric_,
                } 

                read.insert_query_info(item, table, condition[-1], hash_str)

                features= ','.join(features)
                #TEMP: for Messy Data
                if self.temp:
                    gene_table = 'genes_production'
                else:
                    if self.kwarg['gene_table'] == False:
                        gene_table = 'gene_result_long_format'
                    else:
                        gene_table = self.kwarg['gene_table']

                specimens = ','.join(s_specimen.dropna())
                random_delay=10
                do_pivot(gene_table, specimens, features, metric_, destination, random_delay)

                df_genes = wr.s3.read_parquet(
                            path="s3://patient-dl-processed-clinical/ml-outputs/tm-temp/"
                                + destination
                                + "/"
                        )
                
                try:
                    read.dump_query_results(cohort_data, hash_str, 'complete')
                except Exception as e:
                    err = ["failed job", "type error: " + str(e), traceback.format_exc()]
                    read.dump_query_results(cohort_data, hash_str, 'failed', err)
                    # clinical_long = 'none'
                    raise Exception('Error in making the Query', "type error: " + str(e), traceback.format_exc())
            specimen_diff = set(df_genes['specimen_name']) ^ set(s_specimen)
            if len(specimen_diff) > 0:
                print(specimen_diff)
            all_genes.append(df_genes.drop_duplicates(subset=['specimen_name']).set_index('specimen_name'))
            #TODO: change all_genes to df_all_genes with null dataframe and specimen_name index as the initial value. 
            # df_all_genes = pd.DataFrame(index=['a','b']).join(
            # pd.DataFrame([1,2], index=['a','b'], columns=['1'])
            # ).join(
            # pd.DataFrame([2,3], index=['a','b'], columns=['2'])
            # )
        df_all_genes = pd.concat(all_genes, axis=1)
        df_gene_features = pd.merge(
            df_clinical.reset_index(),
            df_all_genes, 
            on='specimen_name',
            how='left'
            ).drop(columns=['data_date', 'facility']).set_index('patient_id')
    
        return df_gene_features


    def get_endpoints(self, condition, endpoint_type=None, month=None):

        cohort_data = self.all_cohorts[condition]
        # cohort_data = aggregate_multisource(
        #     cohort_data,
        #     ['crp', 'pt_global_assess'],
        #     'float',
        #     replace_dict={"<3.6":'3.6'}
        #     )
        endpoint_type = self.endpoint_type
        month = self.month
        clinical_init = self.clinical_init

        clinical_target = patch_cohort_target(cohort_data, month, self.kwarg['boundary_window'])
        self.clinical_init = clinical_init
        self.clinical_target = clinical_target
        self.clinical = pd.concat([clinical_init.drop(columns=['data_date_og']), clinical_target.drop(columns=['data_date_og'])])
        self.clinical.replace({'ND':np.nan, 'nd':np.nan}, inplace=True)
        if 'data_source' in self.clinical:
            clinical = self.clinical.drop(columns=['data_source'])
        else:
            clinical = self.clinical

        endpoints = mf.master_function(clinical, 0) #mf.runall(self.clinical, 0, 1)
        self.endpoints_whole = endpoints
        s_endpoint = endpoints[
            endpoints.visit == month
            ].set_index('patient_id').pivot(columns='parameter', values='value')[endpoint_type]
        
        self.s_endpoint = s_endpoint
        return s_endpoint


    def run_query(self):

        self._recall_cohort()
        if self.new_conditions:

            self.generate_cohorts()

        self.all_feature_data = {}
        self.all_clinical_data = {}
        self.all_specimen_data = {}
        self.all_genes_data = {}
        self.all_endpoints_whole = {}
        self.all_clinical_init = {}
        self.all_clinical_target = {}

        for cond in self.all_cohorts:
            df_gene_features = self.get_cohort_features(cond, self.feature_list)
            if df_gene_features is None:
                df_cohort_feature_data = []
            else:
                s_endpoint = self.get_endpoints(cond, self.endpoint_type, self.month)
                self.all_endpoints_whole[cond] = self.endpoints_whole
                df_cohort_feature_data = df_gene_features.join(s_endpoint)
                self.all_clinical_data[cond] = self.clinical
                self.all_clinical_init[cond] = self.clinical_init
                self.all_clinical_target[cond] = self.clinical_target
                self.all_specimen_data[cond] = self.s_specimen
                self.all_genes_data[cond] = df_gene_features.drop(columns=['specimen_name'])
            self.all_feature_data[cond] = df_cohort_feature_data

        return self.all_feature_data


class QueryCriteria(CohortCurator):
    """
    A class used to perform multiple queries on a given dataset.

    Methods
    -------
    __init__(query)
        Initializes the object and passes the query to the parent class.
    """

    def __init__(self, query):
        """
        Initializes the QueryMultiple object.

        Parameters
        ----------
        query : dict
            The query parameters to pass to the parent class.
        """
        super().__init__(**query)
        super().run_query()


    def _apply_criteria(self, df, key, operand, value):

        if 'ex_' in key:
            key = key.split('ex_')[-1]

        df_parameter = df[df['parameter'] == key].reset_index().sort_values('data_date').groupby('patient_id').first()
        if df_parameter.empty:
            warning_no_param = f'no param {key} in table'
            warnings.warn(warning_no_param)
        if 'patient_id' not in df_parameter.columns:
            df_parameter.reset_index(inplace=True)
        
        if ((not isinstance(value, float)) and (not isinstance(value, int))):
            df_parameter["value"] = df_parameter["value"].str.lower()
            val_dict = dict(zip(df_parameter["value"].unique(), range(len(df_parameter["value"].unique()))))

            if isinstance(value, list):
                value = [val_dict[val] for val in value if val in val_dict]
            elif value in val_dict:
                value = val_dict[value]

            df_parameter["value"] = df_parameter["value"].replace(val_dict)
        elif np.isnan(value):
            
            return (set(df_parameter[df_parameter["value"].isna()].patient_id) | 
                    (set(self.all_feature_data[tuple(self.conditions_list)].dropna().index) - set(df_parameter.patient_id))
                    )
        
        if operand == 'eq':
            return set(df_parameter[df_parameter["value"].astype('float64') == value].patient_id)
        elif operand == 'ge':
            return set(df_parameter[df_parameter["value"].astype('float64') >= value].patient_id)
        elif operand == 'g':
            return set(df_parameter[df_parameter["value"].astype('float64') > value].patient_id)
        elif operand == 'in':
            return set(df_parameter[df_parameter["value"].astype('float64').isin(value)].patient_id)
        
    def _collect_criteria_ids(self, criteria, condition, op_collection='and'):
        qualified_id = set()
        dict_id = {}

        for key in reorder_strings(list(criteria.keys())):

            if key in self.all_endpoints_whole[condition].parameter.unique():
                df = self.all_endpoints_whole[condition]
            else:
                df = self.all_cohorts[condition]

            if 'ex' in key:
                op_collection = 'sub'

            if 'operand' not in criteria[key].keys():
                dict_id[key] = self._collect_criteria_ids(criteria[key], condition, op_collection='or')
            else:
                criterion = criteria[key]
                # Include and exclude patients using id directly
                if 'id' in key:
                    dict_id[key] = set(criterion['value'])
                else:
                    dict_id[key] = self._apply_criteria(
                        df,
                        key,
                        criterion['operand'],
                        criterion['value']
                        )
            
            if op_collection == 'and':
                if len(qualified_id) == 0:
                    qualified_id = dict_id[key]
                else:
                    qualified_id &= dict_id[key]
            elif op_collection == 'or':
                qualified_id |= dict_id[key]
            elif op_collection == 'sub':
                qualified_id -= dict_id[key]

        self.dict_id = dict_id
        return qualified_id

    def _filter_df(self, df, criteria_ids):
        
        if df.index.name == 'patient_id':
            mask = df.index.isin(criteria_ids)
        else:
            mask = df['patient_id'].isin(criteria_ids)

        return df[mask]

    
    def filter_patients(self, criteria):

        if not criteria:
            print('criteria is empty')
        else:
            df_dict_list = [
                self.all_feature_data, self.all_clinical_data,
                self.all_specimen_data, self.all_genes_data,
                self.all_endpoints_whole, self.all_clinical_init,
                self.all_clinical_target, self.all_cohorts
                ]
            for key in df_dict_list[-1]:
                qualified_id = self._collect_criteria_ids(criteria, key)
                for df_dict in df_dict_list:
                    df_dict[key] = self._filter_df(df_dict[key], qualified_id)

        return
        

def get_row_ranges(s):
    # Group by categories and get the first and last index for each one
    ranges = s.groupby(s).apply(lambda x: (x.index.min(), x.index.max()))

    # Create a new dataframe with the categories and their corresponding ranges
    result = pd.DataFrame([(index, row[0], row[1]) for index, row in ranges.items()], 
                          columns=['Category', 'Start', 'End'])

    return result