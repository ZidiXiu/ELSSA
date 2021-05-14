import os

import numpy as np
import pandas

# from utils.preprocessing import one_hot_encoder, formatted_data, missing_proportion, \
#     one_hot_indices, get_train_median_mode, shape_train_valid


def formatted_data_missing(x, t, e, missing, sub_idx):
    death_time = np.array(t[sub_idx], dtype=float)
    censoring = np.array(e[sub_idx], dtype=float)
    covariates = np.array(x[sub_idx])
    mask = np.array(missing[sub_idx])

    print("observed fold:{}".format(sum(e[sub_idx]) / len(e[sub_idx])))
    survival_data = {'x': covariates, 't': death_time, 'e': censoring, 'missing_mask':mask}
    return survival_data


def generate_data(file_path='/data/zidi/cVAE/datasets/'):
    # this program is to extract the preset training/testing/validating dataset
    np.random.seed(31415)
    dir_path = os.path.dirname(file_path)
    

    path = os.path.abspath(os.path.join(dir_path, '', 'flchain.csv'))
    data_frame = pandas.read_csv(path, index_col=0)
    # remove rows with 0 time-to-event
    data_frame = data_frame[data_frame.futime != 0]
    data_frame['pat'] = np.arange(data_frame.shape[0])
    # x_data = data_frame[['age', 'sex', 'kappa', 'lambda', 'flc.grp', 'creatinine', 'mgus']]
    # Preprocess
    to_drop = ['futime', 'death', 'chapter', 'pat']
    dataset = data_frame.drop(labels=to_drop, axis=1)

    cat_var = ['sex', 'flc.grp', 'sample.yr', 'mgus']
    cat_type = dict(zip(cat_var, ['category']*len(cat_var)))
    cat_idx = np.where(np.isin(dataset.columns.values, np.array(cat_var)))[0]
    cts_var = np.setdiff1d(dataset.columns, cat_var)
    cts_idx = np.where(np.isin(dataset.columns.values, np.array(cts_var)))[0]

    dataset = dataset.astype(cat_type)
    # numerically encode the categorical variables
    dataset[cat_var] = dataset[cat_var].apply(lambda x: x.cat.codes)
    
    # split to train/valid/test before calculating imputation values
    # first shuffling all indices
    idx = np.arange(0, dataset.shape[0])

    np.random.seed(123)
    np.random.shuffle(idx)
    num_examples = int(0.80 * dataset.shape[0])
    print("num_examples:{}".format(num_examples))
    train_idx = idx[0: num_examples]
    split = int(( dataset.shape[0] - num_examples) / 2)
    test_idx = idx[num_examples: num_examples + split]
    valid_idx = idx[num_examples + split:  dataset.shape[0]]

    ####
    t_data = data_frame[['futime']]
    e_data = data_frame[['death']]
    pat_data = data_frame[['pat']]

    # positions where are missing indicated by 0
    missing_mask = 1-1*pandas.isna(dataset)
    missing_mask.head()

    # impute missing values
    continuous_median= dataset.median(axis=0).values
    categorical_mode = dataset.mode(axis=0).values[0][cat_idx]
    impute_dict = dict(zip(cat_var,categorical_mode))
    impute_dict.update(dict(zip(cts_var,continuous_median)))

    # fill back the imputed values
    dataset.fillna(impute_dict, inplace=True)


    # print("data description:{}".format(dataset.describe()))
    covariates = np.array(dataset.columns.values)
    # print("columns:{}".format(covariates))
    x = np.array(dataset).reshape(dataset.shape)
    t = np.array(t_data).reshape(len(t_data))
    e = np.array(e_data).reshape(len(e_data))

    missing = np.array(missing_mask).reshape(missing_mask.shape)

    # print("x:{}, t:{}, e:{}, len:{}".format(x[0], t[0], e[0], len(t)))

    print("x_shape:{}".format(x.shape))
    # here idx has been shuffled
    x = x[idx]
    missing = missing[idx]
    t = t[idx]
    e = e[idx]
    end_time = max(t)
    print("end_time:{}".format(end_time))
    print("observed percent:{}".format(sum(e) / len(e)))
    # print("shuffled x:{}, t:{}, e:{}, len:{}".format(x[0], t[0], e[0], len(t)))

    print("test:{}, valid:{}, train:{}, all: {}".format(len(test_idx), len(valid_idx), num_examples,
                                                        len(test_idx) + len(valid_idx) + num_examples))
    # print("test_idx:{}, valid_idx:{},train_idx:{} ".format(test_idx, valid_idx, train_idx))
    train = formatted_data_missing(x=x, t=t, e=e, missing=missing, sub_idx=train_idx)
    test = formatted_data_missing(x=x, t=t, e=e, missing=missing, sub_idx=test_idx)
    valid = formatted_data_missing(x=x, t=t, e=e, missing=missing, sub_idx=valid_idx)

    covariates = np.array([name.replace('.','_') for name in covariates])
    variable_info = {'cov_list':covariates, 'cts_var':covariates[cts_idx], 'cts_idx':cts_idx, 'cat_var':covariates[cat_idx], 'cat_idx':cat_idx }

    return train, valid, test, variable_info


if __name__ == '__main__':
    generate_data()