import hashlib

import pandas as pd
import scipy.sparse as sp

from utility.Tool import df_to_positive_dict, save_dict_to_file

KEEP_CONTEXT = {
    'yelp-nc': ['c_city', 'c_month', 'c_hour', 'c_DoW', 'c_last'],
    'yelp-oh': ['c_city', 'c_month', 'c_hour', 'c_DoW', 'c_last'],
    'amazon-book': ['c_year', 'c_month', 'c_day', 'c_DOW', 'c_last']
}


class GivenData(object):
    def __init__(self, dataset_name, path, data_format, separator, logger):
        self.dataset_name = dataset_name
        self.path = path
        self.data_format = data_format
        self.separator = separator
        self.logger = logger

    def load_data(self):
        side_info, all_data_dict = None, None
        self.logger.info("Loading interaction records from folder: %s "% (self.path))

        train_data = pd.read_csv(self.path + "train.dat", sep=self.separator[0])
        test_data = pd.read_csv(self.path + "test.dat", sep=self.separator[0])

        
        # TAKE ONLY SUB-SET OF THE DATA FOR TRAINING: TODO UNDO THE CHANGES FOR FINAL ALL DATASET TRAINING
        all_train_test = pd.concat([train_data, test_data])
        # train_data = train_data[train_data["user_id"] < 6000]
        train_data["item_id"] = sorted(train_data["item_id"].values.tolist())
        train_data = train_data[train_data["item_id"] < 1000]
        test_data = test_data[test_data["item_id"].isin(train_data["item_id"].values.tolist())]
        test_data = test_data[test_data["user_id"].isin(train_data["user_id"].values.tolist())]


        all_data = pd.concat([train_data, test_data])

        #num_users = len(train_data["user_id"])
        num_users = all_data["user_id"].max() + 1
        num_items = len(train_data["item_id"])
        num_valid_items = all_data["item_id"].max() + 1

        num_train = len(train_data["user_id"])
        num_test = len(test_data["user_id"])
        
        train_matrix = sp.csr_matrix(([1] * num_train, (train_data["user_id"], train_data["item_id"])), shape=(num_users, num_valid_items))
        test_matrix = sp.csr_matrix(([1] * num_test, (test_data["user_id"], test_data["item_id"])),  shape=(num_users, num_valid_items))
        
        if self.data_format == 'UIC':
            side_info, side_info_stats, all_data_dict = {}, {}, {}
            column = all_data.columns.values.tolist()
            context_column = column[2].split(self.separator[1])
            user_feature_column = column[3].split(self.separator[1]) if 'yelp' in self.dataset_name.lower() else None
            item_feature_column = column[-1].split(self.separator[1])

            keep_context = KEEP_CONTEXT[self.dataset_name.lower()]
            new_context_column = '-'.join(keep_context)
            all_data[context_column] = all_data[all_data.columns[2]].str.split(self.separator[1], expand=True)
            all_data[new_context_column] = all_data[keep_context].apply('-'.join, axis=1)


            # map context to id using hashlib
            def hash_the_context(row):
                row = row.split("-")
                row = "-".join(sorted(row))
                return int(hashlib.sha256(row.encode()).hexdigest(), 16) % (10 ** 4)
            all_data["context_id"] = all_data[new_context_column].apply(hash_the_context)

            train_data = all_data.iloc[:num_train, :]
            test_data = all_data.iloc[num_train:, :]

            if user_feature_column:
                user_feature = all_data.drop_duplicates(["user_id", '-'.join(user_feature_column)])
                user_feature = user_feature[["user_id", '-'.join(user_feature_column)]]
                user_feature[user_feature_column] = user_feature[user_feature.columns[-1]].str.split(self.separator[1], expand=True)
                user_feature.drop(user_feature.columns[[1]], axis=1, inplace=True)
            else:
                user_feature = None

            item_feature = all_train_test.drop_duplicates(["item_id", '-'.join(item_feature_column)]) # TODO UNDO THE CHANGES FOR FINAL ALL DATASET TRAINING
            item_feature = item_feature[item_feature["item_id"].isin(all_data["item_id"].values.tolist())] # TODO UNDO THE CHANGES FOR FINAL ALL DATASET TRAINING


            item_feature = item_feature[["item_id", '-'.join(item_feature_column)]]
            item_feature[item_feature_column] = item_feature[item_feature.columns[-1]].str.split(self.separator[1], expand=True)
            item_feature.drop(item_feature.columns[[1]], axis=1, inplace=True)
            context_feature = all_data.drop_duplicates(["context_id", new_context_column])[["context_id", new_context_column]]
            context_feature[keep_context] = context_feature[context_feature.columns[-1]].str.split(self.separator[1], expand=True)
            context_feature.drop(context_feature.columns[[1]], axis=1, inplace=True)
            if user_feature_column:
                side_info['user_feature'] = user_feature.set_index('user_id').astype(int)
                side_info_stats['num_user_features'] = side_info['user_feature'][user_feature_column[-1]].max() + 1
                side_info_stats['num_user_fields'] = len(user_feature_column)
            else:
                side_info['user_feature'] = None
                side_info_stats['num_user_features'] = 0
                side_info_stats['num_user_fields'] = 0
            
            side_info['item_feature'] = item_feature.set_index('item_id').astype(int)
            side_info['context_feature'] = context_feature.set_index('context_id').astype(int)
            side_info_stats['num_item_features'] = side_info['item_feature'][item_feature_column[-1]].max() + 1
            side_info_stats['num_item_fields'] = len(item_feature_column)
            side_info_stats['num_context_features'] = side_info['context_feature'][keep_context[-2]].max() + 1 + num_items
            side_info_stats['num_context_fields'] = len(keep_context)
            self.logger.info("\n" + "\n".join(["{}={}".format(key, value) for key, value in side_info_stats.items()]))
            self.logger.info("context feature name: " + ",".join([f.replace('c_', '') for f in keep_context]))
            all_data_dict['train_data'] = train_data[['user_id', 'item_id', 'context_id']]
            all_data_dict['test_data'] = test_data[['user_id', 'item_id', 'context_id']]

            all_data_dict['positive_dict'] = df_to_positive_dict(all_data_dict['train_data'])
            save_dict_to_file(all_data_dict['positive_dict'], self.path + '/user_pos_dict.txt')
            side_info['side_info_stats'] = side_info_stats
        
        num_ratings = len(train_data["user_id"]) + len(test_data["user_id"])
        self.logger.info("\"num_users\": %d,\"num_items\":%d,\"num_valid_items\":%d, \"num_ratings\":%d"%(num_users, num_items, num_valid_items, num_ratings))

        if side_info['user_feature'] is not None:
            # Create a template of all users we expect
            full_user_range = pd.Index(range(num_users), name='user_id')
            # Reindex the features: users with no features will get NaN
            side_info['user_feature'] = side_info['user_feature'].reindex(full_user_range)
            # Fill NaN with 0 (or a specific padding index used in your embeddings)
            side_info['user_feature'] = side_info['user_feature'].fillna(0).astype(int)

        return train_matrix, test_matrix, all_data_dict, side_info, num_items

class Dataset(object):
    def __init__(self, conf, logger):
        """
        Constructor
        """
        self.logger = logger
        self.separator = conf.data_separator

        self.dataset_name = conf.dataset
        self.dataset_folder = conf.data_path
        
        data_splitter = GivenData(self.dataset_name, self.dataset_folder, conf.data_format, self.separator, self.logger)
        
        self.train_matrix, self.test_matrix, self.all_data_dict, self.side_info, self.num_items = data_splitter.load_data()
        # self.test_context_list = self.all_data_dict['test_data']['context_id'].tolist() if self.side_info is not None else None
        if self.side_info is None:
            self.test_context_dict = None
        else:
            self.test_context_dict = {}
            for user, context in zip(self.all_data_dict['test_data']['user_id'].tolist(), self.all_data_dict['test_data']['context_id'].tolist()):
                self.test_context_dict[user] = context

        self.num_users, self.num_valid_items = self.train_matrix.shape
        if self.side_info is not None:
            self.num_user_features = self.side_info['side_info_stats']['num_user_features']
            self.num_item_featuers = self.side_info['side_info_stats']['num_item_features']
            self.num_context_features = self.side_info['side_info_stats']['num_context_features']
        self.logger.info('Data Loading is Done!')