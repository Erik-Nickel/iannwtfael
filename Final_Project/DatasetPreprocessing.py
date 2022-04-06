import gc
import tensorflow as tf
import pandas as pd
import ast
from tensorflow.keras.layers import CategoryEncoding
import random


class DatasetPreprocessing:
    __TRAIN_DATA_PATH = 'data_train.csv'
    __VAL_DATA_PATH = 'data_val.csv'

    def __init__(self, seq_len) -> None:
        self.seq_len = seq_len
        self.ing_encoding = CategoryEncoding(num_tokens=8023, output_mode="multi_hot")

    def norm(self, x):
        return x.apply(lambda i: (i - x.min()) / (x.max() - x.min()))

    def user_split(self, df, val, var='user_id'):
        num_inter = df[var].value_counts(sort=False).to_numpy()
        n = 0
        for i in num_inter:
            if n < val * len(df):
                n += i
            else:
                return n

    def preprocesses(self, dataset1='RAW_interactions.csv', dataset2='RAW_recipes.csv', dataset3='PP_recipes.csv',
                     train_size=0.70, dataset_chunk=1):
        self.train_size = train_size
        self.dataset_chunk = dataset_chunk
        inter_raw = pd.read_csv(dataset1)
        recipes_raw = pd.read_csv(dataset2)
        recipes_pp = pd.read_csv(dataset3)

        recipes_pp['recipe_id'] = recipes_pp.pop('id')
        recipes_raw['recipe_id'] = recipes_raw.pop('id')

        omni_raw = pd.merge(recipes_raw, inter_raw, on='recipe_id')

        # omni_raw['minutes'] = self.norm(omni_raw['minutes'])
        # omni_raw['n_steps'] = self.norm(omni_raw['n_steps'])
        # omni_raw['n_ingredients'] = self.norm(omni_raw['n_ingredients'])

        omni_raw['recipe_features'] = omni_raw[['minutes', 'n_steps', 'n_ingredients']].values.tolist()

        omni_raw = omni_raw.drop(
            ['minutes', 'n_steps', 'n_ingredients', 'name', 'contributor_id', 'submitted', 'tags', 'nutrition', 'steps',
             'description', 'review', 'ingredients'], axis=1)
        omni_raw = pd.merge(omni_raw, recipes_pp.drop(
            ['name_tokens', 'steps_tokens', 'techniques', 'calorie_level', 'ingredient_tokens', ], axis=1),
                            on='recipe_id')
        omni_raw["recipe_id"] = omni_raw['i']
        omni_raw.pop('i')
        # print(omni_raw.describe())

        counts = omni_raw.value_counts('user_id')
        omni_raw = omni_raw[omni_raw.rating >= 4].drop(['rating'], axis=1)
        omni_raw = omni_raw.loc[omni_raw['user_id'].isin(counts.index[counts >= 10])]

        omni_raw = omni_raw.sort_values(['user_id', 'date']).reset_index()
        omni_raw.pop('date')

        ids = omni_raw["user_id"].unique()
        random.shuffle(ids)
        omni_raw = omni_raw.set_index("user_id").loc[ids].reset_index()
        omni_raw.pop('index')

        omni_raw = omni_raw.iloc[:self.user_split(omni_raw, self.dataset_chunk)]
        # print(omni_raw)

        n = self.user_split(omni_raw, self.train_size)
        data_train, data_val = omni_raw.iloc[:n], omni_raw.iloc[n:]

        data_train.to_csv(self.__TRAIN_DATA_PATH)
        data_val.to_csv(self.__VAL_DATA_PATH)

        self.num_train = data_train['user_id'].value_counts(sort=False).to_numpy()
        self.num_val = data_val['user_id'].value_counts(sort=False).to_numpy()

        # self.num_inter = self.num_inter[:100]

        del omni_raw
        gc.collect()

    def gen_data(self, data_file_path, number_of_inter):
        skip_rows = 0
        read_rows = self.seq_len + 1
        n = 0
        while n < len(number_of_inter):
            m = 1
            while m < number_of_inter[n] - read_rows:
                data = pd.read_csv(data_file_path, skiprows=skip_rows + m, nrows=read_rows, header=None, quotechar='"',
                                   sep=',',
                                   converters={'recipe_features': ast.literal_eval, 'ing_ids': ast.literal_eval},
                                   names=['user_id', 'recipe_id', 'recipe_features', 'ing_ids']).reset_index()
                data['ing_ids'] = data['ing_ids'].apply(lambda x: self.ing_encoding(x))
                # data['recipe_features'] = data['recipe_features'].apply(lambda x: np.array(x))
                m += 1
                # if m == 2:
                # print((data.recipe_id.tolist()[1], data.ing_ids.tolist()[1], data.recipe_features.tolist()[1]),data.recipe_id.tolist()[-1])
                yield (tf.convert_to_tensor(data.recipe_id.tolist()[:-1]),
                       tf.convert_to_tensor(data.ing_ids.tolist()[:-1]),
                       tf.convert_to_tensor(data.recipe_features.tolist()[:-1])), tf.convert_to_tensor(
                    data.recipe_id.tolist()[-1])
            skip_rows += number_of_inter[n]
            n += 1

    def gen_data_train(self):
        return self.gen_data(self.__TRAIN_DATA_PATH, self.num_train)

    def gen_data_val(self):
        return self.gen_data(self.__VAL_DATA_PATH.self.num_val)
