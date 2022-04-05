import gc
import tensorflow as tf
import pandas as pd
import numpy as np
import ast
from tensorflow.keras.layers import CategoryEncoding


class DatasetPreprossesing():

    def __init__(self,seq_len) -> None:
        self.seq_len = seq_len

    def preprocessing(self, dataset1='RAW_interactions.csv', dataset2='RAW_recipes.csv', dataset3='PP_recipes.csv', train_size = 0.70):
        inter_raw = pd.read_csv(dataset1)
        recipes_raw = pd.read_csv(dataset2)
        recipies_pp = pd.read_csv(dataset3)

        recipies_pp['recipe_id'] = recipies_pp.pop('id')
        recipes_raw['recipe_id'] = recipes_raw.pop('id')

        omni_raw = pd.merge(recipes_raw, inter_raw, on='recipe_id')

        omni_raw['recipe_features'] = omni_raw[['minutes', 'n_steps', 'n_ingredients']].values.tolist()

        omni_raw = omni_raw.drop(
            ['minutes', 'n_steps', 'n_ingredients', 'name', 'contributor_id', 'submitted', 'tags', 'nutrition', 'steps',
             'description', 'review', 'ingredients'], axis=1)
        omni_raw = pd.merge(omni_raw, recipies_pp.drop(
            ['name_tokens', 'steps_tokens', 'techniques', 'calorie_level', 'ingredient_tokens', ], axis=1),
                            on='recipe_id')
        omni_raw["recipe_id"] = omni_raw['i']
        omni_raw.pop('i')
        # print(omni_raw.describe())

        counts = omni_raw.value_counts('user_id')
        omni_raw = omni_raw[omni_raw.rating >= 4].drop(['rating'], axis=1)
        omni_raw = omni_raw.loc[omni_raw['user_id'].isin(counts.index[counts >= 10 ])]

        omni_raw = omni_raw.sort_values(['user_id', 'date']).reset_index()
        omni_raw.pop('date')

        data_train, data_val = np.split(omni_raw.sample(frac=1, random_state=42), [int(train_size * len(omni_raw))])
        data_train.to_csv('data_train.csv')
        data_val.to_csv('data_val.csv')

        self.num_inter = omni_raw['user_id'].value_counts(sort = False).to_numpy()
        
        #self.num_inter = self.num_inter[:100]
        
        self.catEnc = CategoryEncoding(num_tokens=8023, output_mode="multi_hot")
       
        del omni_raw
        gc.collect()

    def gen_data_train(self):
        skipRows = 0
        readRows = self.seq_len
        n = 0

        while n < len(self.num_inter): 


            m = 1
            while m < self.num_inter[n] - readRows:
                data = pd.read_csv('data_train.csv', skiprows=skipRows + m, nrows=readRows, header=None, quotechar='"',
                                   sep=',',
                                   converters={'recipe_features': ast.literal_eval, 'ing_ids': ast.literal_eval},
                                   names=['recipe_id', 'user_id', 'recipe_features', 'ing_ids']).reset_index()
                data['ing_ids'] = data['ing_ids'].apply(
                    lambda x: self.catEnc(x))
                #data['recipe_features'] = data['recipe_features'].apply(lambda x: np.array(x))

                m += 1
                # if m == 2:
                #    print((data.recipe_id.tolist()[1], data.ing_ids.tolist()[1], data.recipe_features.tolist()[1]),data.recipe_id.tolist()[-1])
                yield (tf.convert_to_tensor(data.recipe_id.tolist()[:-1]),
                       tf.convert_to_tensor(data.ing_ids.tolist()[:-1]),
                       tf.convert_to_tensor(data.recipe_features.tolist()[:-1])),tf.convert_to_tensor(data.recipe_id.tolist()[-1])
            skipRows += self.num_inter[n]
            n += 1

    def gen_data_val(self):
        skipRows = 0
        readRows = self.seq_len
        n = 0

        while n < len(self.num_inter): 


            m = 1
            while m < self.num_inter[n] - readRows:
                data = pd.read_csv('data_val.csv', skiprows=skipRows + m, nrows=readRows, header=None, quotechar='"',
                                   sep=',',
                                   converters={'recipe_features': ast.literal_eval, 'ing_ids': ast.literal_eval},
                                   names=['recipe_id', 'user_id', 'recipe_features', 'ing_ids']).reset_index()
                data['ing_ids'] = data['ing_ids'].apply(
                    lambda x: self.catEnc(x))
                #data['recipe_features'] = data['recipe_features'].apply(lambda x: np.array(x))

                m += 1
                # if m == 2:
                #    print((data.recipe_id.tolist()[1], data.ing_ids.tolist()[1], data.recipe_features.tolist()[1]),data.recipe_id.tolist()[-1])
                yield (tf.convert_to_tensor(data.recipe_id.tolist()[:-1]),
                       tf.convert_to_tensor(data.ing_ids.tolist()[:-1]),
                       tf.convert_to_tensor(data.recipe_features.tolist()[:-1])),tf.convert_to_tensor(data.recipe_id.tolist()[-1])
            skipRows += self.num_inter[n]
            n += 1

  