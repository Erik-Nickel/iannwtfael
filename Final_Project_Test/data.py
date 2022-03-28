#!/usr/bin/env python

import pandas as pd
class DatasetPreprossesing():


    def __init__(self) -> None:
        pass
    
    def preprocessing(dataset1 = '../../data/RAW_interactions.csv',dataset2 = '../../data/RAW_recipes.csv', dataset3 = '../../data/PP_recipes.csv'):




        inter_raw = pd.read_csv(dataset1)
        recipes_raw = pd.read_csv(dataset2)
        recipies_pp = pd.read_csv(dataset3)


        recipies_pp['recipe_id'] = recipies_pp.pop('id')
        recipes_raw['recipe_id'] = recipes_raw.pop('id')
    
        omni_raw = pd.merge(recipes_raw,inter_raw,on='recipe_id')

        omni_raw['recipe_features'] = omni_raw[['minutes','n_steps','n_ingredients']].values.tolist()

        omni_raw = omni_raw.drop(['minutes','n_steps','n_ingredients', 'name', 'contributor_id', 'submitted', 'tags', 'nutrition', 'steps', 'description', 'review', 'ingredients'], axis=1)
        omni_raw = pd.merge(omni_raw,recipies_pp.drop(['i','name_tokens','steps_tokens','techniques','calorie_level', 'ingredient_tokens',],axis = 1),on='recipe_id')
        omni_raw = omni_raw[omni_raw.rating >= 4].drop(['rating'], axis = 1)
        omni_raw = omni_raw.loc[omni_raw['user_id'].isin(counts.index[counts >= 6])]

        omni_raw = omni_raw.sort_values(['user_id','date']).reset_index()

        omni_raw.to_csv('data_pp.csv')
        self.num_inter = omni_raw.value_counts('user_id').to_numpy()[1]
        print(self.num_inter)
    

new = DatasetPreprossesing()
new.preprocessing('../../data/RAW_interactions.csv','../../data/RAW_recipes.csv','../../data/PP_recipes.csv')