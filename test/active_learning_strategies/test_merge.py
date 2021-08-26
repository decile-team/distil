import random
import unittest

from distil.active_learning_strategies.score_streaming_strategy import merge

class TestMerge(unittest.TestCase):
    
    def test_merge_empty_list(self):
        
        list_1 = [10,9,5,2]
        list_2 = []
        key = lambda x: x
        max_cap = 1000
        merged_list = merge(list_1, list_2, key, max_cap)
        for item_1, item_2 in zip(list_1, merged_list):
            self.assertEqual(item_1, item_2)
    
        list_1 = []
        list_2 = [10,9,5,2]
        key = lambda x: x
        max_cap = 1000
        merged_list = merge(list_1, list_2, key, max_cap)
        for item_1, item_2 in zip(list_2, merged_list):
            self.assertEqual(item_1, item_2)
            
    def test_merge_key(self):
        
        list_1 = [10,9,5,2]
        list_2 = [8,6,5,1]
        list_1 = [(x,i) for i,x in enumerate(list_1)]
        list_2 = [(x,i) for i,x in enumerate(list_2)]
        
        key = lambda x: x[0]
        max_cap = 1000
        
        merged_list = merge(list_1, list_2, key, max_cap)
        
        self.assertEqual(merged_list[0][0], list_1[0][0])
        self.assertEqual(merged_list[1][0], list_1[1][0])
        self.assertEqual(merged_list[2][0], list_2[0][0])
        self.assertEqual(merged_list[3][0], list_2[1][0])
        self.assertEqual(merged_list[4][0], list_1[2][0])
        self.assertEqual(merged_list[5][0], list_2[2][0])
        self.assertEqual(merged_list[6][0], list_1[3][0])
        self.assertEqual(merged_list[7][0], list_2[3][0])
        
    def test_merge_max_cap(self):
        
        list_1 = [10,9,5,2]
        list_2 = [8,6,5,1]
        
        key = lambda x: x
        max_cap = 4
        
        merged_list = merge(list_1, list_2, key, max_cap)
        
        self.assertEqual(merged_list[0], list_1[0])
        self.assertEqual(merged_list[1], list_1[1])
        self.assertEqual(merged_list[2], list_2[0])
        self.assertEqual(merged_list[3], list_2[1])
        
    def test_rand(self):
        
        original_list = [x for x in range(1000,0,-1)]
        
        list_1 = []
        list_2 = []
        
        for item in original_list:
            if random.randint(0,1) == 0:
                list_1.append(item)
            else:
                list_2.append(item)
                
        key = lambda x: x
        max_cap = 10000
        
        merged_list = merge(list_1, list_2, key, max_cap)
        
        for item_1, item_2 in zip(original_list, merged_list):
            self.assertEqual(item_1, item_2)