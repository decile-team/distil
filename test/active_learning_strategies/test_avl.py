import random
import unittest

from distil.scalable_active_learning_strategies.score_streaming_strategy import AVLTreeBuffer, AVLNode

class TestAVLTree(unittest.TestCase):
    
    def assert_AVL_property(self, node):
        
        if node.left_child is None:
            left_child_height = -1
        else:
            left_child_height = self.assert_AVL_property(node.left_child)
            
        if node.right_child is None:
            right_child_height = -1
        else:
            right_child_height = self.assert_AVL_property(node.right_child)
            
        disparity = left_child_height - right_child_height
        disparity = -disparity if disparity < 0 else disparity
        
        # Assert that the difference in heights is no greater than 1
        self.assertLessEqual(disparity, 1)
        
        # Assert that this node's height is correctly calculated from the child nodes
        self.assertEqual(max(left_child_height, right_child_height) + 1, node.height)
        
        # Return this node's height
        return node.height
    
    def test_left_left_rotation(self):
        
        capacity = 20
        my_tree = AVLTreeBuffer(capacity)
        
        # Insert 3,2,1 in that order, causing a left-left rotation
        my_tree.insert(AVLNode(3,1))
        my_tree.insert(AVLNode(2,2))
        my_tree.insert(AVLNode(1,3))
        
        # Should be a triangle-shaped tree, root 2
        self.assertEqual(my_tree.root.height, 1)
        self.assertEqual(my_tree.root.key, 2)
        
        # Get value list; should be in order 3,2,1
        value_list = my_tree.get_value_list()
        self.assertEqual(value_list[0],3)
        self.assertEqual(value_list[1],2)
        self.assertEqual(value_list[2],1)
        
        # Assert that the recorded number of nodes is 3
        self.assertEqual(my_tree.nodes, 3)
        
        # Assert AVL property for this tree
        self.assert_AVL_property(my_tree.root)
        
    def test_left_right_rotation(self):
        
        capacity = 20
        my_tree = AVLTreeBuffer(capacity)
        
        # Insert 3,1,2 in that order, causing a left-right rotation
        my_tree.insert(AVLNode(3,1))
        my_tree.insert(AVLNode(1,2))
        my_tree.insert(AVLNode(2,3))
        
        # Should be a triangle-shaped tree, root 2
        self.assertEqual(my_tree.root.height, 1)
        self.assertEqual(my_tree.root.key, 2)
        
        # Get value list; should be in order 2,3,1
        value_list = my_tree.get_value_list()
        self.assertEqual(value_list[0],2)
        self.assertEqual(value_list[1],3)
        self.assertEqual(value_list[2],1)
        
        # Assert that the recorded number of nodes is 3
        self.assertEqual(my_tree.nodes, 3)
        
        # Assert AVL property for this tree
        self.assert_AVL_property(my_tree.root)
        
    def test_right_right_rotation(self):
        
        capacity = 20
        my_tree = AVLTreeBuffer(capacity)
        
        # Insert 1,2,3 in that order, causing a right-right rotation
        my_tree.insert(AVLNode(1,1))
        my_tree.insert(AVLNode(2,2))
        my_tree.insert(AVLNode(3,3))
        
        # Should be a triangle-shaped tree, root 2
        self.assertEqual(my_tree.root.height, 1)
        self.assertEqual(my_tree.root.key, 2)
        
        # Get value list; should be in order 1,2,3
        value_list = my_tree.get_value_list()
        self.assertEqual(value_list[0],1)
        self.assertEqual(value_list[1],2)
        self.assertEqual(value_list[2],3)
        
        # Assert that the recorded number of nodes is 3
        self.assertEqual(my_tree.nodes, 3)
        
        # Assert AVL property for this tree
        self.assert_AVL_property(my_tree.root)
        
    def test_right_left_rotation(self):
        
        capacity = 20
        my_tree = AVLTreeBuffer(capacity)
        
        # Insert 1,3,2 in that order, causing a right-left rotation
        my_tree.insert(AVLNode(1,1))
        my_tree.insert(AVLNode(3,2))
        my_tree.insert(AVLNode(2,3))
        
        # Should be a triangle-shaped tree, root 2
        self.assertEqual(my_tree.root.height, 1)
        self.assertEqual(my_tree.root.key, 2)
        
        # Get value list; should be in order 1,3,2
        value_list = my_tree.get_value_list()
        self.assertEqual(value_list[0],1)
        self.assertEqual(value_list[1],3)
        self.assertEqual(value_list[2],2)
        
        # Assert that the recorded number of nodes is 3
        self.assertEqual(my_tree.nodes, 3)
        
        # Assert AVL property for this tree
        self.assert_AVL_property(my_tree.root)
        
    def test_capacity(self):
        
        capacity = 5
        my_tree = AVLTreeBuffer(capacity)
        
        # Insert 1 through 10
        for i in range(10):
            my_tree.insert(AVLNode(i,i))
            
        # Assert that 5, 6, 7, 8, 9 are in value list
        self.assertEqual(set(my_tree.get_value_list()), set([5,6,7,8,9]))
        self.assertEqual(capacity, my_tree.nodes)
        
        # Assert height of root is 2 (has to be!)
        self.assertEqual(my_tree.root.height, 2)
        
        # Assert AVL property for this tree
        self.assert_AVL_property(my_tree.root)
        
    def test_stress(self):
        
        capacity = 1000
        my_tree = AVLTreeBuffer(capacity)
        
        # Generate a random permutation of numbers
        number_list = list(range(10000))
        random.shuffle(number_list)
        
        # Insert these into the tree, where key==value
        for i in number_list:
            my_tree.insert(AVLNode(i,i))
        
        # Assert that 9000,9001,...,9990 are in the list
        self.assertEqual(set(my_tree.get_value_list()),set([9000+x for x in range(1000)]))
        
        # Assert that the number of nodes in the tree is at capacity
        self.assertEqual(my_tree.nodes, capacity)
        
        # Assert that the value list is sorted (as key=value and an in-order traversal is done)
        last_number = -1
        for i in my_tree.get_value_list():
            self.assertLess(last_number,i)
            last_number = i
            
        # Assert AVL property for this tree
        self.assert_AVL_property(my_tree.root)