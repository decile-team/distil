from distil.scalable_active_learning_strategies.strategy import Strategy
from torch.utils.data import Subset

class AVLTreeBuffer:
    
    def __init__(self, capacity):
        
        self.root = None
        self.capacity = capacity
        self.nodes = 0

    def print_tree(self):
        
        self.print_tree_(self.root)
        
    def print_tree_(self, compare_node):
        
        if compare_node.left_child is not None:
            self.print_tree_(compare_node.left_child)
        
        print(compare_node.height, compare_node.key, compare_node.value)
        
        if compare_node.right_child is not None:
            self.print_tree_(compare_node.right_child)

    def get_value_list(self):
        
        return self.get_value_list_(self.root)
        
    def get_value_list_(self, compare_node):
        
        if compare_node.left_child is None:
            return_list = []
        else:
            return_list = self.get_value_list_(compare_node.left_child)

        return_list.append(compare_node.value)
        
        if compare_node.right_child is None:
            return return_list
        else:
            return_list.extend(self.get_value_list_(compare_node.right_child))
            return return_list

    def insert(self, node):
        
        if self.root is None:
            self.root = node
            self.nodes += 1
            return
        
        self.root = self.insert_(self.root, node)
        self.nodes += 1

        if self.nodes > self.capacity:
            self.root = self.delete_least_(self.root)
            self.nodes -= 1
    
    def delete_least_(self, compare_node):
        
        if compare_node.left_child is not None:
            leftover_child = self.delete_least_(compare_node.left_child)
            compare_node.left_child = leftover_child
            
            left_child_height = compare_node.left_child.height if compare_node.left_child is not None else -1
            right_child_height = compare_node.right_child.height if compare_node.right_child is not None else -1
        
            if left_child_height - right_child_height > 1:
            
                left_left_grandchild_height = compare_node.left_child.left_child.height if compare_node.left_child.left_child is not None else -1
                left_right_grandchild_height = compare_node.left_child.right_child.height if compare_node.left_child.right_child is not None else -1
            
                if left_left_grandchild_height > left_right_grandchild_height:
                    return_node = self.left_left_rotation(compare_node)
                else:
                    return_node = self.left_right_rotation(compare_node)
            elif left_child_height - right_child_height < -1:
            
                right_left_grandchild_height = compare_node.right_child.left_child.height if compare_node.right_child.left_child is not None else -1
                right_right_grandchild_height = compare_node.right_child.right_child.height if compare_node.right_child.right_child is not None else -1
            
                if right_left_grandchild_height > right_right_grandchild_height:
                    return_node = self.right_left_rotation(compare_node)
                else:
                    return_node = self.right_right_rotation(compare_node)
            else:
            
                compare_node.fix_height()
                return_node = compare_node
                
            return return_node
        else:
            return compare_node.right_child    
    
    def insert_(self, compare_node, insert_node):
        
        compare_key = compare_node.key
        insert_key = insert_node.key
        
        if compare_key > insert_key:
            if compare_node.left_child is not None:
                compare_node.left_child = self.insert_(compare_node.left_child, insert_node)
            else:
                compare_node.left_child = insert_node
        else:
            if compare_node.right_child is not None:
                compare_node.right_child = self.insert_(compare_node.right_child, insert_node)
            else:
                compare_node.right_child = insert_node
        
        left_child_height = compare_node.left_child.height if compare_node.left_child is not None else -1
        right_child_height = compare_node.right_child.height if compare_node.right_child is not None else -1
        
        if left_child_height - right_child_height > 1:
            
            left_left_grandchild_height = compare_node.left_child.left_child.height if compare_node.left_child.left_child is not None else -1
            left_right_grandchild_height = compare_node.left_child.right_child.height if compare_node.left_child.right_child is not None else -1
            
            if left_left_grandchild_height > left_right_grandchild_height:
                return_node = self.left_left_rotation(compare_node)
            else:
                return_node = self.left_right_rotation(compare_node)
        elif left_child_height - right_child_height < -1:
            
            right_left_grandchild_height = compare_node.right_child.left_child.height if compare_node.right_child.left_child is not None else -1
            right_right_grandchild_height = compare_node.right_child.right_child.height if compare_node.right_child.right_child is not None else -1
            
            if right_left_grandchild_height > right_right_grandchild_height:
                return_node = self.right_left_rotation(compare_node)
            else:
                return_node = self.right_right_rotation(compare_node)
        else:
            
            compare_node.fix_height()
            return_node = compare_node
            
        return return_node
        
            
    def left_left_rotation(self, parent_node):
        
        old_parent = parent_node
        old_left_child = parent_node.left_child
        old_left_right_grandchild = old_left_child.right_child

        old_left_child.right_child = old_parent
        old_parent.left_child = old_left_right_grandchild
        
        # Fix heights
        old_parent.fix_height()
        old_left_child.fix_height()
        
        # Return new parent of this subtree
        return old_left_child
        
    def left_right_rotation(self, parent_node):
        
        old_parent = parent_node
        old_left_child = parent_node.left_child
        old_left_right_grandchild = old_left_child.right_child
        old_left_right_left_great_grandchild = old_left_right_grandchild.left_child
        old_left_right_right_great_grandchild = old_left_right_grandchild.right_child
        
        old_left_right_grandchild.left_child = old_left_child
        old_left_right_grandchild.right_child = old_parent
        old_left_child.right_child = old_left_right_left_great_grandchild
        old_parent.left_child = old_left_right_right_great_grandchild

        # Fix heights
        old_left_child.fix_height()
        old_parent.fix_height()
        old_left_right_grandchild.fix_height()

        # Return new parent of this subtree
        return old_left_right_grandchild

    def right_left_rotation(self, parent_node):
        
        old_parent = parent_node
        old_right_child = parent_node.right_child
        old_right_left_grandchild = old_right_child.left_child
        old_right_left_left_great_grandchild = old_right_left_grandchild.left_child
        old_right_left_right_great_grandchild = old_right_left_grandchild.right_child
        
        old_right_left_grandchild.left_child = old_parent
        old_right_left_grandchild.right_child = old_right_child
        old_parent.right_child = old_right_left_left_great_grandchild
        old_right_child.left_child = old_right_left_right_great_grandchild

        # Fix heights
        old_right_child.fix_height()
        old_parent.fix_height()
        old_right_left_grandchild.fix_height()

        # Return new parent of this subtree
        return old_right_left_grandchild
    
    def right_right_rotation(self, parent_node):
        
        old_parent = parent_node
        old_right_child = parent_node.right_child
        old_right_left_grandchild = old_right_child.left_child
        
        old_right_child.left_child = old_parent
        old_parent.right_child = old_right_left_grandchild

        # Fix heights
        old_parent.fix_height()
        old_right_child.fix_height()
        
        # Return new parent of this subtree
        return old_right_child
        
class AVLNode:
    
    def __init__(self, key, value):
        
        self.key = key
        self.value = value
        self.height = 0
        self.left_child = None
        self.right_child = None

    def fix_height(self):
        
        if self.left_child is None and self.right_child is None:
            self.height = 0
        elif self.left_child is None:
            self.height = self.right_child.height + 1
        elif self.right_child is None:
            self.height = self.left_child.height + 1
        else:
            self.height = max(self.left_child.height, self.right_child.height) + 1

class ScoreStreamingStrategy(Strategy):
    
    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):
        
        super(ScoreStreamingStrategy, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
        
        if 'stream_buffer_size' not in args:
            self.stream_buffer_size = 10000
        else:
            self.stream_buffer_size = args['stream_buffer_size']
        
    def acquire_scores(self, unlabeled_batch):
        pass
        
    def select(self, budget):
        
        # Go through the unlabeled data in a stream-like manner, holding only the top `budget` scores in memory at any given point.
        tree_buffer = AVLTreeBuffer(capacity = budget)

        evaluated_points = 0
        
        while evaluated_points < len(self.unlabeled_dataset):
            
            buffered_stream = Subset(self.unlabeled_dataset, list(range(evaluated_points, min(len(self.unlabeled_dataset), evaluated_points + self.stream_buffer_size))))
            batch_scores = self.acquire_scores(buffered_stream)
            
            for batch_score in batch_scores:
                tree_buffer.insert(AVLNode(batch_score, evaluated_points))
                evaluated_points += 1
        
        return tree_buffer.get_value_list()