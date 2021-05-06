import math
import random
import torch

def calculate_class_budgets(budget, num_classes, trn_lbls, N_trn):
    
    """
    Calculates a list of class budgets whose sum is that of the specified budget.
    Furthermore, each budget calculated for a class is based on the proportion 
    of labels of that class that appear in trn_lbls. For example, if trn_lbls has 
    50% "0" labels, then the budget calculated for class "0" will be 50% of the full 
    budget.
    
    Specifically, this function makes sure to at least give every class a budget of 1.
    If this violates the full budget (the sum of the per-class budgets is greater than 
    the full budget), then random class budgets are set to 0 until the budget constraint 
    is satisfied.
    
    If the budget constraint is not broken, then it awards the rest of the full budget 
    in the proportion described above in a best-attempt manner.
    
    Parameters
    ----------
    budget: int
        Full budget to split into class budgets
    num_classes: int
        Number of per-class budgets to calculate
    trn_lbls: Torch tensor
        Label tensor on which to base per-class budgets 
    N_trn: int
        Number of labels in trn_lbls
    """
    
    # Tabulate class populations
    class_pops = list()
    for i in range(num_classes):
        trn_subset_idx = torch.where(trn_lbls == i)[0].tolist()
        class_pops.append(len(trn_subset_idx))

    # Assign initial class budgets, where each class gets at least 1 element (if there are any elements to choose)
    class_budgets = list()
    class_not_zero = list()
    for i in range(num_classes):
        if class_pops[i] > 0:
            class_budgets.append(1)
            class_not_zero.append(i)
        else:
            class_budgets.append(0)

    # Check if we have violated the budget. If so, pick random indices to set to 0.
    current_class_budget_total = 0
    for class_budget in class_budgets:
        current_class_budget_total = current_class_budget_total + class_budget

    if current_class_budget_total > budget:
        set_zero_indices = random.sample(class_not_zero, current_class_budget_total - budget)
			
        for i in set_zero_indices:
            class_budgets[i] = 0
		
    # We can proceed to adjust class budgets if we have not met the budget.
    # Note: if these two quantities are equal, we do not need to do anything.
    elif current_class_budget_total < budget:		
        # Calculate the remaining budget
        remaining_budget = budget - current_class_budget_total

        # Calculate fractions
        floored_class_budgets = list()
        for i in range(num_classes):
            # Fraction is computed off remaining budget to add. Class population is adjusted to remove freebee element (if present).
            # Total elements in consideration needs to remove already guaranteed elements (current_class_budget_total).
            # Add previous freebee element to remaining fractions
            class_budget = class_budgets[i] + remaining_budget * (class_pops[i] - class_budgets[i]) / (N_trn - current_class_budget_total)
            floored_class_budgets.append((i, math.floor(class_budget), class_budget - math.floor(class_budget)))

        # Sort the budgets to partition remaining budget in descending order of floor error.
        list.sort(floored_class_budgets, key=lambda x: x[2], reverse=True)

        # Calculate floored budget sum
        floored_sum = 0
        for _, floored_class_budget, _ in floored_class_budgets:
            floored_sum = floored_sum + floored_class_budget

        # Calculate new remaining total budget
        remaining_budget = budget - floored_sum
        index_iter = 0

        while remaining_budget > 0:
            class_index, class_budget, class_floor_err = floored_class_budgets[index_iter]
            class_budget = class_budget + 1
            floored_class_budgets[index_iter] = (class_index, class_budget, class_floor_err)

            index_iter = index_iter + 1
            remaining_budget = remaining_budget - 1	

        # Rearrange budgets to be sorted by class
        list.sort(floored_class_budgets, key=lambda x: x[0])

        # Override class budgets list with new values
        for i in range(num_classes):
            class_budgets[i] = floored_class_budgets[i][1]
                
    return class_budgets