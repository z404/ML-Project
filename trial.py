import copy
def combinations(target,data):
    lst = []
    for i in range(len(data)):
         new_target = copy.copy(target)
         new_data = copy.copy(data)
         new_target.append(data[i])
         new_data = data[i+1:]
         lst.append(new_target)
         lst.extend(combinations(new_target,new_data))
    return lst

