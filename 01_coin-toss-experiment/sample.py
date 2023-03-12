data_dict = ('isCorrect1', 'isCorrect2', 'isCorrect3')
allowed_values = (0, 1)
observed_data = [dict(zip(data_dict,[(a,b,c) for a in allowed_values for b in allowed_values for c in allowed_values][x]))  for x in range(len(allowed_values)**len(data_dict))]
for i in observed_data:
    print(i)
