import itertools

stuff = [1, 2, 3]
combinations = []
# saves all combinations of the elements to the combinations list
for L in range(1, len(stuff)+1):
    for subset in itertools.combinations(stuff, L):
        combinations.append(list(subset))

feature_list = [
    'MSSubClass',
    'LotArea',
    'OverallQual',
    'OverallCond',
    'YearBuilt',
    'YearRemodAdd',
    '1stFlrSF',
    '2ndFlrSF',
    'LowQualFinSF',
    'GrLivArea',
    'FullBath',
    'HalfBath',
    'BedroomAbvGr',
    'KitchenAbvGr',
    'TotRmsAbvGrd',
    'Fireplaces',
    'WoodDeckSF',
    'OpenPorchSF',
    'EnclosedPorch',
    '3SsnPorch',
    'ScreenPorch',
    'PoolArea',
    'MiscVal',
    'MoSold',
    'YrSold'
]
combinations = 0
# for L in range(1, len(feature_list)+1):
#    for subset in itertools.combinations(feature_list, L):
#        combinations += 1
print(combinations)

print(len(feature_list))

# combinations = 0
# cmbs = []
# feature_list = range(15)
# for L in range(0, len(feature_list)+1):
#     for subset in itertools.combinations(feature_list, L):
#         combinations += 1
#         cmbs.append(list(subset))
# for i, combination in enumerate(cmbs):
#     if i % 100 == 0:
#         print(f'finished {i} runs')
# print(combinations)
