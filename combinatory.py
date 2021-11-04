import itertools

from matplotlib import pyplot as plt

stuff = [1, 2, 3]
combinations = []
# saves all combinations of the elements to the combinations list
for L in range(1, len(stuff) + 1):
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

a = [[17940.75684932, 18331.88356164, 17956.05719178, 16322.60541096,
      19514.73767123],
     [18018.30130137, 17937.21506849, 17779.31373288, 16081.26506849,
      19135.69041096],
     [17847.15949772, 17770.85388128, 17772.65922374, 16070.64442922,
      19234.12251142],
     [17783.18369863, 17826.84972603, 17581.68328767, 16070.93179795,
      19049.71748288],
     [17828.90143836, 17727.66623288, 17671.90356164, 16104.5759726,
      19128.19573973],
     [17908.03229452, 17674.41672374, 17667.58784247, 16150.88763699,
      19037.86054795],
     [17867.52219178, 17648.87040117, 17651.61484344, 16137.71398239,
      19137.64201566],
     [17888.47640411, 17596.12925514, 17673.76358733, 16131.33586473,
      19159.66452055]]
a = [sum(i)/len(i) for i in a]
b = list(range(4, 17, 2))
x = dict(zip(a, b))
plt.plot(b, a)
plt.show()


