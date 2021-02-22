db_feature = 456
features = []
features.append(([1, 2, 3 ,4], 123))
features.append(([4, 5, 6 ,7], 456))
features.append(([7, 8, 9 ,0], 789))
best_match = None
print(len(features))
for i,(bbox,feature) in enumerate(features):
    if feature == db_feature:
        print("found match: ", feature)
        best_match = i
(a, b) = features[best_match]
print(a)