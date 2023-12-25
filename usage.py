import matplotlib.pyplot as plt
from aizynthfinder.aizynthfinder import AiZynthFinder

filename = "config.yml"
finder = AiZynthFinder(filename)

finder.stock.select("zinc")
finder.expansion_policy.select("uspto")
finder.filter_policy.select("uspto")

# Cc1cccc(c1N(CC(=O)Nc2ccc(cc2)c3ncon3)C(=O)C4CCS(=O)(=O)CC4)C
finder.target_smiles = "COc1cc(C=CC(=O)O)cc(OC)c1OC"
# print(psutil.Process().memory_info().rss / (1024 * 1024), "MB")
finder.tree_search()
finder.build_routes()

# stats = finder.extract_statistics()
# routes details
# for d in finder.routes.dicts:
#     print("---" * 20)
#     print(d)
breakpoint()
plt.imshow(finder.routes.images[0])
plt.show()

# stats
# print(stats)
