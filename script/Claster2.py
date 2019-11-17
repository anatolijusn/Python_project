
from IPython.display import Image
import plotnine as gg
import pandas as pd
import json
def compute_fk(k, sse, prev_sse, dim):
    if k == 1 or prev_sse == 0:
        return 1
    weight = weight_factor(k, dim)
    return sse / (weight * prev_sse)

# calculating alpha_k in functional style with tail recursion -- which is not optimized in Python :(
def weight_factor(k, dim):
    if not k > 1:
        raise ValueError("k must be greater than 1")
        
    def weigth_factor_accumulator(acc, k):
        if k == 2:
            return acc
        return weigth_factor_accumulator(acc + (1 - acc) / 6, k - 1)
        
    weight_k2 = 1 - 3 / (4 * dim)
    return weigth_factor_accumulator(weight_k2, k)
def compute_fk_from_k_sse_pairs(k_sse_pairs, dimension):
    triples = make_fk_triples(k_sse_pairs)
    k_fk_pairs = [
        (k, compute_fk(k, sse, prev_sse, dimension))
        for (k, sse, prev_sse) in triples]
    return sorted(k_fk_pairs, key=lambda pair: pair[0])


def make_fk_triples(k_sse_pairs):
    sorted_pairs = sorted(k_sse_pairs, reverse=True)
    candidates = list(zip(sorted_pairs, sorted_pairs[1:] + [(0, 0.0)]))
    triples = [
        (k, sse, prev_sse)
        for ((k, sse), (prev_k, prev_sse)) in candidates
        if k - prev_k == 1
    ]
    return triples

path_metrics_kmeans_sse=""
# cia keliai skaitomi is json failo, jau dirbame su parquet failu
json_file_path = "Params.json"
with open(json_file_path, 'r') as j:
     contents = json.load(j)
cluster=contents['cluster']
for item in cluster:
    path_metrics_kmeans_sse=item['path_metrics_kmeans_sse']
metrics_pddf = pd.read_json(
     path_metrics_kmeans_sse, 
    orient="records",
    lines=True)
k_sse_pddf = metrics_pddf[["k", "sse"]]
dimension = 62 
k_sse_pairs = [tuple(r) for r in k_sse_pddf.to_records(index=False)]
k_fk_pairs = compute_fk_from_k_sse_pairs(k_sse_pairs, dimension)
k_fk_pddf = pd.DataFrame.from_records(k_fk_pairs, columns=["k", "fk"])

plot_k_sse = (
    gg.ggplot(gg.aes(x="k", y="sse"), data=k_sse_pddf) + 
    gg.geom_line() + 
    gg.xlab("K") +
    gg.ylab("SSE") + 
    gg.ggtitle("SSE pagal klasterių skaičių K") +
    gg.theme_bw()
)

plot_k_fk = (
    gg.ggplot(gg.aes(x="k", y="fk"), data=k_fk_pddf) + 
    gg.geom_line() + 
    gg.xlab("K") +
    gg.ylab("f(K)") + 
    gg.ggtitle("f(K) pagal klasterių skaičių K") +
    gg.theme_bw()
)
print(plot_k_sse)
print(plot_k_fk)
plot_k_sse.save("k_sse.png")
plot_k_fk.save("k_fk.png")