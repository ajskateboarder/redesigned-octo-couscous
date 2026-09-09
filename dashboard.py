import pandas as pd
from dct_attrib import StreamedTopK

def k_distribution(top_k: StreamedTopK, bottom_k: StreamedTopK):
    (t_keys, t_values), (b_keys, b_values) = top_k.get(), bottom_k.get()

    top_items = list(zip(t_keys.cpu().tolist(), t_values.cpu().tolist()))
    bottom_items = list(zip(b_keys.cpu().tolist(), b_values.cpu().tolist()))
    items = sorted(bottom_items + top_items, key=lambda item: item[1])

    labels = [" x ".join(map(str, combination)) for combination, _ in items]
    values = [value for _, value in items]

    df = pd.DataFrame({"circuit": labels, "attribution": values}).sort_values(by="attribution", ascending=False).reset_index(drop=True)
    return df