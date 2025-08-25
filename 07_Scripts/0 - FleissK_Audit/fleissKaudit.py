import pandas as pd, numpy as np

df = pd.read_csv("resultados_formatados.csv")

sounds  = sorted(df.sound_id.unique())
images  = sorted(df.image_id.unique())
n_rater = df.participant.nunique()

# Matriz itens × categorias
M = np.array([
    [(df[(df.image_id==img)&(df.sound_id==s)].shape[0]) for s in sounds]
    for img in images
])

# Passos 1‒4
P_i   = ((M**2).sum(1) - n_rater) / (n_rater*(n_rater-1))
Pbar  = P_i.mean()
p_k   = M.sum(0) / (len(images)*n_rater)
Pe    = (p_k**2).sum()
kappa = (Pbar - Pe) / (1 - Pe)
print("κ =", round(kappa,3))
