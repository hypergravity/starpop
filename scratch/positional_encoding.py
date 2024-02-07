import numpy as np


def pe1(pos, dmodel=512, alternately=True):
    pe = np.zeros(dmodel, dtype=float)
    if alternately:
        for i in range(0, dmodel, 2):
            pe[i] = np.sin(pos / 10000 ** (i / dmodel))
            pe[i + 1] = np.cos(pos / 10000 ** (i / dmodel))
    else:
        half_dmodel = int(dmodel / 2)
        for i in range(0, half_dmodel, 1):
            pe[i] = np.sin(pos / 10000 ** (i / dmodel))
            pe[i + half_dmodel] = np.cos(pos / 10000 ** ((i) / dmodel))
    return pe


def pe(ntoken, dmodel=512, alternately=True):
    return np.asarray(
        [pe1(pos + 1, dmodel=dmodel, alternately=alternately) for pos in range(ntoken)]
    )


if __name__ == "__main__":
    # suppose a sentence has 10 token
    demo_pe = pe(ntoken=100, dmodel=512, alternately=True)
    import matplotlib.pyplot as plt

    fig = plt.figure(0, figsize=(15, 5))
    plt.imshow(demo_pe, origin="upper", cmap=plt.cm.jet)

    from sklearn.metrics.pairwise import cosine_similarity

    cosine_similarity([demo_pe[1]], [demo_pe[22]])
