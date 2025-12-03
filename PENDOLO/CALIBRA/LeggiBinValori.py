import numpy as np

def read_elab_file_numpy(filename):
    # Legge il numero di record
    with open(filename, "rb") as f:
        # Legge primo uint32
        num = int(np.fromfile(f, dtype=np.uint32, count=1)[0])
        print(num)
        # Legge tutto il resto come float32
        data = np.fromfile(f, dtype=np.float32)

    if data.size % 12 != 0:
        raise ValueError("File non allineato: numero di float non multiplo di 12")

    n = data.size // 12
    data = data.reshape(n, 12)

    # prime 9 colonne → covarianze
    cov = data[:, 0:9].reshape(n, 3, 3).astype(np.float64)

    # ultime 3 colonne → medie
    avg = data[:, 9:12].astype(np.float64)

    avg_mean = np.mean(np.linalg.norm(avg, axis=1))
    avg = avg/avg_mean
    cov=cov/(avg_mean*avg_mean)

    return cov, avg

cov_raw, avg_raw = read_elab_file_numpy("dati1.bin")

print(cov_raw.shape)   # (n,3,3)
print(avg_raw.shape)   # (n,3)
