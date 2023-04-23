import numpy as np
import hicstraw

fnames = ["GSM1551607_HIC058", "GSM1551608_HIC059", "GSM1551601_HIC052"]
for fname in fnames:
    hic = hicstraw.HiCFile(f"../data/{fname}.hic")
    # for chr in hic.getChromosomes():
    #     print(chr.name, chr.length)
    # print(hic.getGenomeID())
    # print(hic.getResolutions())

    mzd = hic.getMatrixZoomData('21', '21', "observed", "VC", "BP", 5000)
    numpy_matrix = mzd.getRecordsAsMatrix(10000000, 10200000, 10000000, 10200000)
    print(numpy_matrix.shape)
    # numpy_matrix = mzd.getRecordsAsMatrix(47929895, 48129895, 47929895, 48129895)
    # numpy_matrix = mzd.getRecordsAsMatrix(48129895, 48329895, 48129895, 48329895)
    # numpy_matrix = mzd.getRecordsAsMatrix(-1, 200000, -1, 200000)
    # numpy_matrix = mzd.getRecordsAsMatrix(2200000, 2400000, 2200000, 2400000)
    print(numpy_matrix.max())

    import matplotlib.pyplot as plt

    plt.imshow(numpy_matrix)

    # Save the plot to a PNG file
    plt.savefig(f'{fname}.png')
