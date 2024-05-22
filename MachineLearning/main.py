import ML
import ML_PCA
import ML_gaussianNB


def main():
    ML.run_ml()
    ML_gaussianNB.run_gaussian_nb()
    ML_PCA.run_pca()

if __name__ == "__main__":
    main()
