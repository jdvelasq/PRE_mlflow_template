from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor


def select_model(args):
    if args.model == "elasticnet":
        return ElasticNet(alpha=args.alpha, l1_ratio=args.l1_ratio)
    elif args.model == "knn":
        return KNeighborsRegressor(n_neighbors=args.n_neighbors)
    else:
        raise ValueError("Invalid model type. Choose 'elasticnet' or 'knn'.")
