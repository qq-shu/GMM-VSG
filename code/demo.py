# -*- encoding=utf-8 -*-
import argparse

import GmmVSG as Vsg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--covariance_type", default="full", type=str, help="covariance type")
    parser.add_argument("--optimizer", default="aic", type=str, help="method of how to select model")
    parser.add_argument("--result_retain", default=2, type=int, help="Keep decimal places for results")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--file_path", default="../data/23items.csv", type=str)
    parser.add_argument("--feature_start_idx", default=2, type=int)
    parser.add_argument("--feature_end_idx", default=7, type=int)
    parser.add_argument("--n_components", default=[i for i in range(1, 24, 1)], type=list)
    parser.add_argument("--target_name", default="akronAbrasion", type=str)
    parser.add_argument("--new_samples", default=200, type=int)
    args = parser.parse_args()

    gmm_generator = Vsg.GmmVSG(
        covariance_type=args.covariance_type,
        optimizer=args.optimizer,
        result_retain=args.result_retain,
        seed=args.seed
    )
    gmm_generator.fit(
        args.file_path,
        args.feature_start_idx,
        args.feature_end_idx,
        args.n_components,
        args.target_name
    )
    new_df_data = gmm_generator.samples(args.new_samples)
    print(new_df_data)
